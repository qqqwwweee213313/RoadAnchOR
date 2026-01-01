import copy
from math import pi, cos, sin

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmcv.models.builder import HEADS, build_loss 
from mmcv.models.dense_heads import DETRHead
from mmcv.utils import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.core.bbox.builder import build_assigner, build_sampler
from mmcv.core.bbox.coder import build_bbox_coder
from mmcv.models.utils.transformer import inverse_sigmoid
from mmcv.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.models.bricks import Linear
from mmcv.models.utils import bias_init_with_prob, xavier_init
from mmcv.core.utils import (multi_apply, multi_apply, reduce_mean)
from mmcv.models.bricks.transformer import build_transformer_layer_sequence

from mmcv.core.bbox.util import normalize_bbox
from mmcv.models.vad_utils.traj_lr_warmup import get_traj_warmup_loss_weight
from mmcv.models.vad_utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)

# Planning decoders: plain, score-conditioned, and score + heading variants.
from mmcv.models.modules.roadanchor.planning_decoder import PlanningDecoder, safe_atan2
from mmcv.models.modules.roadanchor.planning_decoder_score import PlanningDecoderScore
from mmcv.models.modules.roadanchor.planning_decoder_score_yaw import PlanningDecoderScoreYaw
from mmcv.models.modules.roadanchor.layers.fourier_embedding import FourierEmbedding
from mmcv.models.modules.roadanchor.layers.map_encoder import MapEncoder
from mmcv.models.modules.roadanchor.transformer import PlutoTransformerEncoderLayer
from mmcv.models.modules.roadanchor.layers.agent_encoder import AgentEncoder

# Latent Space Alignment Utils
from mmcv.models.modules.roadanchor.latent_alignment.latent_alignment_utils import latent_alignment_loss_segments, latent_alignment_loss_nn_with_radius

from typing import Union, List, Dict

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


@HEADS.register_module()
class RoadAnchorHead(DETRHead):
    """Planning head for the road-adaptive anchoring axis.

    On top of a BEV perception backbone (3-D detection + online vectorised map)
    this head predicts a multi-modal ego trajectory that is anchored on
    *predicted* road centerlines.  Four ingredients define the axis:

    (1) Road-Anchor Scoring (RAS).  An auxiliary head regresses, for every
        predicted centerline candidate, an importance target I in [0, 1]:

            A     = exp(-ADE / W)
                    GT future trajectory vs. candidate, mean nearest-vertex
                    distance
            B     = mean_k 1[hit_k] * exp(-d_k / W)
                    normal of the GT tangent at step k intersected with the
                    candidate; a step that does not intersect contributes 0
            C_far = exp(-d_far / W)
                    far goal waypoint vs. candidate, nearest-vertex distance

            I = A * B * C_far      if score_target_use_cfar (default)
            I = A * B              if score_target_use_cfar=False

        W = score_lane_w (3.5 m) is the only scale constant and is shared by
        the three terms.  Because the target is a product, a far goal that
        falls outside the BEV range drives C_far -- and therefore the whole
        target -- to ~0 for every candidate; that is why the term can be
        switched off.  Candidates are the top score_cand_lines map queries by
        centerline-class probability, resampled to score_num_pts points.  The
        target is recomputed on the fly under no_grad (no cache), so the same
        function is used whether the candidates are GT or predicted.

    (2) Anchor-Ranked Attention (ARA).  The reference lines handed to the
        planning decoder are the top-R candidates by *predicted* score among
        those whose centerline confidence exceeds 0.6, rather than the R
        centerlines nearest to the ego.  Slots below the threshold are padded
        with zero geometry; if no candidate passes, the single highest-scoring
        one is forced in, because an all-padding slate makes the
        reference-to-reference attention produce NaN.  Selection ranking uses
        detached scores (top-k is not differentiable) and the injected scores
        are detached as well, so the score head is trained by the auxiliary
        regression only while the consumer-side embedding still learns from the
        planning gradient.  Without goal points the head falls back to the
        nearest-to-ego selection and injects nothing.

    (3) Delta parameterisation.  With plan_traj_cumsum=True the decoder's
        location head is read as a per-step displacement and integrated
        (cumsum over time) inside the head.  This inherits the "offsets are
        mostly forward" prior and removes the backward-drift frames seen with
        direct absolute-point regression.  Everything downstream (losses, top-1
        selection, controller, collision, score, cost map, visualisation,
        evaluation) still receives absolute points.

    (4) Heading regression.  The planning decoder emits
        [x, y, cos(theta), sin(theta)] per waypoint.  The heading target is
        derived inside the head from the GT cumulative trajectory by finite
        differences, so the shared data loader stays untouched; steps whose
        displacement is below static_thresh (0.5 m, i.e. 1.0 m/s at the 0.5 s
        sampling interval) get zero heading weight.  Heading uses the same
        smooth L1 and the same weight as the position term and is exposed as
        ra_ego_yaws, while ra_ego_trajs keeps the plain 2-channel position
        layout, so inference and closed-loop driving are unaffected.

    (5) Road-Aware Collision (RAC).  With plan_col_v7=True the head forwards
        the GT trajectory, the object extents (w, l) and the object headings to
        the collision loss, which approximates each object by a chain of discs
        and applies a linear hinge (see PlanCollisionLossV7).

    Structural differences with respect to the upstream VAD-style planning head
    this file derives from:

      (a) No top-k truncation of planning tokens.  The planning decoder's
          cross-attention memory is the full encoded sequence
          [ego | objects | map | goal] instead of a single scene token, so the
          key/value length is 1 + num_query + map_num_vec + 2.  The only
          remaining top-k is the reference-line selection.
      (b) No static/dynamic agent split: one agent set and one map vector set,
          as in VAD.
      (c) MapEncoder receives the inverted padding mask, matching the
          PointsEncoder convention (1 = valid).
    """
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 map_bbox_coder=None,
                 map_num_query=900,
                 map_num_classes=3,
                 map_num_vec=20,
                 map_num_pts_per_vec=2,
                 map_num_pts_per_gt_vec=2,
                 map_query_embed_type='all_pts',
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v0',
                 map_dir_interval=1,
                 map_code_size=None,
                 map_code_weights=None,
                loss_map_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_map_pts=dict(
                    type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0
                 ),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 tot_epoch=None,
                 use_traj_lr_warmup=False,
                 motion_decoder=None,
                 use_pe=False,
                 motion_det_score=None,
                 map_thresh=0.5,
                 dis_thresh=0.2,
                 pe_normalization=True,
                 ego_fut_mode=3,
                #  loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                #  loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanAgentDisLoss', loss_weight=0.1),
                #  loss_plan_dir=dict(type='PlanMapThetaLoss', loss_weight=0.1),
                 query_thresh=None,
                 query_use_fix_pad=None,
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 latent_space_alignment_flag=False,
                 use_map_token=False,
                 with_planning=True,
                 # -- Score auxiliary head --------------------------------------
                 score_cand_lines=40,      # max predicted centerline candidates to score
                 score_num_pts=100,        # points per candidate after arc-length resampling
                 score_hidden=128,         # hidden width of the score MLP
                 score_loss_weight=1.0,    # auxiliary loss weight
                 score_lane_w=3.5,         # W: single shared scale for A / B / C_far
                 # Keep the far-goal term in the score target?
                 #   True  = I = A * B * C_far  (default)
                 #   False = I = A * B          (drops C_far, which otherwise collapses
                 #           the whole target to ~0 once the far goal leaves pc_range)
                 score_target_use_cfar=True,
                 # Collision-loss v4 argument path:
                 #   False = call the plain PlanCollisionLoss (default)
                 #   True  = also forward the GT trajectory, object (w, l) and yaw that
                 #           the v4 loss needs. Pairs with loss_plan_col.type in the config.
                 plan_col_v4=False,
                 # Collision-loss v7 argument path:
                 #   False = keep the branch above (default)
                 #   True  = v7 loss. Its argument contract is identical to v4, so both
                 #           share the same branch. Do not enable v4 and v7 together:
                 #           the loss object itself comes from loss_plan_col.type, the
                 #           flag only selects how arguments are forwarded.
                 plan_col_v7=False,
                 # Ego trajectory parameterisation:
                 #   False = read the location head output as absolute waypoints
                 #           (default; no cumsum on the prediction path)
                 #   True  = read it as per-step displacements and integrate them
                 #           (cumsum) inside the head. Losses and every consumer
                 #           (top-1 selection, controller, collision, score, cost map,
                 #           visualisation, evaluation) still receive absolute points.
                 plan_traj_cumsum=False,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.tot_epoch = tot_epoch
        self.use_traj_lr_warmup = use_traj_lr_warmup
        self.motion_decoder = motion_decoder
        self.use_pe = use_pe
        self.motion_det_score = motion_det_score
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.pe_normalization = pe_normalization
        self.ego_fut_mode = ego_fut_mode
        self.query_thresh = query_thresh
        self.query_use_fix_pad = query_use_fix_pad
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        # When False, forward/loss skip planning entirely (ego trajectory,
        #   ra_planning_decoder and the three planning losses). Detection, map and
        #   centerline perception stay active.
        self.with_planning = with_planning

        if loss_traj_cls['use_sigmoid'] == True:
            self.traj_num_cls = 1
        else:
          self.traj_num_cls = 2

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.map_bbox_coder = build_bbox_coder(map_bbox_coder)
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        map_num_query = map_num_vec * map_num_pts_per_vec
        self.map_num_query = map_num_query
        self.map_num_classes = map_num_classes
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        self.map_dir_interval = map_dir_interval
        
        # (a) top-k removal: the agent / traffic-light / map top-k counts are gone.
        #   Planning tokens use every object query (num_query) and every map polyline
        #   (map_num_vec). The only remaining top-k is the reference-line selection.
        self.ref_line_topk_num = ego_fut_mode

        pluto_radius_m = max(self.pc_range)

        self.radius = pluto_radius_m
        self.mode_interval = self.radius / self.ego_fut_mode

        # (b) No static/dynamic split: a single agent set, as in VAD.

        self.latent_space_alignment_flag = latent_space_alignment_flag
        self.use_map_token = use_map_token
        
        # Print latent_space_alignment_flag value in yellow color
        print("\033[93m" + f"latent_space_alignment_flag: {self.latent_space_alignment_flag}" + "\033[0m")
        print("\033[93m" + f"use_map_token              : {self.use_map_token}" + "\033[0m")

        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1

        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get('class_weight', None)
        if map_class_weight is not None and (self.__class__ is RoadAnchorHead):
            assert isinstance(map_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(map_class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            map_bg_cls_weight = loss_map_cls.get('bg_cls_weight', map_class_weight)
            assert isinstance(map_bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(map_bg_cls_weight)}.'
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({'class_weight': map_class_weight})
            if 'bg_cls_weight' in loss_map_cls:
                loss_map_cls.pop('bg_cls_weight')
            self.map_bg_cls_weight = map_bg_cls_weight
        
        self.traj_bg_cls_weight = 0

        super(RoadAnchorHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)
        
        if kwargs['train_cfg'] is not None:
            assert 'map_assigner' in kwargs['train_cfg'], 'map assigner should be provided '\
                'when train_cfg is set.'
            map_assigner = kwargs['train_cfg']['map_assigner']
            assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_bbox['loss_weight'] == map_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_map_iou['loss_weight'] == map_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be' \
                'exactly the same.'

            self.map_assigner = build_assigner(map_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.map_sampler = build_sampler(sampler_cfg, context=self)
        
        self.loss_traj = build_loss(loss_traj)
        self.loss_traj_cls = build_loss(loss_traj_cls)
        self.loss_map_bbox = build_loss(loss_map_bbox)
        self.loss_map_cls = build_loss(loss_map_cls)
        self.loss_map_iou = build_loss(loss_map_iou)
        self.loss_map_pts = build_loss(loss_map_pts)
        self.loss_map_dir = build_loss(loss_map_dir)
        # self.loss_plan_reg = build_loss(loss_plan_reg)
        # self.loss_plan_bound = build_loss(loss_plan_bound)
        self.loss_plan_col = build_loss(loss_plan_col)
        # self.loss_plan_dir = build_loss(loss_plan_dir)

        # -- Score auxiliary head ---------------------------------------------
        #   Built after super().__init__() so that nn.Module is already initialised.
        self.score_cand_lines = score_cand_lines
        self.score_num_pts = score_num_pts
        self.score_loss_weight = score_loss_weight
        self.score_lane_w = score_lane_w
        self.score_target_use_cfar = bool(score_target_use_cfar)
        self.plan_col_v4 = bool(plan_col_v4)
        self.plan_col_v7 = bool(plan_col_v7)
        self.plan_traj_cumsum = bool(plan_traj_cumsum)
        if self.plan_traj_cumsum:
            print("\033[92m[RoadAnchorHead] ego trajectory = per-step delta + cumsum: "
                  "the location head output is read as offsets and integrated inside the "
                  "head; losses and consumers still receive absolute points.\033[0m")
        if self.plan_col_v4:
            print("\033[92m[RoadAnchorHead] planning collision loss v4 "
                  f"({type(self.loss_plan_col).__name__}): class-table multi-disc "
                  "approximation + linear hinge. GT trajectory, object (w, l) and yaw forwarded.\033[0m")
        if self.plan_col_v7:
            print("\033[92m[RoadAnchorHead] planning collision loss v7 "
                  f"({type(self.loss_plan_col).__name__}): disc filling (r=w/2, n=ceil(l/w)), "
                  "one shared margin D=3 m, pedestrian longitudinal gate. Same contract as v4.\033[0m")
        # point-wise MLP → max-pool over P → per-centerline feature → score
        #   input 4ch = [ norm(p), norm(p - g_far) ]
        self.score_pt_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True), nn.Linear(64, score_hidden))
        self.score_head_mlp = nn.Sequential(
            nn.Linear(score_hidden, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        print("\033[92m[RoadAnchorHead] Centerline Score head "
              f"(target = {'A*B*C_far' if self.score_target_use_cfar else 'A*B (no C_far)'}, "
              f"W={score_lane_w}, cand={score_cand_lines}, "
              f"num_pts={score_num_pts}, w={score_loss_weight}) enabled; "
              "reference lines = top-6 predicted score among candidates with conf > 0.6 "
              "(no geometric dedup, forced top-1 if none pass), injected detached into "
              "PlanningDecoderScoreYaw; nearest-to-ego fallback when goals are absent.\033[0m")

        # -- Smoke-test diagnostics switch (no new hyper-parameter or config key) --
        #   With RA_YAW_SMOKE_DIAG=1 the head prints (b) the per-class map conf>0.5 pass
        #   rate and (c) the teacher-forced (r*, m*) histogram. Training loss is unaffected.
        import os as _os
        self._yaw_smoke_diag = _os.environ.get('RA_YAW_SMOKE_DIAG', '0') == '1'
        self._yaw_diag_interval = int(_os.environ.get('RA_YAW_SMOKE_DIAG_INTERVAL', '5'))
        self._yaw_diag_step = 0
        self._yaw_diag_rm_hist = None
        if self._yaw_smoke_diag:
            print("\033[95m[RoadAnchorHead] ★SMOKE DIAG ON★ "
                  f"(interval={self._yaw_diag_interval}) — map conf pass-rate + (r*,m*) hist\033[0m")
        print("\033[95m[RoadAnchorHead] heading (yaw) regression enabled: "
              "traj=[x, y, cos(theta), sin(theta)], zero heading weight on static steps "
              "(<0.5 m), same smooth L1 and weight as the position term, no velocity head. "
              "Inference and driving use the two position channels only.\033[0m")

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)


        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        motion_num_pred = (num_motion_decoder_layers + 1) if \
            self.as_two_stage else num_motion_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.traj_branches = nn.ModuleList(
                [traj_branch for _ in range(motion_num_pred)])
            self.traj_cls_branches = nn.ModuleList(
                [traj_cls_branch for _ in range(motion_num_pred)])
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            if self.map_query_embed_type == 'all_pts':
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                    self.embed_dims * 2)
            elif self.map_query_embed_type == 'instance_pts':
                self.map_query_embedding = None
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)
        
        if self.motion_decoder is not None:
            self.motion_decoder = build_transformer_layer_sequence(self.motion_decoder)
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)	
            self.motion_mode_query.weight.requires_grad = True
            if self.use_pe:
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
        else:
            raise NotImplementedError('Not implement yet')

        # self.lane_encoder = LaneNet(256, 128, 3)
        if self.use_pe:
            self.pos_mlp = nn.Linear(2, self.embed_dims)

        self.ego_query = nn.Embedding(1, self.embed_dims)	
                
        # positional / goal Fourier embeddings for the planning tokens
        self.ra_pos_emb = FourierEmbedding(2, self.embed_dims, 64)
        self.ra_goal_emb = FourierEmbedding(2, self.embed_dims, 64)
        
        if self.use_map_token:
            self.ra_map_encoder = MapEncoder(dim=self.embed_dims, polygon_channel=6)
        
        # Score-conditioned planning decoder with a heading regression head.
        #   With ref_score=None only the score injection is skipped; the heading head
        #   still runs. traj is (bs, R, M, T, 4) = [x, y, cos(theta), sin(theta)].
        self.ra_planning_decoder = PlanningDecoderScoreYaw(
            num_mode=self.ego_fut_mode,
            decoder_depth=3,
            dim=self.embed_dims,
            num_heads=8,
            mlp_ratio=4,
            dropout=0.1,
            future_steps=self.fut_ts,
        )
        
        if self.latent_space_alignment_flag:
            # ============================================
            # Agent encoder (latent-space alignment branch)
            # ============================================
            self.agent_encoder = AgentEncoder(
                # history_channel=history_channel, # 9
                dim=self.embed_dims,
                hist_steps=6,
                num_classes=self.num_classes,
                # drop_path=drop_path,
                # use_ego_history=use_ego_history,
                # state_attn_encoder=state_attn_encoder,
                # state_dropout=state_dropout,
            )
        
            # ego state information
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(18, self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(inplace=True),
            )
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
            
            # ✅ Cross-Attention layer
            self.agent_can_cross_attn = nn.MultiheadAttention(
                embed_dim=self.embed_dims,
                num_heads=4,
                batch_first=True
            )
            self.agent_can_norm = nn.LayerNorm(self.embed_dims)
        
        drop_path = 0.2
        encoder_depth = 4
    
        self.ra_encoder_blocks = nn.ModuleList(
            PlutoTransformerEncoderLayer(dim=self.embed_dims, num_heads=8, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.ra_norm = nn.LayerNorm(self.embed_dims)

        self.agent_fus_mlp = nn.Sequential(
            nn.Linear(self.fut_mode*2*self.embed_dims, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_map_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.map_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_traj_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.traj_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.map_reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.map_reg_branches[0][-1].bias.data[2:], 0.)
        if self.motion_decoder is not None:
            for p in self.motion_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(self.motion_mode_query.weight)
            if self.use_pe:
                xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)
        # for p in self.lane_encoder.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        if self.use_pe:
            xavier_init(self.pos_mlp, distribution='uniform', bias=0.)

    def prepare_gt_for_planning(self, gt_bboxes_list, gt_labels_list, device):
        """
        Prepare GT bounding boxes in the planning input format.
        
        Returns:
            gt_bboxes_m: (B, max_N, code_size) padded GT boxes in meters
            gt_labels: (B, max_N) GT labels
            gt_masks: (B, max_N) valid mask
        """
        
        if not isinstance(gt_bboxes_list, list):
            gt_bboxes_list = [gt_bboxes_list]
            gt_labels_list = [gt_labels_list]
        
        B = len(gt_bboxes_list)
        max_N = max(len(gt) for gt in gt_bboxes_list) if gt_bboxes_list else 0
        
        if max_N == 0:
            # No GT boxes
            return (torch.zeros(B, 1, self.code_size, device=device),
                    torch.zeros(B, 1, dtype=torch.long, device=device),
                    torch.zeros(B, 1, dtype=torch.bool, device=device))
        
        # Pad to max_N
        gt_bboxes_m = torch.zeros(B, max_N, self.code_size, device=device)
        gt_labels = torch.zeros(B, max_N, dtype=torch.long, device=device)
        gt_masks = torch.ones(B, max_N, dtype=torch.bool, device=device)
        
        for b in range(B):
            gt_b = gt_bboxes_list[b]  # (Ni, code_size)
            label_b = gt_labels_list[b]  # (Ni,)
            
            if isinstance(gt_b, torch.Tensor):
                # Already a tensor
                gt_b_tensor = gt_b
            else:
                # LiDARInstance3DBoxes or similar object
                gt_b_tensor = gt_b.tensor
            
            # ✅ Get actual GT count for this batch
            N_gt = len(gt_b_tensor)
            
            if N_gt == 0:
                continue
            
            normalized_gt_b = normalize_bbox(gt_b_tensor, self.pc_range)

            # ✅ Fill only N_gt elements (not all max_N!)
            gt_bboxes_m[b, :N_gt] = normalized_gt_b[:, :self.code_size]
            gt_labels[b, :N_gt] = label_b[:N_gt]
            gt_masks[b, :N_gt] = False  # ✅ False = valid
        
        return gt_bboxes_m, gt_labels, gt_masks
    
    def _get_padded_gt_query_inputs(self, 
                                    gt_features,
                                    gt_bboxes_m,
                                    gt_masks):
        B, N_gt, D = gt_features.shape
            
        if N_gt < self.num_query:
            # number of padding slots
            num_padding = self.num_query - N_gt
            
            # 1. Feature padding
            padding_features = torch.zeros(
                B, num_padding, D, 
                device=gt_features.device, 
                dtype=gt_features.dtype
            )
            gt_features_padded = torch.cat([gt_features, padding_features], dim=1)  # (B, num_query, D)
            
            # 2. BBox padding (for motion_pos)
            padding_bboxes = torch.zeros(
                B, num_padding, gt_bboxes_m.shape[-1],
                device=gt_bboxes_m.device,
                dtype=gt_bboxes_m.dtype
            )
            gt_bboxes_m_padded = torch.cat([gt_bboxes_m, padding_bboxes], dim=1)  # (B, num_query, code_size)
            
            # 3. Mask padding
            padding_masks = torch.ones(
                B, num_padding,
                device=gt_masks.device,
                dtype=torch.bool
            )
            gt_masks_padded = torch.cat([gt_masks, padding_masks], dim=1)  # (B, num_query)
            
        elif N_gt > self.num_query:
            gt_features_padded = gt_features[:, :self.num_query]  # (B, num_query, D)
            gt_bboxes_m_padded = gt_bboxes_m[:, :self.num_query]
            gt_masks_padded = gt_masks[:, :self.num_query]
            
        else:
            gt_features_padded = gt_features
            gt_bboxes_m_padded = gt_bboxes_m
            gt_masks_padded = gt_masks
    
        return gt_features_padded, gt_bboxes_m_padded, gt_masks_padded
    
    # (a) top-k removal: these three selection helpers lost their call sites and
    #   · _select_gt_topk                                          (GT agent class-wise top-k)
    #   · select_topk_query_by_class_with_threshold                (predicted agent class-wise top-k)
    #   · select_topk_polyline_query_excluding_classes_by_closest_point (map top-k)
    #   were removed with them. The only selection helper kept is the centerline
    #   reference-line one, select_topk_polyline_query_by_closest_point(topk=ref_line_topk_num).
    def extract_non_centerline_polylines_batch(self,
                                               centerline_pts_list,
                                               centerline_labels_list,
                                               centerline_label=3,
                                               max_lines=20,
                                               fixed_points=None):
        """
        Extract every map polyline except the centerline class (label=centerline_label).

        Args:
            centerline_pts_list: list of B centerline point sets (map_gt_bboxes_list)
            centerline_labels_list: list of B label arrays (map_gt_labels_list)
            centerline_label: label that marks centerlines (default: 3)
            max_lines: maximum number of lines to extract
            fixed_points: points sampled per line (None -> self.map_num_pts_per_vec)

        Returns:
            other_lines: (B, max_lines, fixed_points, 2)
            other_mask:  (B, max_lines)  # False=valid, True=padding
        """
        if fixed_points is None:
            fixed_points = self.map_num_pts_per_vec

        batch = len(centerline_pts_list)
        device = self.device if hasattr(self, 'device') else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        other_lines = torch.zeros(
            (batch, max_lines, fixed_points, 2),
            dtype=torch.float32,
            device=device
        )
        other_mask = torch.ones(
            (batch, max_lines),
            dtype=torch.bool,
            device=device
        )

        for b in range(batch):
            centerline_pts = centerline_pts_list[b]
            centerline_labels = centerline_labels_list[b]

            # select labels other than centerline
            if isinstance(centerline_labels, torch.Tensor):
                target_mask = (centerline_labels != centerline_label)
                target_indices = torch.where(target_mask)[0]
            else:
                centerline_labels = np.array(centerline_labels)
                target_indices = np.where(centerline_labels != centerline_label)[0]
                target_indices = torch.from_numpy(target_indices)

            if len(target_indices) == 0:
                continue

            line_list = centerline_pts.instance_list

            lines_batch = []
            distances = []  # sorted by minimum distance to the ego origin (0,0), nearest first

            for idx in target_indices:
                if idx >= len(line_list):
                    continue

                line = line_list[idx]

                if isinstance(line, torch.Tensor):
                    line_tensor = line.to(device)
                elif hasattr(line, 'coords'):
                    line_tensor = torch.tensor(line.coords, dtype=torch.float32, device=device)
                elif isinstance(line, np.ndarray):
                    line_tensor = torch.from_numpy(line).float().to(device)
                else:
                    try:
                        line_tensor = torch.tensor(line, dtype=torch.float32, device=device)
                    except Exception:
                        continue

                if line_tensor.ndim == 1:
                    line_tensor = line_tensor.reshape(-1, 2)
                elif line_tensor.shape[-1] > 2:
                    line_tensor = line_tensor[:, :2]

                if len(line_tensor) == 0:
                    continue

                # distance from (0,0) to the closest point
                point_distances = torch.sqrt(line_tensor[:, 0] ** 2 + line_tensor[:, 1] ** 2)
                min_distance = point_distances.min().item()

                lines_batch.append(line_tensor)
                distances.append(min_distance)

            if not lines_batch:
                continue

            distances_tensor = torch.tensor(distances, device=device)
            _, sorted_indices = torch.sort(distances_tensor)  # nearest first

            num_selected = min(max_lines, len(sorted_indices))
            selected_indices = sorted_indices[:num_selected]

            selected_lines = [lines_batch[i] for i in selected_indices]

            interpolated = self._interpolate_lanes_batch_torch(
                selected_lines, fixed_points, device
            )

            num_valid = len(interpolated)
            other_lines[b, :num_valid] = interpolated
            other_mask[b, :num_valid] = False

        return other_lines, other_mask
    
    def extract_reference_centerlines_batch(self,
                                            centerline_pts_list,
                                            centerline_labels_list,
                                            target_label=3,
                                            max_lines=3,
                                            fixed_points=None):
        """
        Batch-optimized reference centerline extraction (distance-based selection)
        
        Args:
            centerline_pts_list: list of B centerline point sets
            centerline_labels_list: list of B label arrays
            target_label: label to extract (default: 3 for centerline)
            max_lines: maximum lines per batch
            fixed_points: number of points per line
        
        Returns:
            reference_lines: torch.Tensor (B, max_lines, fixed_points, 2)
            reference_mask: torch.Tensor (B, max_lines) bool mask
        """
        if fixed_points is None:
            fixed_points = self.map_num_pts_per_vec
        
        batch = len(centerline_pts_list)
        device = self.device if hasattr(self, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize output tensors on GPU
        reference_lines = torch.zeros(
            (batch, max_lines, fixed_points, 2),
            dtype=torch.float32,
            device=device
        )
        reference_mask = torch.ones(
            (batch, max_lines),
            dtype=torch.bool,
            device=device
        )
        
        # ============================================
        # Batch processing with minimal conversions
        # ============================================
        for b in range(batch):
            centerline_pts = centerline_pts_list[b]
            centerline_labels = centerline_labels_list[b]
            
            # ✅ Fast label filtering
            if isinstance(centerline_labels, torch.Tensor):
                target_mask = (centerline_labels == target_label)
                target_indices = torch.where(target_mask)[0]
            else:
                centerline_labels = np.array(centerline_labels)
                target_indices = np.where(centerline_labels == target_label)[0]
                target_indices = torch.from_numpy(target_indices)
            
            if len(target_indices) == 0:
                continue
            
            # ✅ Extract lines and compute distances
            line_list = centerline_pts.instance_list
            
            lines_batch = []
            distances = []
            
            for idx in target_indices:
                if idx >= len(line_list):
                    continue
                
                line = line_list[idx]
                
                # Fast conversion
                if isinstance(line, torch.Tensor):
                    line_tensor = line.to(device)
                elif hasattr(line, 'coords'):
                    line_tensor = torch.tensor(line.coords, dtype=torch.float32, device=device)
                elif isinstance(line, np.ndarray):
                    line_tensor = torch.from_numpy(line).float().to(device)
                else:
                    try:
                        line_tensor = torch.tensor(line, dtype=torch.float32, device=device)
                    except:
                        continue
                
                # Ensure 2D
                if line_tensor.ndim == 1:
                    line_tensor = line_tensor.reshape(-1, 2)
                elif line_tensor.shape[-1] > 2:
                    line_tensor = line_tensor[:, :2]
                
                if len(line_tensor) == 0:
                    continue
                
                # ============================================
                # ✅ Compute distance to ego (0, 0)
                # ============================================
                # Distance from each point to (0, 0)
                point_distances = torch.sqrt(line_tensor[:, 0]**2 + line_tensor[:, 1]**2)
                
                # Use minimum distance (closest point on this centerline)
                min_distance = point_distances.min().item()
                
                lines_batch.append(line_tensor)
                distances.append(min_distance)
            
            if not lines_batch:
                continue
            
            # ============================================
            # ✅ Sort by distance and select top-k closest
            # ============================================
            distances_tensor = torch.tensor(distances, device=device)
            _, sorted_indices = torch.sort(distances_tensor)  # ascending order
            
            # Select max_lines closest centerlines
            num_selected = min(max_lines, len(sorted_indices))
            selected_indices = sorted_indices[:num_selected]
            
            selected_lines = [lines_batch[i] for i in selected_indices]
            
            # ✅ Vectorized interpolation
            interpolated = self._interpolate_lanes_batch_torch(
                selected_lines, fixed_points, device
            )
            
            # Store results
            num_valid = len(interpolated)
            reference_lines[b, :num_valid] = interpolated
            reference_mask[b, :num_valid] = False
        
        return reference_lines, reference_mask
    
    def _interpolate_lanes_batch_torch(self, lines_list, num_points, device):
        """
        ✅ GPU-accelerated batch interpolation
        
        Args:
            lines_list: list of tensors [(N1, 2), (N2, 2), ...]
            num_points: target number of points
            device: torch device
        
        Returns:
            interpolated: torch.Tensor (num_lines, num_points, 2)
        """
        num_lines = len(lines_list)
        
        # ✅ Pre-allocate output tensor
        interpolated = torch.zeros(num_lines, num_points, 2, device=device, dtype=torch.float32)
        
        for i, line in enumerate(lines_list):
            # line: (N, 2)
            N = len(line)
            
            if N == 0:
                # Keep zeros
                continue
            
            if N == 1:
                # Single point, repeat
                interpolated[i] = line.repeat(num_points, 1)
                continue
            
            # ✅ Compute cumulative distances (GPU)
            diffs = line[1:] - line[:-1]  # (N-1, 2)
            segment_lengths = torch.norm(diffs, dim=1)  # (N-1,)
            cumsum = torch.cat([
                torch.zeros(1, device=device),
                torch.cumsum(segment_lengths, dim=0)
            ])  # (N,)
            
            total_length = cumsum[-1]
            
            if total_length < 1e-6:
                # Degenerate line, all points same
                interpolated[i] = line[0:1].repeat(num_points, 1)
                continue
            
            # ✅ Target distances (evenly spaced)
            target_dists = torch.linspace(
                0, total_length, num_points, device=device
            )  # (num_points,)
            
            # ✅ Find segments for each target point (vectorized)
            indices = torch.searchsorted(cumsum, target_dists, right=False)
            indices = torch.clamp(indices, 0, N - 2)  # Valid segment indices
            
            # ✅ Interpolate within segments (vectorized)
            t = (target_dists - cumsum[indices]) / (segment_lengths[indices] + 1e-8)
            t = torch.clamp(t, 0, 1).unsqueeze(-1)  # (num_points, 1)
            
            p0 = line[indices]       # (num_points, 2)
            p1 = line[indices + 1]   # (num_points, 2)
            
            interpolated_line = p0 * (1 - t) + p1 * t  # (num_points, 2)
            interpolated[i] = interpolated_line
        
        return interpolated  # (num_lines, num_points, 2)
    
    def _prepare_agent_encoder_inputs(
        self,
        gt_abs_fut_traj,
        gt_fut_traj,
        gt_fut_mask,
        gt_bboxes_list,
        gt_labels,
        gt_attr_labels,
        device
    ):
        """
        Prepare inputs for AgentEncoder from GT future trajectory data
        
        Args:
            gt_abs_fut_traj: (B, N, T, 2) - absolute future positions
            gt_fut_traj: (B, N, T, 2) - offset future trajectory
            gt_fut_mask: (B, N, T) - valid timestep mask (1=valid)
            gt_bboxes_list: list of GT bboxes
            gt_labels: (B, N) - class labels
            gt_attr_labels: list of (N_i, T*3) - contains future_yaw_offset
            device: torch device
        
        Returns:
            data: dict with keys [position, heading, velocity, shape, category, valid_mask]
        """
        B, N, T, _ = gt_abs_fut_traj.shape
        
        # ============================================
        # 1. Position: absolute coordinates (already prepared)
        # ============================================
        position = gt_abs_fut_traj  # (B, N, T, 2)
        
        # ============================================
        # 2. Heading: future_yaw_offset -> absolute yaw
        # ============================================
        # future_yaw_offset: the last T values of attr_labels
        heading = torch.zeros(B, N, T, device=device)
        
        for b in range(B):
            attr = gt_attr_labels[b]  # (N_i, T*3 + ...)
            bbox = gt_bboxes_list[b].tensor.to(device)
            N_i = len(attr)
            
            if N_i == 0:
                continue
            
            # Extract future_yaw_offset
            # attr structure: [fut_traj (T*2), fut_mask (T), fut_goal (1), lcf_feat (9), yaw_offset (T)]
            yaw_offset_start = self.fut_ts * 3 + 1 + 9
            yaw_offset = attr[:, yaw_offset_start:yaw_offset_start + self.fut_ts]  # (N_i, T)
            
            # Current yaw from bbox (index 6)
            current_yaw = bbox[:, 6]  # (N_i,)
            
            # Compute absolute yaw
            # option 1: cumulative sum (when the offset is a delta yaw)
            yaw_cumsum = yaw_offset.cumsum(dim=-1)  # (N_i, T)
            absolute_yaw = current_yaw.unsqueeze(-1) + yaw_cumsum  # (N_i, T)
            
            heading[b, :N_i] = absolute_yaw
        
        # ============================================
        # 3. Velocity: offset -> velocity (0.5 s interval)
        # ============================================
        dt = 0.5  # 0.5 seconds
        velocity = gt_fut_traj / dt  # (B, N, T, 2)
        
        # ============================================
        # 4. Shape: take width and length from the GT bbox
        # ============================================
        shape = torch.zeros(B, N, T, 2, device=device)
        
        for b in range(B):
            bbox = gt_bboxes_list[b].tensor.to(device)
            N_i = len(bbox)
            
            if N_i == 0:
                continue
            
            # bbox structure: [x, y, z, width, length, height, yaw, vx, vy]
            width = bbox[:, 3]   # (N_i,)
            length = bbox[:, 4]  # (N_i,)
            
            # Broadcast to all timesteps
            shape[b, :N_i, :, 0] = width.unsqueeze(-1).expand(-1, T)
            shape[b, :N_i, :, 1] = length.unsqueeze(-1).expand(-1, T)
        
        # ============================================
        # 5. Category: GT labels
        # ============================================
        category = torch.full((B, N), 0, dtype=torch.long, device=device)
        for b in range(B):
            N_i = len(gt_bboxes_list[b].tensor)
            if N_i > 0:
                # gt_labels[b] has shape (N_gt,)
                category[b, :N_i] = gt_labels[b, :N_i]
        
        # ============================================
        # 6. Valid Mask: future timestep mask
        # ============================================
        valid_mask = gt_fut_mask.bool()  # (B, N, T)
        
        # ============================================
        # 7. Build the data dictionary
        # ============================================
        data = {
            'agent': {
                'position': position,      # (B, N, T, 2)
                'heading': heading,        # (B, N, T)
                'velocity': velocity,      # (B, N, T, 2)
                'shape': shape,            # (B, N, T, 2)
                'category': category,      # (B, N)
                'valid_mask': valid_mask,  # (B, N, T)
            }
        }
        
        return data
    
    def m2m_input_query_generation(self, 
                                   can_bus_infos,
                                   gt_bboxes_list,
                                   gt_labels_list,
                                   gt_attr_labels,
                                   ego_goal_points,
                                   map_gt_bboxes_list,
                                   map_gt_labels_list):
        
        device = ego_goal_points.device
        # extract motion info from can_bus
        can_bus_motion_only = torch.zeros_like(can_bus_infos)
        can_bus_motion_only[:, 7:16] = can_bus_infos[:, 7:16] # vel, accel, rot_rate
        
        can_bus_embed = self.can_bus_mlp(can_bus_motion_only)
        
        # GT Embedding
        # gt_masks: valid: False
        gt_bboxes_m, gt_labels, gt_masks = self.prepare_gt_for_planning(gt_bboxes_list, gt_labels_list, device)
        batch_size = gt_bboxes_m.shape[0]
        
        # -------------------------
        # (1) build the GT future trajectory
        # -------------------------
        # gt_attr_labels_list: list of (N_i, fut_ts*3)
        B = len(gt_attr_labels)
        fut_ts = self.fut_ts

        # pad trajectories (B, N, T*3)
        # max_N = gt_features_padded.shape[1]
        max_N = gt_bboxes_m.shape[1]
        gt_fut_traj = torch.zeros(B, max_N, fut_ts, 2, device=device)
        gt_abs_fut_traj = torch.zeros(B, max_N, fut_ts, 2, device=device)
        gt_fut_mask = torch.zeros(B, max_N, fut_ts, device=device) # True: valid

        for b in range(B):
            attr = gt_attr_labels[b]  # (N_i, T*3)
            bbox = gt_bboxes_list[b].tensor.to(device)  # (N_i, 2
            
            bbox_vel = bbox[:, -2:] # vx, vy
            
            N_i = len(attr)
            if N_i == 0:
                continue
            
            # [future_track_offset (frames*2), future_mask_offset (frames), 
            #  gt_fut_goal (1), agent_lcf_feat (9), future_yaw_offset (frames)]
            
            traj = attr[:, :fut_ts*2].reshape(N_i, fut_ts, 2)
            accumulated_traj = traj.cumsum(dim=-2)
            bbox_pos = bbox[:, 0:2].unsqueeze(1)
            abs_accumulated_traj = accumulated_traj + bbox_pos
            
            mask = attr[:, fut_ts*2 : fut_ts*3]
            
            gt_abs_fut_traj[b, :N_i] = abs_accumulated_traj
            
            gt_fut_traj[b, :N_i] = traj
            gt_fut_mask[b, :N_i] = mask
            
            num_valid = mask.sum(dim=1) # (N_i,) - valid timesteps per agent
            
            # ============================================
            # Case 1: fully invalid (num_valid == 0)
            # Case 2: partially valid (0 < num_valid < fut_ts)
            # ============================================
            needs_completion = (num_valid < fut_ts)

            if needs_completion.any():
                for agent_idx in needs_completion.nonzero(as_tuple=False).squeeze(1):
                    agent_num_valid = num_valid[agent_idx].item()
                    
                    if agent_num_valid == 0:
                        # ============================================
                        # Case 1: fully invalid -> synthesise from velocity
                        # ============================================
                        pos0 = bbox[agent_idx, 0:2]
                        vel = bbox_vel[agent_idx]
                        
                        dt = 0.5
                        t_range = torch.arange(1, fut_ts + 1, device=device, dtype=torch.float32)
                        
                        # Absolute trajectory
                        synthetic_abs = pos0 + vel * (t_range.unsqueeze(1) * dt)
                        
                        # compute the offset
                        synthetic_offset = torch.zeros(fut_ts, 2, device=device)
                        synthetic_offset[0] = synthetic_abs[0] - pos0
                        synthetic_offset[1:] = synthetic_abs[1:] - synthetic_abs[:-1]
                        
                        gt_abs_fut_traj[b, agent_idx] = synthetic_abs
                        gt_fut_traj[b, agent_idx] = synthetic_offset
                        gt_fut_mask[b, agent_idx] = 1.0
                        
                    else:
                        # ============================================
                        # Case 2: partially valid -> repeat the last value (padding)
                        # ============================================
                        agent_mask = mask[agent_idx]  # (T,)
                        valid_timesteps = torch.where(agent_mask > 0.5)[0]
                        last_valid_t = valid_timesteps[-1].item()
                        
                        # last valid offset
                        last_valid_offset = gt_fut_traj[b, agent_idx, last_valid_t]  # (2,)
                        
                        # copy it onto the remaining steps
                        gt_fut_traj[b, agent_idx, last_valid_t + 1:] = last_valid_offset
                        
                        # update the absolute trajectory as well (cumulative)
                        for t in range(last_valid_t + 1, fut_ts):
                            gt_abs_fut_traj[b, agent_idx, t] = (
                                gt_abs_fut_traj[b, agent_idx, t - 1] + last_valid_offset
                            )
                        
                        # update the mask
                        gt_fut_mask[b, agent_idx, last_valid_t + 1:] = 1.0
        
        
        # Pluto Agent Encoder Ver.
        agent_encoder_data = self._prepare_agent_encoder_inputs(
            gt_abs_fut_traj=gt_abs_fut_traj, # bs, query, 6, 2
            gt_fut_traj=gt_fut_traj, # bs, query, 6, 2
            gt_fut_mask=gt_fut_mask, # bs, query, 6
            gt_bboxes_list=gt_bboxes_list,
            gt_labels=gt_labels,
            gt_attr_labels=gt_attr_labels,
            device=device
        )
        
        agent_token = self.agent_encoder(agent_encoder_data)
        agent_query = agent_token
        
        can_bus_token = can_bus_embed.unsqueeze(1)  # (B, 1, 256)
        attn_out, _ = self.agent_can_cross_attn(
            query=agent_query,      # (B, N_gt, 256)
            key=can_bus_token,      # (B, 1, 256)
            value=can_bus_token,    # (B, 1, 256)
        )
        
        # 2. zero out the outputs of padding agents only
        gt_valid_mask = ~gt_masks  # (B, N_gt) True=valid
        attn_out = attn_out * gt_valid_mask.unsqueeze(-1)
        agent_query = agent_query * gt_valid_mask.unsqueeze(-1) 
        
        # Residual connection
        agent_query = self.agent_can_norm(agent_query + attn_out)
        
        # ----------------------
        # Auxiliary Head Outputs
        # ----------------------        
        # (a)+(b) The GT path is cleaned up like the e2e path:
        #   before: _select_gt_topk picked top-20 dynamic and top-20 static agents.
        #   now: no class split and no truncation, every GT agent is used (gt_masks marks padding).
        #   This branch only runs with latent_space_alignment_flag=True (False in the base config).
        agent_pos = gt_bboxes_m[:, :, :2]
        agent_mask = gt_masks

        # interaction modelling is not used here
        ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        ego_query = ego_his_feats
        ego_pos = torch.zeros((batch_size, 1, 2), device=ego_query.device)

        # Extract GT centerline & maks
        centerline_pos, centerline_mask = self.extract_reference_centerlines_batch(
            centerline_pts_list=map_gt_bboxes_list,
            centerline_labels_list=map_gt_labels_list,
            target_label=3,
            max_lines=self.ref_line_topk_num,
            fixed_points=self.map_num_pts_per_vec
        )
        
        # Reference Line Embedding
        # input concat
        x = torch.cat([ego_query, agent_query], dim=1)  # (B, 1+N_gt, D)
        pos = torch.cat([ego_pos, agent_pos], dim=1)

        ego_mask = torch.zeros((B, 1), device=ego_query.device, dtype=torch.bool)
        x_mask = torch.cat([ego_mask, agent_mask], dim=1)

        if self.use_map_token:
            # ============================================================
            # 1) take every map polyline (centerlines included, single set)
            #    centerline_label=-1 matches no GT label, so nothing is excluded.
            # ============================================================
            other_map_pos, other_map_mask = self.extract_non_centerline_polylines_batch(
                centerline_pts_list=map_gt_bboxes_list,
                centerline_labels_list=map_gt_labels_list,
                centerline_label=-1,
                max_lines=self.map_num_vec,
                fixed_points=self.map_num_pts_per_vec
            )   # other_map_pos: (B, Nm, P, 2), other_map_mask: (B, Nm)

            # ============================================================
            # 2) build the MapEncoder input features and encode them
            # ============================================================
            if other_map_pos.numel() > 0:
                # PointsEncoder expects 1=valid, so invert the padding mask.
                map_tokens = self.ra_map_encoder(other_map_pos, ~other_map_mask)  # (B, Nm, D)

                # representative coordinate for the positional encoding (polyline mean / midpoint)
                map_pos = other_map_pos.mean(dim=2)  # (B, Nm, 2)
            else:
                # fall back to a zero token with the right shape
                B = ego_query.shape[0]
                map_tokens = torch.zeros(
                    (B, 0, self.embed_dims),
                    device=ego_query.device,
                    dtype=ego_query.dtype
                )
                map_pos = torch.zeros(B, 0, 2, device=ego_query.device)
                other_map_mask = torch.zeros(B, 0, dtype=torch.bool, device=ego_query.device)
            
            x = torch.cat([x, map_tokens], dim=1)
            pos = torch.cat([pos, map_pos], dim=1)
            x_mask = torch.cat([x_mask, other_map_mask], dim=1)
        
        # M2M inputs
        bs, R, P, C = centerline_pos.shape
        
        # compute the vectors
        reference_vector = torch.zeros_like(centerline_pos) 
        reference_vector[:, :, :-1, :] = centerline_pos[:, :, 1:, :] - centerline_pos[:, :, :-1, :] # TODO: static indexing once the reference-line length is fixed
        reference_vector[:, :, -1, :] = reference_vector[:, :, -2, :]
        
        # compute the orientation (angle of the direction vector)
        r_orientation = safe_atan2(reference_vector[:, :, :, 1], reference_vector[:, :, :, 0])  # (bs, R, P)

        r_feature = torch.cat(
            [
                centerline_pos,
                reference_vector,
                torch.stack([r_orientation.cos(), r_orientation.sin()], dim=-1),
            ],
            dim=-1,
        )  # (bs, R, P, 6)
        
        reference_valid_mask = ~centerline_mask

        bs, R, P, C = r_feature.shape
        r_valid_mask = reference_valid_mask.unsqueeze(-1).expand(bs, R, P)
        r_valid_mask = r_valid_mask.view(bs * R, P)
        r_feature = r_feature.reshape(bs * R, P, C)
        
        r_emb = self.ra_planning_decoder.r_encoder(r_feature, r_valid_mask).view(bs, R, -1)
        r_pos = torch.cat([centerline_pos[:, :, 0], r_orientation[:, :, 0, None]], dim=-1)
        r_emb = r_emb + self.ra_planning_decoder.r_pos_emb(r_pos)
        r_emb = r_emb.unsqueeze(2).repeat(1, 1, self.ra_planning_decoder.num_mode, 1)
        
        return x, pos, x_mask, r_emb, centerline_pos, centerline_mask

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                ego_goal_points=None,
                gt_bboxes_list=None,
                gt_labels_list=None,
                gt_attr_labels=None,
                map_gt_bboxes_list=None,
                map_gt_labels_list=None,
            ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        
        if self.latent_space_alignment_flag and self.training:
            with torch.no_grad():
                device = ego_goal_points.device
                can_bus_infos = torch.tensor([each['can_bus'] for each in img_metas], dtype=torch.float32, device=device)
            
                m2m_outputs = self.m2m_input_query_generation(can_bus_infos,
                                                                gt_bboxes_list,
                                                                gt_labels_list,
                                                                gt_attr_labels,
                                                                ego_goal_points,
                                                                map_gt_bboxes_list,
                                                                map_gt_labels_list)
                
                m2m_x, m2m_pos, m2m_x_mask, m2m_centerline_embed, m2m_centerline_pos, centerline_mask = m2m_outputs
                
                m2m_latent_features = dict()
                m2m_latent_features['x'] = m2m_x
                m2m_latent_features['pos'] = m2m_pos
                m2m_latent_features['x_mask'] = m2m_x_mask
                
                m2m_latent_features['ref_embed'] = m2m_centerline_embed
                m2m_latent_features['ref_pos'] = m2m_centerline_pos
                m2m_latent_features['ref_mask'] = centerline_mask
                
        else:
            m2m_x = m2m_pos = m2m_x_mask = None
            m2m_centerline_pos = m2m_centerline_mask = None
            m2m_latent_features = None
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        
        if self.map_query_embed_type == 'all_pts':
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        elif self.map_query_embed_type == 'instance_pts':
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
            
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                map_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
                map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        outputs_trajs = []
        outputs_trajs_classes = []

        map_hs = map_hs.permute(0, 2, 1, 3)
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach()) # normalized
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs,self.map_num_vec, self.map_num_pts_per_vec,-1).mean(2)
            )
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid() # cx,cy,w,h
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_coords_bev.append(map_outputs_pts_coord.clone().detach())
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)
            
        if self.motion_decoder is not None:
            batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
            # motion_query
            motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
            mode_query = self.motion_mode_query.weight  # [fut_mode, D]
            # [M, B, D], M=A*fut_mode
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
            if self.use_pe:
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
                motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]
            else:
                motion_pos = None

            if self.motion_det_score is not None:
                motion_score = outputs_classes[-1]
                max_motion_score = motion_score.max(dim=-1)[0]
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
            else:
                invalid_motion_idx = None

            motion_hs = self.motion_decoder(
                query=motion_query,
                key=motion_query,
                value=motion_query,
                query_pos=motion_pos,
                key_pos=motion_pos,
                key_padding_mask=invalid_motion_idx)

            # motion-map cross-attention removed
            ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

            batch_size = outputs_coords_bev[-1].shape[0]
            motion_hs = motion_hs.permute(1, 0, 2).unflatten(
                dim=1, sizes=(num_agent, self.fut_mode)
            )
            ca_motion_query = ca_motion_query.squeeze(0).unflatten(
                dim=0, sizes=(batch_size, num_agent, self.fut_mode)
            )
            motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]
        else:
            raise NotImplementedError('Not implement yet')

        # ----------------------
        # Auxiliary Head Outputs
        # ----------------------
        outputs_traj = self.traj_branches[0](motion_hs)
        outputs_trajs.append(outputs_traj)
        outputs_traj_class = self.traj_cls_branches[0](motion_hs)
        outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
        (batch, num_agent) = motion_hs.shape[:2]
             
        map_outputs_classes = torch.stack(map_outputs_classes)
        map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = torch.stack(outputs_trajs)
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)

        # planning
        (batch, num_agent) = motion_hs.shape[:2]
        ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
        
        # interaction modelling is not used here
        ego_query = ego_his_feats
        ego_pos = torch.zeros((batch, 1, 2), device=ego_query.device)
        
        agent_conf = outputs_classes[-1]
        agent_query = motion_hs.reshape(batch, num_agent, -1)
        agent_query = self.agent_fus_mlp(agent_query) # [B, A, fut_mode, 2*D] -> [B, A, D]
        # agent_pos = outputs_coords_bev[-1] # normalized pos
        agent_pos = outputs_coords[-1][:, :, :2] # denormalized pos
        
        score_threshold = 0.6

        # (a) top-k removal + (b) static/dynamic merge
        #   before: dynamic (classes 0,1,2,3,7) and static (classes 4,5) were each cut
        #         to the top-20 by score, giving 40 planning tokens.
        #   now: no class split, every object query (num_query) is used, as in VAD.
        #         Without truncation there is no padding, so the mask is all-valid (False).
        agent_mask = torch.zeros(
            (batch, num_agent), device=agent_query.device, dtype=torch.bool
        )

        # ego <-> map interaction is not used here
        ego_pos = torch.zeros((batch, 1, 2), device=agent_query.device)
        map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
        map_conf = map_outputs_classes[-1]
        map_pos = map_outputs_coords_bev[-1]
        
        # centerline candidates only
        # -- Reference selection: top-6 candidates by predicted score ---------
        #   Unlike the nearest-to-ego variant, the candidates and their scores are
        #   computed *before* the reference selection. Pure score top-k, no geometric dedup.
        cl_cand, cl_cand_mask, cl_cand_idx, cl_score, goal_far = None, None, None, None, None
        ref_score_sel, ref_sel_idx, cand_conf = None, None, None
        if ego_goal_points is not None:
            goal_far = self._far_goal(ego_goal_points)                          # (B,2)
            cl_cand, cl_cand_mask, cl_cand_idx = self._score_candidates(map_pos, map_conf)
            cl_score = self._predict_scores(cl_cand, goal_far)                  # (B,N)

            # -- Eligibility = centerline confidence threshold ---------------
            #   A candidate must be perceptually confident to become a reference.
            #   The same local constant score_threshold (=0.6) that the nearest-to-ego
            #   selector uses is reused here; the auxiliary score regression (all 40
            #   candidates) is unaffected.
            cand_conf = torch.gather(map_conf.sigmoid()[..., 3], 1, cl_cand_idx)  # (B,N)
            eligible = (cand_conf > score_threshold) & (~cl_cand_mask.bool())     # (B,N)

            # Ranking uses detach (top-k is not differentiable); ineligible and padded
            #   candidates are pushed to the back with -1.
            B_sel, N_sel = cl_score.shape
            k_sel = min(self.ref_line_topk_num, N_sel)
            rank = cl_score.detach().masked_fill(~eligible, -1.0)
            # Empty-slate guard: if no candidate passes the threshold, keep the single
            #   highest-scoring non-padded one (forced top-1). An all-padding slate makes
            #   the reference-to-reference attention produce NaN.
            none_pass = eligible.sum(dim=1) == 0                                  # (B,)
            if none_pass.any():
                any_rank = cl_score.detach().masked_fill(cl_cand_mask.bool(), -1.0)
                best = any_rank.argmax(dim=1)                                     # (B,)
                rank[none_pass, best[none_pass]] = any_rank[none_pass, best[none_pass]]
            ref_sel_idx = rank.topk(k_sel, dim=1).indices                         # (B,k)
            # Slots below the threshold (rank<0) get zero geometry + padding.
            #   The sigmoid score is always > 0, so rank<0 means an ineligible filler slot.
            sel_pad = torch.gather(rank, 1, ref_sel_idx) < 0.0                    # (B,k) True=filler
            q_idx = torch.gather(cl_cand_idx, 1, ref_sel_idx)                     # (B,k) original map query idx
            P_map = map_pos.shape[2]
            centerline_pos = torch.gather(
                map_pos, 1, q_idx.view(B_sel, k_sel, 1, 1).expand(B_sel, k_sel, P_map, 2))
            centerline_query = torch.gather(
                map_query, 1,
                q_idx.view(B_sel, k_sel, 1, 1).expand(B_sel, k_sel, P_map, map_query.shape[-1]))
            centerline_pos = centerline_pos.masked_fill(sel_pad.view(B_sel, k_sel, 1, 1), 0.0)
            centerline_query = centerline_query.masked_fill(sel_pad.view(B_sel, k_sel, 1, 1), 0.0)
            centerline_mask = torch.gather(cl_cand_mask, 1, ref_sel_idx) | sel_pad  # (B,k) True=pad
            # Detached injection: the score head is trained by the auxiliary loss only
            #   (interpretability and selection stability); the consumer-side score_emb
            #   parameters still learn from the planning gradient. Filler slots score 0.
            ref_score_sel = torch.gather(cl_score, 1, ref_sel_idx).detach()      # (B,k)
            ref_score_sel = ref_score_sel.masked_fill(sel_pad, 0.0)
        else:
            # -- Fallback: no goal points -> nearest-to-ego top-6 selection ----
            centerline_query, centerline_pos, centerline_mask = self.select_topk_polyline_query_by_closest_point(
                map_query, map_pos, map_conf,
                target_classes=3,  # centerline
                score_threshold=score_threshold,
                topk=self.ref_line_topk_num
            )
        
        # centerline_query: [B,R,P,D]
        # centerline_pos: [B,R,P,2]
        
        
        # ---------------------------------------
        # ----------------- Planning ------------
        # ---------------------------------------
        # input concat - [ego | all objects] (B, 1+num_query, D)
        x = torch.cat([ego_query, agent_query], dim=1)
        pos = torch.cat([ego_pos, agent_pos], dim=1)

        ego_mask = torch.zeros((batch, 1), device=ego_query.device, dtype=torch.bool)
        x_mask = torch.cat([ego_mask, agent_mask], dim=1)

        if self.use_map_token:
            # (a) top-k removal + (b) single map set
            #   before: map polylines excluding the centerline class were cut to the 20
            #         nearest to the ego.
            #   now: every map query (map_num_vec) is used, centerlines included
            #         (reference-line extraction is a separate path).
            map_pos_all = denormalize_2d_pts(map_pos, self.pc_range)  # (B, map_num_vec, P, 2)
            map_mask_all = torch.zeros(
                (batch, self.map_num_vec), device=map_query.device, dtype=torch.bool
            )

            # MapEncoder/PointsEncoder use 1=valid, so the padding mask is inverted here.
            map_tokens = self.ra_map_encoder(map_pos_all, ~map_mask_all)  # (B, map_num_vec, D)

            map_pos_mean = map_pos_all.mean(dim=2)

            x = torch.cat([x, map_tokens], dim=1)
            pos = torch.cat([pos, map_pos_mean], dim=1)
            x_mask = torch.cat([x_mask, map_mask_all], dim=1)

        # Denoramlized centerline pts
        denormed_centerline_pos = denormalize_2d_pts(centerline_pos, self.pc_range)

        # -- Score aux computation moved up into the reference selection block --
        #   cl_cand / cl_cand_mask / cl_cand_idx / cl_score / goal_far are already filled.
        #   The auxiliary loss (loss_ra_cl_score) still covers all 40 candidates.


        # # Debugging Inputs
        # visualize_and_save(pos, denormed_centerline_pos, ego_goal_points, 'debug.png')
        
        # mask --> true: ignore, false: use
        # With with_planning=False, ra_planning (ego trajectory and ra_planning_decoder)
        #   and the planning latents are skipped. denormed_centerline_pos (centerline
        #   perception) survives, but its only consumer (loss_ra_planning) is off, so the
        #   planning keys of outs (ra_ego_trajs / ego_probability) stay None placeholders.
        #   Planning submodules then receive no gradient (find_unused_parameters=True).
        if self.with_planning:
            # Pass the detached score of the selected centerlines to the decoder.
            #   The heading prediction (ra_ego_yaws) is returned in addition.
            ra_ego_trajs, ra_ego_yaws, ego_probability, ref_emb = self.ra_planning(
                denormed_centerline_pos, centerline_mask, x, pos, x_mask, ego_goal_points,
                ref_score=ref_score_sel)

            if self.latent_space_alignment_flag and self.training:
                e2e_latent_features = dict()
                e2e_latent_features['x'] = x
                e2e_latent_features['pos'] = pos
                e2e_latent_features['x_mask'] = x_mask
                e2e_latent_features['ref_embed'] = ref_emb
                e2e_latent_features['ref_pos'] = denormed_centerline_pos
                e2e_latent_features['ref_mask'] = centerline_mask
            else:
                e2e_latent_features = None
        else:
            ra_ego_trajs = None
            ra_ego_yaws = None
            ego_probability = None
            e2e_latent_features = None

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_traj_preds': outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1),
            'all_traj_cls_scores': outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1),
            'map_all_cls_scores': map_outputs_classes,
            'map_all_bbox_preds': map_outputs_coords,
            'map_all_pts_preds': map_outputs_pts_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'map_enc_cls_scores': None,
            'map_enc_bbox_preds': None,
            'map_enc_pts_preds': None,
            'ra_ego_trajs': ra_ego_trajs,      # (B,R,M,T,2) positions only
            # Heading prediction (cos, sin): supervision, dumps and visualisation only.
            #   The controller and the open-loop evaluation do not read this key.
            'ra_ego_yaws': ra_ego_yaws,        # (B,R,M,T,2) raw regression (not normalised)
            'ego_probability': ego_probability,
            'ra_centerlines': denormed_centerline_pos,
            'ra_centerlines_mask': centerline_mask,
            'm2m_latent_features': m2m_latent_features,
            'e2e_latent_features': e2e_latent_features,
            # Score aux outputs (consumed by the loss and the visualisation hooks)
            'ra_cl_pos_metric': cl_cand,        # (B,N,P',2) predicted centerline candidates (metric)
            'ra_cl_mask': cl_cand_mask,         # (B,N) True=pad
            'ra_cl_query_idx': cl_cand_idx,     # (B,N) original map query index
            'ra_cl_score': cl_score,            # (B,N) predicted score in [0,1]
            'ra_goal_far': goal_far,            # (B,2) far waypoint(metric)
            # Score-based reference selection results (visualisation / analysis)
            'ra_cl_conf': cand_conf,            # (B,N) candidate centerline conf (sigmoid) / None on fallback
            'ra_cl_sel_idx': ref_sel_idx,       # (B,k) index within the candidates / None on fallback
            'ra_cl_sel_score': ref_score_sel,   # (B,k) score injected into the decoder (detached, 0 on filler) / None on fallback
        }

        return outs

    # ==================================================================
    # Score auxiliary head: the five methods below are the new ones.
    #   The reference-line selection and the planning path are untouched (aux only).
    # ==================================================================

    def _norm_xy(self, xy):
        """Normalise positions by pc_range (BEV extent -> roughly [-1, 1])."""
        x = xy[..., 0] / (self.pc_range[3] + 1e-6)
        y = xy[..., 1] / (self.pc_range[4] + 1e-6)
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _far_goal(goal_points):
        """(B,G,2) -> far waypoint (B,2). Index 1 (=far) when G>=2, else the last point."""
        if isinstance(goal_points, (list, tuple)):
            goal_points = goal_points[0]
        g = goal_points
        return g[:, 1, :] if g.shape[1] >= 2 else g[:, -1, :]

    def _resample_polylines(self, pts, num_pts):
        """Uniform arc-length resampling. (B,N,P,2) -> (B,N,num_pts,2).

        Batched version of _interpolate_lanes_batch_torch (per-line loop); same rule:
        cumulative arc length -> uniform targets -> linear interpolation inside a
        segment; a zero-length line repeats its first point.
        """
        B, N, P, _ = pts.shape
        if P < 2:
            return pts.expand(B, N, num_pts, 2).contiguous()
        seg = (pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1)                    # (B,N,P-1)
        cum = torch.cat([torch.zeros_like(seg[..., :1]), seg.cumsum(dim=-1)], dim=-1)  # (B,N,P)
        total = cum[..., -1:]                                                  # (B,N,1)
        t = torch.linspace(0.0, 1.0, num_pts, device=pts.device, dtype=pts.dtype)
        tgt = t.view(1, 1, -1) * total                                         # (B,N,num_pts)
        idx = torch.searchsorted(cum.contiguous(), tgt.contiguous(), right=True) - 1
        idx = idx.clamp(0, P - 2)
        c0 = torch.gather(cum, 2, idx)                                         # (B,N,num_pts)
        s0 = torch.gather(seg, 2, idx)
        w = ((tgt - c0) / (s0 + 1e-8)).clamp(0.0, 1.0).unsqueeze(-1)           # (B,N,num_pts,1)
        gi = idx.unsqueeze(-1).expand(B, N, num_pts, 2)
        p0 = torch.gather(pts, 2, gi)
        p1 = torch.gather(pts, 2, gi + 1)
        out = p0 * (1.0 - w) + p1 * w
        degen = (total < 1e-6).unsqueeze(-1)                                   # (B,N,1,1)
        return torch.where(degen, pts[:, :, :1].expand(B, N, num_pts, 2), out)

    def _score_candidates(self, map_pos, map_conf):
        """Pick centerline candidates from the *predicted* map queries.

        map_pos  (B,Q,P,2) normalized (= map_outputs_coords_bev[-1])
        map_conf (B,Q,C)   logits     (= map_outputs_classes[-1])
        → cand (B,N,score_num_pts,2) metric, mask (B,N) True=pad

        Candidates = top N (=score_cand_lines) by centerline-class (3) probability.
        No threshold: that avoids a new constant and keeps the auxiliary gradient alive
         early in training, when confidences are low. Poor candidates simply get a low target.
        """
        B, Q, P, _ = map_pos.shape
        cls3 = map_conf.sigmoid()[..., 3]                                      # (B,Q)
        k = min(self.score_cand_lines, Q)
        idx = cls3.topk(k, dim=1).indices                                      # (B,k)
        sel = torch.gather(map_pos, 1, idx.view(B, k, 1, 1).expand(B, k, P, 2))
        cand = self._resample_polylines(
            denormalize_2d_pts(sel, self.pc_range), self.score_num_pts)        # (B,k,P',2) metric
        mask = torch.zeros(B, k, dtype=torch.bool, device=map_pos.device)      # all valid
        return cand, mask, idx

    def _predict_scores(self, cl, goal_far):
        """cl (B,N,P,2) metric, goal_far (B,2) metric → score (B,N) ∈[0,1]."""
        B, N, P, _ = cl.shape
        g = goal_far.to(device=cl.device, dtype=cl.dtype).view(B, 1, 1, 2)
        feat = torch.cat([self._norm_xy(cl), self._norm_xy(cl - g)], dim=-1)   # (B,N,P,4)
        feat = self.score_pt_mlp(feat).max(dim=2).values                       # (B,N,H)
        return torch.sigmoid(self.score_head_mlp(feat).squeeze(-1))            # (B,N)

    @torch.no_grad()
    def _target_importance_abc(self, cl, cl_mask, gt, gt_mask, goal_far):
        """GT score target I = A * B * C_far (or I = A * B when score_target_use_cfar=False).

        (vectorised torch, no_grad)

        cl (B,R,P,2) metric / cl_mask (B,R) True=pad / gt (B,T,2) metric (cumulative absolute)
        gt_mask (B,T) / goal_far (B,2) metric.   returns (B,R) in [0,1]

        A and B follow the reference implementation (tangents / normal intersections /
        ADE score / direction score); C_far keeps only the far term of the goal score.
        """
        B, R, P, _ = cl.shape
        dev, dt = cl.device, cl.dtype
        W = self.score_lane_w
        tgt = torch.zeros(B, R, device=dev, dtype=dt)
        ego_heading = torch.tensor([0.0, 1.0], device=dev, dtype=dt)   # ego BEV frame forward = +y
        gf = goal_far.to(device=dev, dtype=dt)                         # (B,2)
        for b in range(B):
            gm = gt_mask[b] > 0.5
            gv = gt[b][gm] if int(gm.sum()) >= 2 else gt[b]            # (T,2)
            if gv.shape[0] < 2:
                continue
            valid = ~cl_mask[b].bool()                                # (R,) True=valid
            if not bool(valid.any()):
                continue
            c = cl[b]                                                 # (R,P,2)
            T = gv.shape[0]

            # ── A: ADE score ──
            d_pt = torch.cdist(gv.unsqueeze(0).expand(R, T, 2), c)     # (R,T,P)
            ade = d_pt.min(dim=2).values.mean(dim=1)                   # (R,)
            A = torch.exp(-ade / W)

            # -- local heading t_k (central difference, forward/backward at the ends) -> normal n_k --
            ip = torch.arange(1, T + 1, device=dev).clamp(max=T - 1)
            im = torch.arange(-1, T - 1, device=dev).clamp(min=0)
            v = gv[ip] - gv[im]                                        # (T,2)
            vn = v.norm(dim=1, keepdim=True)
            tg = torch.where(vn > 1e-6, v / vn.clamp(min=1e-12), ego_heading.expand_as(v))
            n = torch.stack([-tg[:, 1], tg[:, 0]], dim=1)              # (T,2) unit normal

            # -- B: normal x centerline segment intersection --
            Aseg = c[:, :-1, :]                                        # (R,P-1,2)
            E = c[:, 1:, :] - Aseg                                     # (R,P-1,2)
            den = (n[None, :, None, 0] * E[:, None, :, 1]
                   - n[None, :, None, 1] * E[:, None, :, 0])           # (R,T,P-1) cross(n,E)
            ok = den.abs() > 1e-12                                     # skip parallel
            w = Aseg[:, None, :, :] - gv[None, :, None, :]             # (R,T,P-1,2)  a_j - p_k
            dd = torch.where(ok, den, torch.ones_like(den))
            s = (w[..., 0] * E[:, None, :, 1] - w[..., 1] * E[:, None, :, 0]) / dd   # normal parameter (signed distance)
            u = (w[..., 0] * n[None, :, None, 1] - w[..., 1] * n[None, :, None, 0]) / dd  # segment parameter
            sel = ok & (u >= 0.0) & (u <= 1.0)                         # keep intersections inside the segment
            hit = sel.any(dim=2)                                       # (R,T)
            inf = torch.full_like(s, float('inf'))
            d_k = torch.where(sel, s.abs(), inf).min(dim=2).values     # (R,T) nearest intersection distance
            per = torch.where(hit, torch.exp(-d_k.clamp(max=1e6) / W),
                              torch.zeros_like(d_k))                   # no intersection -> 0 contribution
            Bd = per.mean(dim=1)                                       # (R,)

            # -- C_far: far goal x centerline nearest distance (far term only, vertex based) --
            #   Skipped when score_target_use_cfar=False (I = A * B), so that a single
            #   C_far ~ 0 cannot drive the whole multiplicative target to zero.
            if self.score_target_use_cfar:
                d_far = (c - gf[b].view(1, 1, 2)).norm(dim=-1).min(dim=1).values   # (R,)
                C = torch.exp(-d_far / W)
                I = A * Bd * C
            else:
                I = A * Bd
            tgt[b] = torch.where(valid, I, torch.zeros_like(A))
        return tgt

    def ra_planning(self, centerline_pos, centerline_mask, x, pos, x_mask, goal_points,
                      ref_score=None):
        device = x.device
        B = x.shape[0]
    
        # ============================================
        # Goal Points
        # ============================================
        if isinstance(goal_points, list):
            goal_points = goal_points[0]
            
        goal_embed = self.ra_goal_emb(goal_points)
        goal_pos = torch.zeros(B, 2, 2, device=device, dtype=torch.float32)
        goal_pos[:, :, 0] = goal_points[:, :, 0]
        goal_pos[:, :, 1] = goal_points[:, :, 1]
        x = torch.cat([x, goal_embed], dim=1)
        pos = torch.cat([pos, goal_pos], dim=1)
        goal_mask = torch.zeros(B, 2, device=device, dtype=torch.bool)
        x_mask = torch.cat([x_mask, goal_mask], dim=1)
        
        # ============================================
        # Positional Embedding
        # ============================================
        pos_embed = self.ra_pos_emb(pos)
        x = x + pos_embed
        
        for blk in self.ra_encoder_blocks:
            x = blk(x)
        x = self.ra_norm(x)

        # (a) Planning decoder cross-attention key/value.
        #   before: x = x[:, 0:1, :] passed a single scene (ego) token as memory, so the
        #         selected agents and map polylines were squeezed into one vector.
        #   now: the whole encoded sequence [ego | objects | map | goal(2)] is the memory,
        #         i.e. key/value length = 1 + num_query + map_num_vec + 2.
        #   x_mask is likewise passed through unchanged as memory_key_padding_mask.

        # Planning
        if self.latent_space_alignment_flag and self.training:
            with torch.no_grad():
                ra_ego_trajs, ego_probability, ref_emb = self.ra_planning_decoder(
                centerline_pos, centerline_mask, x, x_mask, self.latent_space_alignment_flag,
                ref_score=ref_score
                )
        else:
            ra_ego_trajs, ego_probability = self.ra_planning_decoder(
                centerline_pos, centerline_mask, x, x_mask, ref_score=ref_score
            )
            ref_emb = None
        # with torch.no_grad():
        #     ra_ego_trajs, ego_probability, ref_emb = self.ra_planning_decoder(
        #         centerline_pos, centerline_mask, x, x_mask, self.latent_space_alignment_flag
        #     )
            
        # ra_ego_trajs, ego_probability = self.ra_planning_decoder(
        #     centerline_pos, centerline_mask, x, x_mask, self.latent_space_alignment_flag
        # )
        # ref_emb = None

        # -- Split the 4-channel trajectory into position (2ch) and heading (2ch) ----
        #   The decoder emits [x, y, cos(theta), sin(theta)].
        #   Downstream consumers keep seeing a 2-channel ra_ego_trajs; heading goes to a
        #   separate key used for supervision and dumps only, so inference and driving
        #   shapes are unchanged.
        ra_ego_yaws = ra_ego_trajs[..., 2:4]     # (B,R,M,T,2)  raw (cos,sin)
        ra_ego_trajs = ra_ego_trajs[..., 0:2]    # (B,R,M,T,2) position (cumulative, metres)

        # -- Integrate the two position channels from deltas into absolute points ----
        #   With plan_traj_cumsum=False this block is skipped and the output is identical
        #   to the absolute-point variant. dim=-2 is T (the location head is viewed as
        #   (bs, R, M, future_steps, 2)). The heading channels are angles and are never
        #   accumulated. Everything downstream (loss, top-1 selection, controller,
        #   collision, score head, cost map, visualisation, evaluation) is unchanged.
        if self.plan_traj_cumsum:
            ra_ego_trajs = ra_ego_trajs.cumsum(dim=-2)

        return ra_ego_trajs, ra_ego_yaws, ego_probability, ref_emb

    def map_transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                                self.map_num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts*2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts*2:self.fut_ts*3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        traj_weights = torch.zeros_like(traj_targets)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks

        # Extra future timestamp mask for controlling pred horizon
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                   dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels, label_weights, bbox_targets, bbox_weights, traj_targets,
            traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )

    def _map_get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, traj_targets_list, traj_weights_list,
         gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
         )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                traj_targets_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    def map_get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)
    
    def compute_future_projection_torch(
        self,
        traj_xy,                 # (B, 100, 2)
        centerlines_xy,           # (B, R, P, 2)   ← map_centerline …
        start_idx = 4,
        step = 5,
        max_T = 8,):
        """
        Args
        ----
        traj_xyz        : (B,1,40,3)  - planning GT (XYZ) at 10 Hz
        centerlines_xy  : (B,R,P,2)   - predicted / GT reference lines, XY only
        start_idx       : slice start index (=0.4 s)
        step            : slice stride (=0.5 s -> 5 frames)
        max_T           : maximum number of points (8 -> 4 s)

        Returns
        -------
        future_proj : (B,R,T,2)  · [s,d]  (T ≤ max_T)
        """
        device  = traj_xy.device
        B, R, P = centerlines_xy.shape[:3]

        # -- 1. future samples at 0.5 s intervals ------------------------
        traj_xy = traj_xy[..., :2]              # (B, 40, 2)
        idxs    = torch.arange(start_idx, traj_xy.size(1), step,
                            device=device)[:max_T]       # (T,)
        T       = idxs.numel()
        if T == 0:
            raise ValueError("trajectory too short for sampling")
        future  = traj_xy[:, idxs, :]                       # (B, T, 2)

        # -- Orient the centerline along the ego forward direction (+y) ----------
        # Root cause: if the point order of the selected centerline runs against the ego
        #   (net forward y < 0), the arc length (cumsum) grows backwards, so s_abs < s_ego
        #   even while moving forward -> s_rel < 0 -> m = floor(s_rel/interval) clamps to
        #   0 -> the winner-take-all output collapses onto a single mode.
        # Fix: if the net direction fwd = last - first has y < 0, flip the point order
        #   along P and re-measure seg / cum_len / s_ego / s_abs. Padding rows are
        #   all-zero (net = 0) and are never flipped.
        fwd_y     = centerlines_xy[..., -1, 1] - centerlines_xy[..., 0, 1]   # (B,R) net +y displacement
        flip_mask = (fwd_y < 0)                                              # (B,R) reversed lines
        centerlines_xy = torch.where(
            flip_mask[..., None, None],
            torch.flip(centerlines_xy, dims=[-2]),                           # reverse the point order (P axis)
            centerlines_xy,
        )

        # -- 2. centerline segment quantities ---------------------
        pts0    = centerlines_xy[..., :-1, :]               # (B,R,P-1,2)
        seg_vec = centerlines_xy[..., 1:, :] - pts0         # (B,R,P-1,2)
        seg_len = torch.linalg.norm(seg_vec, dim=-1)        # (B,R,P-1)
        
        seg_len_safe = torch.where(seg_len < 1e-4,          # avoid division by zero
                                torch.full_like(seg_len, 1e-4),
                                seg_len)
        seg_unit = seg_vec / seg_len_safe.unsqueeze(-1)     # (B,R,P-1,2)

        # cumulative arc length s0 (per P point)
        cum_len = torch.zeros((B, R, P), device=device)
        cum_len[..., 1:] = torch.cumsum(seg_len, dim=-1)    # (B,R,P)

        # -- ego (origin) projected arc length s_ego --------------
        # A reference line that starts at the ego is already ego-relative. Here the
        #   centerline spans the whole map and the ego sits in the middle, so s is always
        #   large and m saturates. Projecting the ego origin onto each centerline and
        #   subtracting s_ego restores an ego-relative arc length.
        p_ego    = torch.zeros((B, 1, 1, 2), device=device)     # ego now = origin (ego-centric)
        w_e      = p_ego - pts0                                 # (B,R,P-1,2)
        t_raw_e  = (w_e * seg_unit).sum(-1) / seg_len_safe      # (B,R,P-1)
        t_clamp_e= t_raw_e.clamp(0.0, 1.0)
        proj_e   = pts0 + seg_unit * t_clamp_e.unsqueeze(-1)    # (B,R,P-1,2)
        d2_e     = ((p_ego - proj_e) ** 2).sum(-1)             # (B,R,P-1)
        _, idx_e = d2_e.min(dim=-1)                             # (B,R)
        idx_e_exp= idx_e.unsqueeze(-1)                          # (B,R,1)
        s_ego    = (cum_len.gather(-1, idx_e_exp).squeeze(-1)
                    + seg_len.gather(-1, idx_e_exp).squeeze(-1)
                    * t_clamp_e.gather(-1, idx_e_exp).squeeze(-1))   # (B,R)

        # output buffers
        proj_out = torch.zeros((B, R, T, 2), device=device)

        # -- 3. compute (s, d) for each future point ---------------
        for t in range(T):
            p      = future[:, t, :].unsqueeze(-2).unsqueeze(-2)   # (B,1,1,2) broadcasting
            w      = p - pts0                                     # (B,R,P-1,2)
            t_raw  = (w * seg_unit).sum(-1) / seg_len_safe        # (B,R,P-1)
            t_clamp= t_raw.clamp(0.0, 1.0)                        # projection ratio inside the segment
            proj   = pts0 + seg_unit * t_clamp.unsqueeze(-1)      # (B,R,P-1,2)
            d2     = ((p - proj)**2).sum(-1)                      # (B,R,P-1)
            d_min, idx_min = d2.min(dim=-1)                       # (B,R)
            d_min = d_min.sqrt()                                  # (B,R)

            # s = cumulative length at the segment start + segment length * t_clamp
            idx_min_exp = idx_min.unsqueeze(-1)                   # (B,R,1)
            s_base = cum_len.gather(-1, idx_min_exp).squeeze(-1)  # (B,R)
            t_sel  = t_clamp.gather(-1, idx_min_exp).squeeze(-1)  # (B,R)
            seg_sel= seg_len.gather(-1, idx_min_exp).squeeze(-1)  # (B,R)
            s_val  = s_base + seg_sel * t_sel                     # (B,R)

            # ego-relative arc length: how far the GT future point advanced along the centerline
            proj_out[..., t, 0] = s_val - s_ego
            proj_out[..., t, 1] = d_min

        return proj_out   # (B, R, T, 2)
    
    @staticmethod
    def heading_target_from_traj(ego_fut_gt_cumsum, static_thresh=0.5):
        """Build the GT heading target inside the head.

        The shared data loader is left untouched: heading is derived from the GT
        cumulative trajectory that already reaches the loss path.

        [algorithm]
          1. prepend the ego origin (0,0) to the cumulative positions P -> P_prev
          2. step displacement d_t = P_t - P_{t-1}   (= the original offset)
          3. displacement magnitude disp_t = ||d_t||
          4. θ_t = atan2(d_y, d_x)  →  target (cosθ, sinθ)
          5. static mask: disp_t < static_thresh -> zero heading weight at that step
             (at a standstill d ~ 0, so atan2 returns a meaningless angle)

        Args:
            ego_fut_gt_cumsum (Tensor): (B, T, 2) GT future cumulative positions (ego frame, metres)
            static_thresh (float): minimum step displacement [m]. With sample_interval=5
                one step is 0.5 s, so 0.5 m corresponds to 1.0 m/s.
        Returns:
            yaw_target (Tensor): (B, T, 2)  (cosθ, sinθ)
            yaw_weight (Tensor): (B, T)     1.0 = supervise heading, 0.0 = static (excluded)
        """
        P = ego_fut_gt_cumsum
        P_prev = torch.cat([torch.zeros_like(P[:, :1]), P[:, :-1]], dim=1)   # (B,T,2)
        d = P - P_prev                                                       # (B,T,2)
        disp = torch.linalg.norm(d, dim=-1)                                  # (B,T)
        theta = torch.atan2(d[..., 1], d[..., 0])                            # (B,T)
        yaw_target = torch.stack([theta.cos(), theta.sin()], dim=-1)         # (B,T,2)
        yaw_weight = (disp >= static_thresh).to(P.dtype)                     # (B,T)
        return yaw_target, yaw_weight

    def loss_ra_planning(self,
                            ego_fut_preds,
                            ego_fut_gt,
                            ego_fut_masks,
                            ego_probability,
                            ref_line_inputs,
                            ref_line_mask,
                            ego_yaw_preds=None):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): (B, k, M, fut_ts, 2)
            ego_fut_gt (Tensor): (B, fut_ts, 2)
            ego_yaw_preds (Tensor|None): (B, k, M, fut_ts, 2) heading prediction.
                None reproduces the behaviour without heading supervision exactly.
        Returns:
            loss_plan_reg (Tensor): planning regression loss (position + heading terms).
            loss_plan_cls (Tensor): planning map boundary constraint loss.
            best_trajectory (Tensor): (B, fut_ts, 2) positions only (collision loss input).
            yaw_diag (dict): heading diagnostics (detached); empty when heading is unused.
        """
        # Pluto
        bs = ego_probability.shape[0]
        
        future_proj = self.compute_future_projection_torch(
            traj_xy=ego_fut_gt,
            centerlines_xy=ref_line_inputs,   # already a torch.Tensor
            start_idx=0, step=1, max_T=self.fut_ts
        )

        future_projection = future_proj[:bs][
            torch.arange(bs), :, self.fut_ts - 1
        ]
        
        # meaning of r_padding_mask:
        # False = valid
        # True  = padding (ignored)

        r_padding_mask = ref_line_mask[:bs]
        
        # pick the centerline with the smallest lateral distance d
        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )
        # decide the mode from the longitudinal distance s
        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()
        target_m_index.clamp_(min=0, max=self.ego_fut_mode - 1)
        
        target_label = torch.zeros_like(ego_probability)
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1
        
        best_trajectory = ego_fut_preds[torch.arange(bs), target_r_index, target_m_index]
        
        # visualize_planning_selection(
        #     ego_fut_gt=ego_fut_gt,
        #     ego_fut_preds=ego_fut_preds,
        #     ref_line_inputs=ref_line_inputs,
        #     ref_line_mask=ref_line_mask,
        #     target_r_index=target_r_index,
        #     target_m_index=target_m_index,
        #     future_projection=future_projection,
        #     batch_idx=0,  # first batch only
        #     save_dir='./work_dirs/planning_viz',
        #     step=0
        # )
        
        # 1. regression loss
        # loss_plan_reg = F.smooth_l1_loss(best_trajectory, ego_fut_gt, reduction='none').sum(-1)
        # Element-wise loss
        loss_per_point = F.smooth_l1_loss(
            best_trajectory,
            ego_fut_gt,
            reduction='none'
        )  # (B, fut_ts, 2)

        # Sum over coordinate dimension
        loss_per_timestep = loss_per_point.sum(-1)  # (B, fut_ts)

        # -- heading (yaw) regression term ------------------------------------
        #   The decoder attaches (cos, sin) to each waypoint and supervises it directly.
        #   The same smooth L1, the same weight and the same .sum(-1) as the position
        #   term are used, so there is no new hyper-parameter.
        #   The teacher-forced (r*, m*) slot is exactly the one used by the position term.
        yaw_diag = {}
        loss_yaw_timestep = None
        if ego_yaw_preds is not None:
            best_yaw = ego_yaw_preds[torch.arange(bs), target_r_index, target_m_index]  # (B,T,2)
            yaw_target, yaw_weight = self.heading_target_from_traj(ego_fut_gt)
            yaw_target = yaw_target.to(best_yaw.dtype)
            yaw_weight = yaw_weight.to(best_yaw.dtype)
            loss_yaw_point = F.smooth_l1_loss(best_yaw, yaw_target, reduction='none')   # (B,T,2)
            loss_yaw_timestep = loss_yaw_point.sum(-1) * yaw_weight                     # (B,T)
            loss_per_timestep = loss_per_timestep + loss_yaw_timestep

        # Apply mask and compute mean
        # ego_fut_masks: 1=valid, 0=invalid
        if ego_fut_masks.sum() == 0:
            loss_plan_reg = loss_per_timestep.sum() * 0.0
        else:
            # apply the mask: keep valid timesteps only
            masked_loss = loss_per_timestep * ego_fut_masks  # (B, fut_ts)
            loss_plan_reg = masked_loss.sum() / ego_fut_masks.sum()

        # -- heading diagnostics (detached, never added to the loss) -----------
        #   Logs the absolute magnitude and the ratio of the position and heading terms.
        if loss_yaw_timestep is not None:
            with torch.no_grad():
                denom = ego_fut_masks.sum().clamp(min=1.0)
                pos_term = (loss_per_point.sum(-1) * ego_fut_masks).sum() / denom
                yaw_term = (loss_yaw_timestep * ego_fut_masks).sum() / denom
                yaw_diag = {
                    'diag_plan_pos_term': pos_term.detach(),
                    'diag_plan_yaw_term': yaw_term.detach(),
                    'diag_plan_yaw_ratio': (yaw_term / pos_term.clamp(min=1e-8)).detach(),
                    # fraction of timesteps where heading supervision is alive (static excluded)
                    'diag_plan_yaw_valid_frac': (
                        (yaw_weight * ego_fut_masks).sum() / denom).detach(),
                }

        bs, R, M = ego_probability.shape
        C = R * M

        masked_ego_probability = ego_probability.masked_fill(
            r_padding_mask.unsqueeze(-1), -1e6
        )
        
        target_cls_idx = (target_r_index * M + target_m_index).long()
        loss_plan_cls = F.cross_entropy(masked_ego_probability.view(bs, C), target_cls_idx)

        # -- teacher-forced (r*, m*) histogram --------------------------------
        #   Diagnostics only: no effect on the loss, enabled by an environment variable.
        if self._yaw_smoke_diag:
            with torch.no_grad():
                r_hist = torch.bincount(target_r_index.detach().view(-1), minlength=R)
                m_hist = torch.bincount(target_m_index.detach().view(-1), minlength=M)
                self._yaw_diag_rm_hist = (r_hist.cpu().tolist(), m_hist.cpu().tolist())

        return loss_plan_reg, loss_plan_cls, best_trajectory, yaw_diag

    def loss_planning_collision(self,
                                ego_fut_preds,
                                ego_fut_gt,
                                ego_fut_masks,
                                ego_fut_cmd,
                                agent_preds,
                                agent_fut_preds,
                                agent_score_preds,
                                agent_fut_cls_preds,
                                agent_bbox_preds=None):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """

        # Original (B, fut_ts, 2) before the repeat below: repeat(1,1,1,1) turns the 3-D
        #   tensor into a 4-D one. v4/v7 actually consume the GT, so keep a handle on the
        #   pre-repeat tensor. The existing line is left untouched.
        ego_fut_gt_v4 = ego_fut_gt
        ego_fut_gt = ego_fut_gt.repeat(1, 1, 1, 1)
        loss_plan_l1_weight = ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        # ego_fut_preds is already cumulative; convert it back to per-step deltas
        ego_fut_preds_diff = ego_fut_preds.clone()
        ego_fut_preds_diff[:, 1:, :] = ego_fut_preds_diff[:, 1:, :] - ego_fut_preds_diff[:, :-1, :]

        if self.plan_col_v4 or self.plan_col_v7:
            # v4 / v7 need the object extent and yaw. all_bbox_preds is a normalize_bbox
            #   [cx, cy, log w, log l, cz, log h, sin rot, cos rot, vx, vy]
            #   array (see mmcv/core/bbox/util.py), so (w, l) and rot are recovered here.
            #   The BEV heading conversion (-rot - pi/2) happens inside the loss.
            agent_wl = torch.stack([agent_bbox_preds[..., 2].exp(),
                                    agent_bbox_preds[..., 3].exp()], dim=-1)
            agent_yaw = torch.atan2(agent_bbox_preds[..., 6], agent_bbox_preds[..., 7])
            loss_plan_col = self.loss_plan_col(
                ego_fut_preds_diff,
                ego_fut_gt_v4,
                agent_preds,
                agent_fut_preds,
                agent_score_preds,
                agent_fut_cls_preds,
                agent_wl,
                agent_yaw,
                weight=ego_fut_masks
            )
        else:
            loss_plan_col = self.loss_plan_col(
                ego_fut_preds_diff,
                agent_preds,
                agent_fut_preds,
                agent_score_preds,
                agent_fut_cls_preds,
                weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
            )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_plan_col = torch.nan_to_num(loss_plan_col)

        return loss_plan_col
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    traj_preds,
                    traj_cls_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # traj regression loss
        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

        neg_inds = (bbox_weights[:, 0] == 0)
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks, neg_inds)

        loss_traj = self.loss_traj(
            best_traj_preds[isnotnan],
            traj_targets[isnotnan],
            traj_weights[isnotnan],
            avg_factor=num_total_pos)

        if self.use_traj_lr_warmup:
            loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
            loss_traj = loss_scale_factor * loss_traj

        # traj classification loss
        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        loss_traj_cls = self.loss_traj_cls(
            traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    def get_best_fut_preds(self,
             traj_preds,
             traj_targets,
             gt_fut_masks):
        """"Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts*2)

        return best_traj_preds

    def get_traj_cls_target(self,
             traj_preds,
             traj_targets,
             gt_fut_masks,
             neg_inds):
        """"Get Trajectory mode classification target.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            traj_labels (Tensor): traj cls labels (num_box_preds,).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels

    def map_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
 
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights

        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan,:4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.map_dir_interval:,:] - \
            denormed_pts_preds[:,:-self.map_dir_interval,:]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:,:] - pts_targets[:,:-self.map_dir_interval,:]

        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan,:,:],
            pts_targets_dir[isnotnan,:,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_traj_preds = preds_dicts['all_traj_preds']
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']
        # ego_fut_preds = preds_dicts['ego_fut_preds']
        
        # planning outputs
        ra_ego_trajs = preds_dicts['ra_ego_trajs']
        ra_ego_yaws = preds_dicts.get('ra_ego_yaws', None)
        ego_probability = preds_dicts['ego_probability']
        ra_centerlines = preds_dicts['ra_centerlines']
        ra_centerlines_mask = preds_dicts['ra_centerlines_mask']
        
        # # extract top1 trajectory
        # bs, lat, lon, pts, dim = ra_ego_trajs.shape
        
        # ego_probability_bs = ego_probability.detach().reshape(bs, -1)
        # top_vals, top_idx = torch.topk(ego_probability_bs, k=1, largest=True, sorted=True)  # [bs, 1]
        # top_idx = top_idx.squeeze(-1)  # [bs]
        
        # ego_fut_preds = ra_ego_trajs.reshape(bs, -1, pts, dim)  # [bs, lat*lon, pts, dim]
        
        # batch_indices = torch.arange(bs, device=ego_fut_preds.device)  # [bs]
        # ego_fut_preds = ego_fut_preds[batch_indices, top_idx].unsqueeze(1)  # [bs, 1, pts, dim]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, loss_traj, loss_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_bboxes_ignore_list)
        

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        if self.map_gt_shift_pts_pattern == 'v0':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v1':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v2':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v3':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v4':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]
        else:
            raise NotImplementedError
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        map_losses_cls, map_losses_bbox, map_losses_iou, \
            map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_traj'] = loss_traj[-1]
        loss_dict['loss_traj_cls'] = loss_traj_cls[-1]
        # loss from the last decoder layer
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # Planning Loss
        # With with_planning=False none of the three planning losses
        #   (loss_ra_plan_reg / loss_ra_plan_cls / loss_ra_plan_col) is computed or
        #   added to loss_dict. The detection and map losses above are unaffected.
        if self.with_planning:
            ego_fut_gt = ego_fut_gt.squeeze(1)
            ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
            ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

            # planning losses
            ego_fut_gt_cumsum = torch.cumsum(ego_fut_gt, dim=1)
            loss_ra_plan_inputs = [ra_ego_trajs, ego_fut_gt_cumsum, ego_fut_masks, ego_probability, ra_centerlines, ra_centerlines_mask]
            # Forward the heading prediction so it is summed with the same weight as position.
            loss_plan_reg, loss_plan_cls, best_trajectory, yaw_diag = self.loss_ra_planning(
                *loss_ra_plan_inputs, ego_yaw_preds=ra_ego_yaws)

            loss_dict['loss_ra_plan_reg'] = loss_plan_reg
            loss_dict['loss_ra_plan_cls'] = loss_plan_cls
            # Diagnostics: a key containing 'loss' would be added to the total by
            #   base._parse_losses, so always use the 'diag_' prefix.
            loss_dict.update(yaw_diag)

            batch, num_agent = all_traj_preds[-1].shape[:2]
            agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
            agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)

            loss_plan_input = [best_trajectory, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
                                all_bbox_preds[-1][..., 0:2], agent_fut_preds,
                                all_cls_scores[-1].sigmoid(), agent_fut_cls_preds.sigmoid(),
                                # unused when plan_col_v4=False
                                all_bbox_preds[-1]]

            loss_plan_col = self.loss_planning_collision(*loss_plan_input)
            loss_dict['loss_ra_plan_col'] = loss_plan_col

        # for Latent Space Alignment
        if self.latent_space_alignment_flag:
            m2m_latent_features = preds_dicts['m2m_latent_features']
            e2e_latent_features = preds_dicts['e2e_latent_features']
            
            m2m_x = m2m_latent_features['x']
            m2m_pos = m2m_latent_features['pos']
            m2m_x_mask = m2m_latent_features['x_mask']
            
            e2e_x = e2e_latent_features['x']
            e2e_pos = e2e_latent_features['pos']
            e2e_x_mask = e2e_latent_features['x_mask']
            
            # (b) Removing the static/dynamic split changed the token layout to
            #   (ego | agents | map). e2e keeps agents=num_query, but m2m (GT) has a
            #   per-batch GT count, so the agent span is measured from each tensor
            #   (the map block is the last map_num_vec entries).
            num_ego = 1
            num_map = self.map_num_vec

            ego_start = 0
            ego_end   = ego_start + num_ego

            agent_start = ego_end
            e2e_agent_end = e2e_x.shape[1] - num_map
            m2m_agent_end = m2m_x.shape[1] - num_map

            map_start = -num_map
            map_end   = None
            
            def class_slice(x, s, e):
                return x[:, s:e]

            # # ego
            # m2m_x_ego     = class_slice(m2m_x, ego_start, ego_end)
            # m2m_pos_ego   = class_slice(m2m_pos, ego_start, ego_end)
            # m2m_mask_ego  = class_slice(m2m_x_mask, ego_start, ego_end)

            # e2e_x_ego     = class_slice(e2e_x, ego_start, ego_end)
            # e2e_pos_ego   = class_slice(e2e_pos, ego_start, ego_end)
            # e2e_mask_ego  = class_slice(e2e_x_mask, ego_start, ego_end)

            # # static
            # m2m_x_static    = class_slice(m2m_x, static_start, static_end)
            # m2m_pos_static  = class_slice(m2m_pos, static_start, static_end)
            # m2m_mask_static = class_slice(m2m_x_mask, static_start, static_end)

            # e2e_x_static    = class_slice(e2e_x, static_start, static_end)
            # e2e_pos_static  = class_slice(e2e_pos, static_start, static_end)
            # e2e_mask_static = class_slice(e2e_x_mask, static_start, static_end)

            # # dynamic
            # m2m_x_dynamic    = class_slice(m2m_x, dynamic_start, dynamic_end)
            # m2m_pos_dynamic  = class_slice(m2m_pos, dynamic_start, dynamic_end)
            # m2m_mask_dynamic = class_slice(m2m_x_mask, dynamic_start, dynamic_end)

            # e2e_x_dynamic    = class_slice(e2e_x, dynamic_start, dynamic_end)
            # e2e_pos_dynamic  = class_slice(e2e_pos, dynamic_start, dynamic_end)
            # e2e_mask_dynamic = class_slice(e2e_x_mask, dynamic_start, dynamic_end)

            # # map
            # m2m_x_map    = class_slice(m2m_x, map_start, map_end)
            # m2m_pos_map  = class_slice(m2m_pos, map_start, map_end)
            # m2m_mask_map = class_slice(m2m_x_mask, map_start, map_end)

            # e2e_x_map    = class_slice(e2e_x, map_start, map_end)
            # e2e_pos_map  = class_slice(e2e_pos, map_start, map_end)
            # e2e_mask_map = class_slice(e2e_x_mask, map_start, map_end)
            
            # ----- per-class alignment loss -----
            align_loss = latent_alignment_loss_nn_with_radius(
                feat_e2e=e2e_x,
                pos_e2e=e2e_pos,
                mask_e2e=e2e_x_mask,
                feat_m2m=m2m_x,
                pos_m2m=m2m_pos,
                mask_m2m=m2m_x_mask,
                use_js=True,
            )
            # align_loss_ego = latent_alignment_loss_nn_with_radius(
            #     feat_e2e=e2e_x_ego,
            #     pos_e2e=e2e_pos_ego,
            #     mask_e2e=e2e_mask_ego,
            #     feat_m2m=m2m_x_ego,
            #     pos_m2m=m2m_pos_ego,
            #     mask_m2m=m2m_mask_ego,
            #     use_js=True,
            # )

            # align_loss_static = latent_alignment_loss_nn_with_radius(
            #     feat_e2e=e2e_x_static,
            #     pos_e2e=e2e_pos_static,
            #     mask_e2e=e2e_mask_static,
            #     feat_m2m=m2m_x_static,
            #     pos_m2m=m2m_pos_static,
            #     mask_m2m=m2m_mask_static,
            #     use_js=True,
            # )

            # align_loss_dynamic = latent_alignment_loss_nn_with_radius(
            #     feat_e2e=e2e_x_dynamic,
            #     pos_e2e=e2e_pos_dynamic,
            #     mask_e2e=e2e_mask_dynamic,
            #     feat_m2m=m2m_x_dynamic,
            #     pos_m2m=m2m_pos_dynamic,
            #     mask_m2m=m2m_mask_dynamic,
            #     use_js=True,
            # )

            # align_loss_map = latent_alignment_loss_nn_with_radius(
            #     feat_e2e=e2e_x_map,
            #     pos_e2e=e2e_pos_map,
            #     mask_e2e=e2e_mask_map,
            #     feat_m2m=m2m_x_map,
            #     pos_m2m=m2m_pos_map,
            #     mask_m2m=m2m_mask_map,
            #     use_js=True,
            # )
            
            # align_loss = (
            #     align_loss_ego * 0.5
            #     + align_loss_static * 4.5
            #     + align_loss_dynamic * 4.5
            #     + align_loss_map * 0.5
            # ) / 10.0
            
            loss_dict['latent_alignment_loss'] = align_loss

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i in zip(
            map_losses_cls[:-1],
            map_losses_bbox[:-1],
            map_losses_iou[:-1],
            map_losses_pts[:-1],
            map_losses_dir[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if map_enc_cls_scores is not None:
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i])
                for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            map_enc_loss_cls, map_enc_loss_bbox, map_enc_loss_iou, \
                 map_enc_loss_pts, map_enc_loss_dir = \
                self.map_loss_single(
                    map_enc_cls_scores, map_enc_bbox_preds,
                    map_enc_pts_preds, map_gt_bboxes_list,
                    map_binary_labels_list, map_gt_pts_list,
                    map_gt_bboxes_ignore
                )
            loss_dict['enc_loss_map_cls'] = map_enc_loss_cls
            loss_dict['enc_loss_map_bbox'] = map_enc_loss_bbox
            loss_dict['enc_loss_map_iou'] = map_enc_loss_iou
            loss_dict['enc_loss_map_pts'] = map_enc_loss_pts
            loss_dict['enc_loss_map_dir'] = map_enc_loss_dir

        # -- Auxiliary centerline score regression ----------------------------
        #   The target (A * B * C_far) is computed on the fly here (no cache).
        pred = preds_dicts.get('ra_cl_score', None)
        cl_metric = preds_dicts.get('ra_cl_pos_metric', None)
        cl_mask = preds_dicts.get('ra_cl_mask', None)
        goal_far = preds_dicts.get('ra_goal_far', None)
        if (pred is not None) and (cl_metric is not None) and (goal_far is not None):
            # GT ego future (metric) = cumulative offsets (same path as loss_ra_planning)
            gt = ego_fut_gt.squeeze(1)                       # (B,T,2) offsets
            gt = torch.cumsum(gt, dim=1)
            # match dtypes (ego_fut_gt is double while cl_metric is float -> cdist mismatch)
            gt = gt.to(device=cl_metric.device, dtype=cl_metric.dtype)
            gm = ego_fut_masks.squeeze(1).squeeze(1).to(device=cl_metric.device)  # (B,T)

            target = self._target_importance_abc(cl_metric, cl_mask, gt, gm, goal_far)  # (B,N)
            valid = ~cl_mask.bool()
            if valid.any():
                loss_score = F.smooth_l1_loss(pred[valid], target[valid])
            else:
                loss_score = pred.sum() * 0.0
            loss_dict['loss_ra_cl_score'] = self.score_loss_weight * loss_score

        # -- Smoke diagnostics (only with RA_YAW_SMOKE_DIAG=1) ----------------
        #   (b) per-class map conf>0.5 pass rate.
        #   (c) teacher-forced (r*, m*) histogram, collected in loss_ra_planning.
        #   Never added to loss_dict - diagnostics go to stdout only.
        if self._yaw_smoke_diag:
            self._yaw_diag_step += 1
            if self._yaw_diag_step % self._yaw_diag_interval == 0:
                with torch.no_grad():
                    conf = map_all_cls_scores[-1].sigmoid()          # (B, map_num_vec, num_cls)
                    passed = (conf > 0.5).float()
                    rate = passed.mean(dim=(0, 1))                   # (num_cls,)
                    n_tok = conf.shape[0] * conf.shape[1]
                    rate_s = ' '.join(f'c{i}={r*100:.2f}%' for i, r in enumerate(rate.tolist()))
                    print(f"[YAWDIAG][it{self._yaw_diag_step}] map conf>0.5 pass-rate "
                          f"(tokens={n_tok}, classes={conf.shape[-1]}): {rate_s}", flush=True)
                    if self._yaw_diag_rm_hist is not None:
                        r_hist, m_hist = self._yaw_diag_rm_hist
                        print(f"[YAWDIAG][it{self._yaw_diag_step}] teacher-forced "
                              f"r* hist={r_hist}  m* hist={m_hist}", flush=True)
                    if 'diag_plan_yaw_term' in loss_dict:
                        print(f"[YAWDIAG][it{self._yaw_diag_step}] "
                              f"pos_term={loss_dict['diag_plan_pos_term'].item():.6f} "
                              f"yaw_term={loss_dict['diag_plan_yaw_term'].item():.6f} "
                              f"ratio={loss_dict['diag_plan_yaw_ratio'].item():.4f} "
                              f"yaw_valid_frac={loss_dict['diag_plan_yaw_valid_frac'].item():.4f}",
                              flush=True)

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), \
             'len(preds_dict) should be equal to len(map_preds_dicts)'
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds['map_bboxes']
            map_scores = map_preds['map_scores']
            map_labels = map_preds['map_labels']
            map_pts = map_preds['map_pts']

            ret_list.append([bboxes, scores, labels, trajs, map_bboxes,
                             map_scores, map_labels, map_pts])

        return ret_list

    def select_and_pad_pred_map(
        self,
        motion_pos,
        map_query,
        map_score,
        map_pos,
        map_thresh=0.5,
        dis_thresh=None,
        pe_normalization=True,
        use_fix_pad=False
    ):
        """select_and_pad_pred_map.
        Args:
            motion_pos: [B, A, 2]
            map_query: [B, P, D].
            map_score: [B, P, 3].
            map_pos: [B, P, pts, 2].
            map_thresh: map confidence threshold for filtering low-confidence preds
            dis_thresh: distance threshold for masking far maps for each agent in cross-attn
            use_fix_pad: always pad one lane instance for each batch
        Returns:
            selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
            selected_map_pos: [B*A, P1(+1), 2]
            selected_padding_mask: [B*A, P1(+1)]
        """
        
        if dis_thresh is None:
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask


    def select_and_pad_query(
        self,
        query,
        query_pos,
        query_score,
        score_thresh=0.5,
        use_fix_pad=True
    ):
        """select_and_pad_query.
        Args:
            query: [B, Q, D].
            query_pos: [B, Q, 2]
            query_score: [B, Q, C].
            score_thresh: confidence threshold for filtering low-confidence query
            use_fix_pad: always pad one query instance for each batch
        Returns:
            selected_query: [B, Q', D]
            selected_query_pos: [B, Q', 2]
            selected_padding_mask: [B, Q']
        """

        # select & pad query for different batch using score_thresh
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0)
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        return selected_query, selected_query_pos, selected_padding_mask
    
    # (a) top-k removal: these three selection helpers lost their call sites and
    #   · _select_gt_topk                                          (GT agent class-wise top-k)
    #   · select_topk_query_by_class_with_threshold                (predicted agent class-wise top-k)
    #   · select_topk_polyline_query_excluding_classes_by_closest_point (map top-k)
    #   were removed with them. The only selection helper kept is the centerline
    #   reference-line one, select_topk_polyline_query_by_closest_point(topk=ref_line_topk_num).
    def select_topk_polyline_query_by_closest_point(
        self,
        query,
        query_pos,
        query_score,
        *,
        target_classes: Union[int, List[int]],
        topk: int = 50,
        score_threshold: float = 0.3,
        use_closest_point_for_ranking: bool = True
    ):
        """select_topk_polyline_query_by_closest_point.
        
        Select top-k polyline queries based on closest point to ego vehicle.
        Each instance is ranked by: (1) class score, (2) distance to ego.
        
        Args:
            query: [B, Q, P, D] - Query features
            query_pos: [B, Q, P, 2] - Query positions
            query_score: [B, Q, C] - Query classification scores
            target_classes: int or List[int] - Target class(es)
            topk: int - Number of instances to select
            score_threshold: float - Minimum score threshold
            use_fix_pad: bool - Whether to add padding instance
            use_closest_point_for_ranking: bool - If True, use distance of closest point for ranking
            
        Returns:
            selected_query: [B, K, P, D]
            selected_query_pos: [B, K, P, 2]
            selected_padding_mask: [B, K]
        """
        batch_size = query.shape[0]
        num_instances = query.shape[1]
        num_points = query.shape[2]
        feat_dim = query.shape[3]
        num_classes = query_score.shape[-1]
        
        query_score = query_score.sigmoid()
        
        # Validation
        if isinstance(target_classes, int):
            target_classes_list = [target_classes]
        else:
            target_classes_list = list(target_classes)
        
        invalid_classes = [c for c in target_classes_list if c < 0 or c >= num_classes]
        if invalid_classes:
            raise ValueError(
                f"target_classes {invalid_classes} out of range [0, {num_classes-1}]"
            )
        
        # Get class scores
        if len(target_classes_list) == 1:
            class_scores = query_score[:, :, target_classes_list[0]]
        else:
            target_classes_tensor = torch.tensor(target_classes_list, device=query_score.device)
            class_scores = query_score[:, :, target_classes_tensor]
            class_scores = class_scores.max(dim=-1)[0]
        
        # Apply threshold
        threshold_mask = class_scores > score_threshold
        
        # Calculate distance to ego (0, 0) for each point
        distances = torch.sqrt(query_pos[..., 0]**2 + query_pos[..., 1]**2)  # [B, Q, P]
        
        if use_closest_point_for_ranking:
            # Use minimum distance per instance for ranking
            min_distances, _ = distances.min(dim=-1)  # [B, Q]
        else:
            # Use mean distance per instance
            min_distances = distances.mean(dim=-1)  # [B, Q]
        
        output_size = topk
        
        selected_query_list = []
        selected_pos_list = []
        selected_mask_list = []
        
        for i in range(batch_size):
            valid_mask = threshold_mask[i]
            valid_indices = torch.where(valid_mask)[0]
            num_valid = len(valid_indices)
            
            if num_valid == 0:
                # Nothing passed the threshold, so rank over all queries
                all_scores = class_scores[i]     # [Q]
                all_dists = min_distances[i]     # [Q]
                
                # Same combined-score ranking as above
                score_norm = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-6)
                dist_norm = (all_dists - all_dists.min()) / (all_dists.max() - all_dists.min() + 1e-6)
                combined_score = score_norm - 0.3 * dist_norm
                
                # Take the single best query
                _, top1_idx = torch.topk(combined_score, 1, largest=True)
                
                # Keep the dimension: [1, P, D]
                valid_query = query[i, top1_idx]       
                valid_pos = query_pos[i, top1_idx]
                
                # Zero-pad the remaining (topk - 1) slots
                pad_num = output_size - 1
                pad_query = torch.zeros((pad_num, num_points, feat_dim), device=query.device)
                pad_pos = torch.zeros((pad_num, num_points, 2), device=query.device)
                
                valid_query = torch.cat([valid_query, pad_query], dim=0)
                valid_pos = torch.cat([valid_pos, pad_pos], dim=0)
                
                # Mask: only index 0 (top-1) is valid (False); the rest are padding (True)
                padding_mask = torch.ones(output_size, dtype=torch.bool, device=query.device)
                padding_mask[0] = False
            elif num_valid >= topk:
                # Rank by: (1) score (descending), (2) distance (ascending)
                valid_scores = class_scores[i, valid_indices]
                valid_dists = min_distances[i, valid_indices]
                
                # Combined ranking: higher score is better, lower distance is better
                # Normalize both to [0, 1] range and combine
                score_norm = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-6)
                dist_norm = (valid_dists - valid_dists.min()) / (valid_dists.max() - valid_dists.min() + 1e-6)
                combined_score = score_norm - 0.3 * dist_norm  # Prioritize score over distance
                
                _, topk_valid_idx = torch.topk(combined_score, topk, largest=True, sorted=True)
                selected_indices = valid_indices[topk_valid_idx]
                
                valid_query = query[i, selected_indices]
                valid_pos = query_pos[i, selected_indices]
                padding_mask = torch.zeros(output_size, dtype=torch.bool, device=query.device)
            
            else:
                # num_valid < topk
                valid_scores = class_scores[i, valid_indices]
                _, sorted_idx = torch.sort(valid_scores, descending=True)
                selected_indices = valid_indices[sorted_idx]
                
                valid_query = query[i, selected_indices]
                valid_pos = query_pos[i, selected_indices]
                
                pad_num = topk - num_valid
                pad_query = torch.zeros((pad_num, num_points, feat_dim), device=query.device)
                pad_pos = torch.zeros((pad_num, num_points, 2), device=query.device)
                
                valid_query = torch.cat([valid_query, pad_query], dim=0)
                valid_pos = torch.cat([valid_pos, pad_pos], dim=0)
                
                padding_mask = torch.zeros(output_size, dtype=torch.bool, device=query.device)
                padding_mask[num_valid:] = True
            
            selected_query_list.append(valid_query)
            selected_pos_list.append(valid_pos)
            selected_mask_list.append(padding_mask)
        
        selected_query = torch.stack(selected_query_list, dim=0)
        selected_query_pos = torch.stack(selected_pos_list, dim=0)
        selected_padding_mask = torch.stack(selected_mask_list, dim=0)

        return selected_query, selected_query_pos, selected_padding_mask

def visualize_and_save(pos, centerline_pos, ego_goal_points, save_path='result.png'):
    """
    pos: (1, 41, 2) - surrounding object centres (agents)
    centerline_pos: (1, 6, 20, 2) - centerline segments (map)
    save_path: output file path
    """
    
    # 1. convert tensors to numpy (drop the batch dimension: index 0)
    # detach() cuts gradient tracking and cpu() covers the GPU case.
    agents = pos[0].detach().cpu().numpy()            # shape: (41, 2)
    lines = centerline_pos[0].detach().cpu().numpy()  # shape: (6, 20, 2)
    
    goal_points = ego_goal_points[0].detach().cpu().numpy()

    plt.figure(figsize=(10, 10))

    # 2. draw the centerlines (connected)
    # iterate over the segments.
    for i in range(lines.shape[0]):
        line = lines[i] # (20, 2)
        # label only the first line so the legend stays readable
        label = 'Centerline' if i == 0 else None
        plt.plot(line[:, 0], line[:, 1], color='green', alpha=0.7, linewidth=2, label=label)
        # (optional) uncomment below to also show the points
        # plt.scatter(line[:, 0], line[:, 1], color='green', s=10)

    # 3. draw the surrounding objects (agents)
    plt.scatter(agents[:, 0], agents[:, 1], color='blue', s=50, label='Agents', zorder=5)
    
    # draw the goal point
    plt.scatter(goal_points[:, 0], goal_points[:, 1], color='red', s=80, label='Agents', zorder=5)

    # 4. style
    plt.title("Agent Positions & Centerlines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')  # 1:1 aspect ratio to avoid distortion

    # 5. save and close
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image saved to {save_path}")

import os

def visualize_planning_selection(
    ego_fut_gt,          # (B, T, 2)
    ego_fut_preds,       # (B, R, M, T, 2)
    ref_line_inputs,     # (B, R, P, 2)
    ref_line_mask,       # (B, R)
    target_r_index,      # (B,)
    target_m_index,      # (B,)
    future_projection,   # (B, R, 2) - [s, d]
    batch_idx=0,
    save_dir='./planning_viz',
    step=0
):
    """
    Visualize planning trajectory selection
    
    Args:
        ego_fut_gt: Ground truth trajectory (cumsum applied)
        ego_fut_preds: Predicted trajectories for all R centerlines and M modes
        ref_line_inputs: Reference centerlines
        ref_line_mask: Centerline validity mask (True=padding)
        target_r_index: Selected centerline index
        target_m_index: Selected mode index
        future_projection: Frenet coordinates (s, d)
        batch_idx: Which batch to visualize
        save_dir: Directory to save images
        step: Training step number
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data for this batch
    b = batch_idx
    gt_traj = ego_fut_gt[b].detach().cpu().numpy()  # (T, 2)
    pred_trajs = ego_fut_preds[b].detach().cpu().numpy()  # (R, M, T, 2)
    centerlines = ref_line_inputs[b].detach().cpu().numpy()  # (R, P, 2)
    centerline_mask = ref_line_mask[b].detach().cpu().numpy()  # (R,)
    proj = future_projection[b].detach().cpu().numpy()  # (R, 2)
    
    target_r = target_r_index[b].item()
    target_m = target_m_index[b].item()
    
    R, M, T, _ = pred_trajs.shape
    
    # ============================================
    # Create figure
    # ============================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ============================================
    # Left plot: Overview with all predictions
    # ============================================
    ax1 = axes[0]
    
    # 1. Draw centerlines
    for r in range(R):
        if centerline_mask[r]:  # True = padding
            continue
        
        line = centerlines[r]  # (P, 2)
        is_selected = (r == target_r)
        
        color = 'red' if is_selected else 'gray'
        linewidth = 3 if is_selected else 1
        alpha = 1.0 if is_selected else 0.3
        label = f'Centerline {r} (SELECTED)' if is_selected else f'Centerline {r}'
        
        ax1.plot(line[:, 0], line[:, 1], 
                color=color, linewidth=linewidth, alpha=alpha, 
                linestyle='--', label=label)
        
        # Show projection info
        if not centerline_mask[r]:
            s, d = proj[r]
            ax1.text(line[0, 0], line[0, 1], 
                    f'  s={s:.1f}m\n  d={d:.1f}m', 
                    fontsize=8, color=color)
    
    # 2. Draw all predicted trajectories (light)
    for r in range(R):
        if centerline_mask[r]:
            continue
        for m in range(M):
            traj = pred_trajs[r, m]  # (T, 2)
            is_selected = (r == target_r and m == target_m)
            
            if not is_selected:
                ax1.plot(traj[:, 0], traj[:, 1], 
                        color='blue', linewidth=0.5, alpha=0.1)
    
    # 3. Draw selected prediction (thick)
    selected_pred = pred_trajs[target_r, target_m]  # (T, 2)
    ax1.plot(selected_pred[:, 0], selected_pred[:, 1], 
            color='green', linewidth=3, marker='o', markersize=4,
            label=f'Selected Pred (R={target_r}, M={target_m})')
    
    # 4. Draw GT trajectory
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 
            color='orange', linewidth=3, marker='s', markersize=4,
            label='GT Trajectory', linestyle='-', alpha=0.8)
    
    # 5. Mark start point
    ax1.plot(0, 0, 'ko', markersize=10, label='Ego Start (0,0)')
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'Planning Trajectory Selection (Step {step}, Batch {b})', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # ============================================
    # Right plot: Zoomed in comparison
    # ============================================
    ax2 = axes[1]
    
    # Selected centerline
    selected_centerline = centerlines[target_r]
    ax2.plot(selected_centerline[:, 0], selected_centerline[:, 1],
            'r--', linewidth=2, label=f'Selected Centerline {target_r}')
    
    # Selected prediction
    ax2.plot(selected_pred[:, 0], selected_pred[:, 1],
            'g-', linewidth=2, marker='o', markersize=6,
            label=f'Selected Prediction (M={target_m})')
    
    # GT
    ax2.plot(gt_traj[:, 0], gt_traj[:, 1],
            'orange', linewidth=2, marker='s', markersize=6,
            label='GT Trajectory', linestyle='-', alpha=0.8)
    
    # All modes for selected centerline
    for m in range(M):
        if m != target_m:
            traj = pred_trajs[target_r, m]
            ax2.plot(traj[:, 0], traj[:, 1],
                    'b-', linewidth=1, alpha=0.3)
    
    # Start point
    ax2.plot(0, 0, 'ko', markersize=10, label='Ego Start')
    
    # Error visualization
    errors = np.linalg.norm(selected_pred - gt_traj, axis=1)
    final_error = errors[-1]
    avg_error = errors.mean()
    
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title(f'Selected vs GT\nAvg Error: {avg_error:.2f}m, Final: {final_error:.2f}m',
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # ============================================
    # Save
    # ============================================
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'planning_step_{step:06d}_batch_{b}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Planning Viz] Saved to {save_path}")
    print(f"  Selected: R={target_r}, M={target_m}")
    print(f"  Projection: s={proj[target_r, 0]:.2f}m, d={proj[target_r, 1]:.2f}m")
    print(f"  Avg error: {avg_error:.2f}m, Final error: {final_error:.2f}m")