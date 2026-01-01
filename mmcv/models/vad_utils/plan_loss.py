import math
import torch
from torch import nn as nn
from mmcv.models.losses.utils import weighted_loss
from mmcv.models.builder import LOSSES


@LOSSES.register_module()
class PlanMapBoundLoss(nn.Module):
    """Planning constraint to push ego vehicle away from the lane boundary.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        map_thresh (float, optional): confidence threshold to filter map predictions.
        lane_bound_cls_idx (float, optional): lane_boundary class index.
        dis_thresh (float, optional): distance threshold between ego vehicle and lane bound.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        map_thresh=0.5,
        lane_bound_cls_idx=2,
        dis_thresh=1.0,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        perception_detach=False
    ):
        super(PlanMapBoundLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.map_thresh = map_thresh
        self.lane_bound_cls_idx = lane_bound_cls_idx
        self.dis_thresh = dis_thresh
        self.pc_range = point_cloud_range
        self.perception_detach = perception_detach

    def forward(self,
                ego_fut_preds,
                lane_preds,
                lane_score_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.perception_detach:
            lane_preds = lane_preds.detach()
            lane_score_preds = lane_score_preds.detach()

        # filter lane element according to confidence score and class
        not_lane_bound_mask = lane_score_preds[..., self.lane_bound_cls_idx] < self.map_thresh
        # denormalize map pts
        lane_bound_preds = lane_preds.clone()
        lane_bound_preds[...,0:1] = (lane_bound_preds[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
        lane_bound_preds[...,1:2] = (lane_bound_preds[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
        # pad not-lane-boundary cls and low confidence preds
        lane_bound_preds[not_lane_bound_mask] = 1e6

        loss_bbox = self.loss_weight * plan_map_bound_loss(ego_fut_preds, lane_bound_preds,
                                                           weight=weight, dis_thresh=self.dis_thresh,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

@weighted_loss
def plan_map_bound_loss(pred, target, dis_thresh=1.0):
    """Planning map bound constraint (L1 distance).

    Args:
        pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 2].
        target (torch.Tensor): lane_bound_preds, [B, num_vec, num_pts, 2].
        weight (torch.Tensor): [B, fut_ts]

    Returns:
        torch.Tensor: Calculated loss [B, fut_ts]
    """
    pred = pred.cumsum(dim=-2)
    ego_traj_starts = pred[:, :-1, :]
    ego_traj_ends = pred
    B, T, _ = ego_traj_ends.size()
    padding_zeros = torch.zeros((B, 1, 2), dtype=pred.dtype, device=pred.device)  # initial position
    ego_traj_starts = torch.cat((padding_zeros, ego_traj_starts), dim=1)
    _, V, P, _ = target.size()
    ego_traj_expanded = ego_traj_ends.unsqueeze(2).unsqueeze(3)  # [B, T, 1, 1, 2]
    maps_expanded = target.unsqueeze(1)  # [1, 1, M, P, 2]
    dist = torch.linalg.norm(ego_traj_expanded - maps_expanded, dim=-1)  # [B, T, M, P]
    dist = dist.min(dim=-1, keepdim=False)[0]
    min_inst_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    bd_target = target.unsqueeze(1).repeat(1, pred.shape[1], 1, 1, 1)
    min_bd_insts = bd_target[batch_idxs, ts_idxs, min_inst_idxs]  # [B, T, P, 2]
    bd_inst_starts = min_bd_insts[:, :, :-1, :].flatten(0, 2)
    bd_inst_ends = min_bd_insts[:, :, 1:, :].flatten(0, 2)
    ego_traj_starts = ego_traj_starts.unsqueeze(2).repeat(1, 1, P-1, 1).flatten(0, 2)
    ego_traj_ends = ego_traj_ends.unsqueeze(2).repeat(1, 1, P-1, 1).flatten(0, 2)

    intersect_mask = segments_intersect(ego_traj_starts, ego_traj_ends,
                                        bd_inst_starts, bd_inst_ends)
    intersect_mask = intersect_mask.reshape(B, T, P-1)
    intersect_mask = intersect_mask.any(dim=-1)
    intersect_idx = (intersect_mask == True).nonzero()

    target = target.view(target.shape[0], -1, target.shape[-1])
    # [B, fut_ts, num_vec*num_pts]
    dist = torch.linalg.norm(pred[:, :, None, :] - target[:, None, :, :], dim=-1)
    min_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    min_dist = dist[batch_idxs, ts_idxs, min_idxs]
    loss = min_dist
    safe_idx = loss > dis_thresh
    unsafe_idx = loss <= dis_thresh
    loss[safe_idx] = 0
    loss[unsafe_idx] = dis_thresh - loss[unsafe_idx]

    for i in range(len(intersect_idx)):
        loss[intersect_idx[i, 0], intersect_idx[i, 1]:] = 0

    return loss


def segments_intersect(line1_start, line1_end, line2_start, line2_end):
    # Calculating the differences
    dx1 = line1_end[:, 0] - line1_start[:, 0]
    dy1 = line1_end[:, 1] - line1_start[:, 1]
    dx2 = line2_end[:, 0] - line2_start[:, 0]
    dy2 = line2_end[:, 1] - line2_start[:, 1]

    # Calculating determinants
    det = dx1 * dy2 - dx2 * dy1
    det_mask = det != 0

    # Checking if lines are parallel or coincident
    parallel_mask = torch.logical_not(det_mask)

    # Calculating intersection parameters
    t1 = ((line2_start[:, 0] - line1_start[:, 0]) * dy2 
          - (line2_start[:, 1] - line1_start[:, 1]) * dx2) / det
    t2 = ((line2_start[:, 0] - line1_start[:, 0]) * dy1 
          - (line2_start[:, 1] - line1_start[:, 1]) * dx1) / det

    # Checking intersection conditions
    intersect_mask = torch.logical_and(
        torch.logical_and(t1 >= 0, t1 <= 1),
        torch.logical_and(t2 >= 0, t2 <= 1)
    )

    # Handling parallel or coincident lines
    intersect_mask[parallel_mask] = False

    return intersect_mask


@LOSSES.register_module()
class PlanCollisionLoss(nn.Module):
    """Planning constraint to push ego vehicle away from other agents.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        agent_thresh (float, optional): confidence threshold to filter agent predictions.
        x_dis_thresh (float, optional): distance threshold between ego and other agents in x-axis.
        y_dis_thresh (float, optional): distance threshold between ego and other agents in y-axis.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        agent_thresh=0.5,
        x_dis_thresh=1.5,
        y_dis_thresh=3.0,
        point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    ):
        super(PlanCollisionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.agent_thresh = agent_thresh
        self.x_dis_thresh = x_dis_thresh
        self.y_dis_thresh = y_dis_thresh
        self.pc_range = point_cloud_range

    def forward(self,
                ego_fut_preds,
                agent_preds,
                agent_fut_preds,
                agent_score_preds,
                agent_fut_cls_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_fut_cls_preds (Tensor): [B, num_agent, fut_mode]
            agent_score_preds (Tensor): [B, num_agent, 10]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # filter agent element according to confidence score
        agent_max_score_preds, agent_max_score_idxs = agent_score_preds.max(dim=-1)
        not_valid_agent_mask = agent_max_score_preds < self.agent_thresh
        # filter low confidence preds
        agent_fut_preds[not_valid_agent_mask] = 1e6
        # filter not vehicle preds
        not_veh_pred_mask = agent_max_score_idxs > 4  # veh idxs are 0-4
        agent_fut_preds[not_veh_pred_mask] = 1e6
        # only use best mode pred
        best_mode_idxs = torch.argmax(agent_fut_cls_preds, dim=-1).tolist()
        batch_idxs = [[i] for i in range(agent_fut_cls_preds.shape[0])]
        agent_num_idxs = [[i for i in range(agent_fut_cls_preds.shape[1])] for j in range(agent_fut_cls_preds.shape[0])]
        agent_fut_preds = agent_fut_preds[batch_idxs, agent_num_idxs, best_mode_idxs]

        loss_bbox = self.loss_weight * plan_col_loss(ego_fut_preds, agent_preds,
                                                           agent_fut_preds=agent_fut_preds, weight=weight,
                                                           x_dis_thresh=self.x_dis_thresh,
                                                           y_dis_thresh=self.y_dis_thresh,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

@weighted_loss
def plan_col_loss(
    pred,
    target,
    agent_fut_preds,
    x_dis_thresh=1.5,
    y_dis_thresh=3.0,
    dis_thresh=3.0
):
    """Planning ego-agent collsion constraint.

    Args:
        pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 2].
        target (torch.Tensor): agent_preds, [B, num_agent, 2].
        agent_fut_preds (Tensor): [B, num_agent, fut_ts, 2].
        weight (torch.Tensor): [B, fut_ts, 2].
        x_dis_thresh (float, optional): distance threshold between ego and other agents in x-axis.
        y_dis_thresh (float, optional): distance threshold between ego and other agents in y-axis.
        dis_thresh (float, optional): distance threshold to filter distant agents.

    Returns:
        torch.Tensor: Calculated loss [B, fut_mode, fut_ts, 2]
    """
    pred = pred.cumsum(dim=-2)
    agent_fut_preds = agent_fut_preds.cumsum(dim=-2)
    target = target[:, :, None, :] + agent_fut_preds
    # filter distant agents from ego vehicle
    dist = torch.linalg.norm(pred[:, None, :, :] - target, dim=-1)
    dist_mask = dist > dis_thresh
    target[dist_mask] = 1e6

    # [B, num_agent, fut_ts]
    x_dist = torch.abs(pred[:, None, :, 0] - target[..., 0])
    y_dist = torch.abs(pred[:, None, :, 1] - target[..., 1])
    x_min_idxs = torch.argmin(x_dist, dim=1).tolist()
    y_min_idxs = torch.argmin(y_dist, dim=1).tolist()
    batch_idxs = [[i] for i in range(y_dist.shape[0])]
    ts_idxs = [[i for i in range(y_dist.shape[-1])] for j in range(y_dist.shape[0])]

    # [B, fut_ts]
    x_min_dist = x_dist[batch_idxs, x_min_idxs, ts_idxs]
    y_min_dist = y_dist[batch_idxs, y_min_idxs, ts_idxs]
    x_loss = x_min_dist
    safe_idx = x_loss > x_dis_thresh
    unsafe_idx = x_loss <= x_dis_thresh
    x_loss[safe_idx] = 0
    x_loss[unsafe_idx] = x_dis_thresh - x_loss[unsafe_idx]
    y_loss = y_min_dist
    safe_idx = y_loss > y_dis_thresh
    unsafe_idx = y_loss <= y_dis_thresh
    y_loss[safe_idx] = 0
    y_loss[unsafe_idx] = y_dis_thresh - y_loss[unsafe_idx]
    loss = torch.cat([x_loss.unsqueeze(-1), y_loss.unsqueeze(-1)], dim=-1)

    return loss


@LOSSES.register_module()
class PlanMapDirectionLoss(nn.Module):
    """Planning loss to force the ego heading angle consistent with lane direction.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        theta_thresh (float, optional): angle diff thresh between ego and lane.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        map_thresh=0.5,
        dis_thresh=2.0,
        lane_div_cls_idx=0,
        point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    ):
        super(PlanMapDirectionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.lane_div_cls_idx = lane_div_cls_idx
        self.pc_range = point_cloud_range

    def forward(self,
                ego_fut_preds,
                lane_preds,
                lane_score_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # filter lane element according to confidence score and class
        not_lane_div_mask = lane_score_preds[..., self.lane_div_cls_idx] < self.map_thresh
        # denormalize map pts
        lane_div_preds = lane_preds.clone()
        lane_div_preds[...,0:1] = (lane_div_preds[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
        lane_div_preds[...,1:2] = (lane_div_preds[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
        # pad not-lane-divider cls and low confidence preds
        lane_div_preds[not_lane_div_mask] = 1e6

        loss_bbox = self.loss_weight * plan_map_dir_loss(ego_fut_preds, lane_div_preds,
                                                           weight=weight, dis_thresh=self.dis_thresh,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

@weighted_loss
def plan_map_dir_loss(pred, target, dis_thresh=2.0):
    """Planning ego-map directional loss.

    Args:
        pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 2].
        target (torch.Tensor): lane_div_preds, [B, num_vec, num_pts, 2].
        weight (torch.Tensor): [B, fut_ts]

    Returns:
        torch.Tensor: Calculated loss [B, fut_ts]
    """
    num_map_pts = target.shape[2]
    pred = pred.cumsum(dim=-2)
    traj_dis = torch.linalg.norm(pred[:, -1, :] - pred[:, 0, :], dim=-1)
    static_mask = traj_dis < 1.0
    target = target.unsqueeze(1).repeat(1, pred.shape[1], 1, 1, 1)

    # find the closest map instance for ego at each timestamp
    dist = torch.linalg.norm(pred[:, :, None, None, :] - target, dim=-1)
    dist = dist.min(dim=-1, keepdim=False)[0]
    min_inst_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    target_map_inst = target[batch_idxs, ts_idxs, min_inst_idxs]  # [B, fut_ts, num_pts, 2]

    # calculate distance
    dist = torch.linalg.norm(pred[:, :, None, :] - target_map_inst, dim=-1)
    min_pts_idxs = torch.argmin(dist, dim=-1)
    min_pts_next_idxs = min_pts_idxs.clone()
    is_end_point = (min_pts_next_idxs == num_map_pts-1)
    not_end_point = (min_pts_next_idxs != num_map_pts-1)
    min_pts_next_idxs[is_end_point] = num_map_pts - 2
    min_pts_next_idxs[not_end_point] = min_pts_next_idxs[not_end_point] + 1
    min_pts_idxs = min_pts_idxs.tolist()
    min_pts_next_idxs = min_pts_next_idxs.tolist()
    traj_yaw = torch.atan2(torch.diff(pred[..., 1]), torch.diff(pred[..., 0]))  # [B, fut_ts-1]
    # last ts yaw assume same as previous
    traj_yaw = torch.cat([traj_yaw, traj_yaw[:, [-1]]], dim=-1)  # [B, fut_ts]
    min_pts = target_map_inst[batch_idxs, ts_idxs, min_pts_idxs]
    dist = torch.linalg.norm(min_pts - pred, dim=-1)
    dist_mask = dist > dis_thresh
    min_pts = min_pts.unsqueeze(2)
    min_pts_next = target_map_inst[batch_idxs, ts_idxs, min_pts_next_idxs].unsqueeze(2)
    map_pts = torch.cat([min_pts, min_pts_next], dim=2)
    lane_yaw = torch.atan2(torch.diff(map_pts[..., 1]).squeeze(-1), torch.diff(map_pts[..., 0]).squeeze(-1))  # [B, fut_ts]
    yaw_diff = traj_yaw - lane_yaw
    yaw_diff[yaw_diff > math.pi] =  yaw_diff[yaw_diff > math.pi] - math.pi
    yaw_diff[yaw_diff > math.pi/2] = yaw_diff[yaw_diff > math.pi/2] - math.pi
    yaw_diff[yaw_diff < -math.pi] = yaw_diff[yaw_diff < -math.pi] + math.pi
    yaw_diff[yaw_diff < -math.pi/2] = yaw_diff[yaw_diff < -math.pi/2] + math.pi
    yaw_diff[dist_mask] = 0  # loss = 0 if no lane around ego
    yaw_diff[static_mask] = 0  # loss = 0 if ego is static

    loss = torch.abs(yaw_diff)

    return loss  # [B, fut_ts]


# ════════════════════════════════════════════════════════════════════════════
# Collision loss v4. PlanCollisionLoss above is left untouched; this class is
#   selected only through loss_plan_col.type in the config together with the head
#   flag plan_col_v4=True.
#
#
#   L_col = (1/T) Sum_k Sum_m  nu_m * w_c * max(0, d_safe,c - dist_{k,m})   [linear hinge]
#
#   Differences with respect to the plain collision loss:
#     (1) the axis-separated hinge (|dx| and |dy| argmin-ed over *different* objects)
#         becomes a per-object sum
#     (2) centre-point distance becomes the minimum distance between multi-disc
#         approximations (box extent and yaw are taken into account)
#     (3) the class filter idx<=4 (which wrongly included traffic signs and excluded
#         pedestrians) becomes an explicit class table
#     (4) the 3.0 m distance gate is dropped: shorter than the 4.89 m ego length, it
#         made most rear-end situations invisible to the loss
#     (5) pedestrians get a corridor + stopping line instead of isotropic repulsion,
#         so the loss can only ask for deceleration (zero lateral gradient)
# ════════════════════════════════════════════════════════════════════════════
@LOSSES.register_module()
class PlanCollisionLossV4(nn.Module):
    """Planning collision constraint (v4): per-class multi-disc approximation + linear hinge.

    Args:
        loss_weight (float): w_c. In a linear hinge this equals the displacement
            authority in metres (delta* = T * g = w_c).
        agent_thresh (float): agent confidence threshold (0.5, unchanged).
        perception_detach (bool): detach the agent predictions (centre, future, yaw,
            score, dims). Default True.
        ego_size (tuple): (W, L) [m].
        dt (float): waypoint interval [s].
        n_circ (int): number of discs for vehicle-like objects and for the ego.
        radius_mode (str): 'covering' = sqrt((w/2)^2+(l/2n)^2) (fully covers the box),
            'half' = w/2.
        ped_lat_margin (float): constant term of the pedestrian corridor half-width Q.
        ped_react_time (float): velocity coefficient tau of the pedestrian stopping line.
        gt_tangent_min (float): lower bound on the GT tangent ||dg|| [m]; below it the
            fallback direction (0,1) is used.
        stop_guard (bool): off by default. When on, a planned-speed deadband removes
            standstill frames.
        stop_guard_range (tuple): deadband (v0, v1) [m/s].
    """

    # Class table - indices into the B2D class_names list
    #   0 car · 1 van · 2 truck · 3 bicycle · 4 traffic_sign · 5 traffic_cone
    #   6 traffic_light · 7 pedestrian · 8 others
    #   Classes 4, 6 and 8 get nu=0 (excluded): roadside fixtures and overhead
    #   structures are not planning constraints.
    CLS_MARGIN = {0: 0.30, 1: 0.30, 2: 0.30, 3: 0.60, 5: 0.80, 7: 1.00}
    VEH_CLS = (0, 1, 2, 3)      # multi-disc model (n_circ discs)
    PT_CLS = (5,)               # single-disc model (r = w/2)
    PED_CLS = 7                 # corridor + stopping line

    def __init__(self,
                 loss_weight=0.5,
                 agent_thresh=0.5,
                 perception_detach=True,
                 ego_size=(1.8367, 4.8924),
                 dt=0.5,
                 n_circ=3,
                 radius_mode='covering',
                 ped_lat_margin=0.35,
                 ped_react_time=0.50,
                 gt_tangent_min=0.25,
                 stop_guard=False,
                 stop_guard_range=(0.5, 2.0)):
        super(PlanCollisionLossV4, self).__init__()
        self.loss_weight = loss_weight
        self.agent_thresh = agent_thresh
        self.perception_detach = perception_detach
        self.ego_w, self.ego_l = ego_size
        self.dt = dt
        self.n = n_circ
        self.radius_mode = radius_mode
        self.ped_lat = ped_lat_margin
        self.tau = ped_react_time
        self.gt_tan_min = gt_tangent_min
        self.stop_guard = stop_guard
        self.vg0, self.vg1 = stop_guard_range

    def _radius(self, w, l):
        """Radius of the multi-disc approximation; 'covering' fully encloses the box."""
        if self.radius_mode == 'covering':
            return torch.sqrt((w / 2) ** 2 + (l / (2 * self.n)) ** 2)
        return w / 2

    def _gt_tangent(self, gt_cum):
        """GT tangent h_k (B,T,2). g_{-1} = origin. Falls back to (0,1) when ||dg|| < min.

        The GT enters the loss as a constant, so no detach is needed; because h does not
        depend on p, the pedestrian term is linear in p and its lateral gradient is
        structurally zero.
        """
        pad = torch.zeros_like(gt_cum[:, :1, :])
        d = torch.cat([gt_cum[:, :1, :] - pad, gt_cum[:, 1:, :] - gt_cum[:, :-1, :]], dim=1)
        n = d.norm(dim=-1, keepdim=True)
        fb = torch.tensor([0.0, 1.0], device=gt_cum.device, dtype=gt_cum.dtype).view(1, 1, 2)
        return torch.where(n >= self.gt_tan_min, d / n.clamp_min(1e-12), fb)

    def forward(self,
                ego_fut_preds,
                ego_fut_gt,
                agent_preds,
                agent_fut_preds,
                agent_score_preds,
                agent_fut_cls_preds,
                agent_wl,
                agent_yaw,
                weight=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2] per-step offsets (unchanged convention).
            ego_fut_gt (Tensor): [B, fut_ts, 2] GT offsets, a constant used to derive h.
            agent_preds (Tensor): [B, num_agent, 2] current centres (m).
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2] ★offset★.
            agent_score_preds (Tensor): [B, num_agent, num_cls] after sigmoid.
            agent_fut_cls_preds (Tensor): [B, num_agent, fut_mode].
            agent_wl (Tensor): [B, num_agent, 2] = (w, l) [m].
            agent_yaw (Tensor): [B, num_agent] - the rot value from denormalize_bbox
                (BEV heading = -yaw - pi/2).
            weight (Tensor, optional): [B, fut_ts] valid-timestep mask.
        """
        if self.perception_detach:
            agent_preds = agent_preds.detach()
            agent_fut_preds = agent_fut_preds.detach()
            agent_score_preds = agent_score_preds.detach()
            agent_fut_cls_preds = agent_fut_cls_preds.detach()
            agent_wl = agent_wl.detach()
            agent_yaw = agent_yaw.detach()

        ego = ego_fut_preds.cumsum(dim=-2)                          # (B,T,2) absolute
        B, T, _ = ego.shape
        N = agent_preds.shape[1]
        dev, dtp = ego.device, ego.dtype

        head = self._gt_tangent(ego_fut_gt.cumsum(dim=-2))          # (B,T,2) constant
        nrm = torch.stack([-head[..., 1], head[..., 0]], dim=-1)    # left normal

        # planned speed v_k for the pedestrian stopping line / stop guard (constant)
        pad = torch.zeros_like(ego[:, :1, :])
        spd = torch.cat([ego[:, :1, :] - pad, ego[:, 1:, :] - ego[:, :-1, :]],
                        dim=1).norm(dim=-1).detach() / self.dt      # (B,T)

        # -- ego discs (evenly spaced along the heading h) --
        off = torch.linspace(-self.ego_l / 2 + self.ego_l / (2 * self.n),
                             self.ego_l / 2 - self.ego_l / (2 * self.n),
                             self.n, device=dev, dtype=dtp)
        ec = ego.unsqueeze(2) + off.view(1, 1, self.n, 1) * head.unsqueeze(2)   # (B,T,n,2)
        r_ego = self._radius(torch.tensor(self.ego_w, device=dev, dtype=dtp),
                             torch.tensor(self.ego_l, device=dev, dtype=dtp))

        # -- agent best-mode absolute trajectory --
        conf, cls_idx = agent_score_preds.max(dim=-1)               # (B,N)
        best = agent_fut_cls_preds.argmax(dim=-1)                   # (B,N)
        gid = best.view(B, N, 1, 1, 1).expand(-1, -1, 1, T, 2)
        fut = agent_fut_preds.gather(2, gid).squeeze(2).cumsum(dim=-2)          # (B,N,T,2)
        tgt = agent_preds.unsqueeze(2) + fut                        # (B,N,T,2)

        # -- class table -> margin_c, model selection, validity weight nu_m --
        #   Do not assign 1e6 into the agent tensor in place (that would mutate the head output).
        w_o, l_o = agent_wl[..., 0], agent_wl[..., 1]               # (B,N)
        margin = torch.zeros_like(w_o)
        is_veh = torch.zeros_like(w_o, dtype=torch.bool)
        is_pt = torch.zeros_like(w_o, dtype=torch.bool)
        for c, m in self.CLS_MARGIN.items():
            sel = cls_idx == c
            margin = torch.where(sel, torch.full_like(margin, m), margin)
            if c in self.VEH_CLS:
                is_veh = is_veh | sel
            elif c in self.PT_CLS:
                is_pt = is_pt | sel
        is_ped = cls_idx == self.PED_CLS
        nu = ((conf >= self.agent_thresh) & (is_veh | is_pt | is_ped)).to(dtp)  # (B,N)

        # -- vehicle-like (multi-disc) and cones (single disc): disc-pair minimum distance hinge --
        th = -agent_yaw - math.pi / 2                               # BEV heading
        dvec = torch.stack([torch.cos(th), torch.sin(th)], dim=-1)  # (B,N,2)
        o_n = torch.linspace(-0.5 + 1.0 / (2 * self.n), 0.5 - 1.0 / (2 * self.n),
                             self.n, device=dev, dtype=dtp)         # normalised offsets
        occ = tgt.unsqueeze(3) + (o_n.view(1, 1, 1, self.n, 1)
                                  * l_o.view(B, N, 1, 1, 1) * dvec.view(B, N, 1, 1, 2))
        occ = torch.where(is_veh.view(B, N, 1, 1, 1), occ, tgt.unsqueeze(3))   # cone / pedestrian = centre
        r_obj = torch.where(is_veh, self._radius(w_o, l_o), w_o / 2)           # (B,N)
        # (B,1,T,n,1,2) - (B,N,T,1,n,2) -> (B,N,T,n,n): every ego disc x object disc pair
        d = (ec.unsqueeze(1).unsqueeze(4) - occ.unsqueeze(3)).norm(dim=-1)     # (B,N,T,n,n)
        dist = d.flatten(3).min(dim=-1).values                                 # (B,N,T)
        x_circ = ((r_ego + r_obj.unsqueeze(-1) + margin.unsqueeze(-1)) - dist).clamp_min(0.0)

        # -- pedestrians: corridor (|l| <= Q) and forward (s > 0) -> longitudinal stopping hinge --
        rel = tgt - ego.unsqueeze(1)                                # (B,N,T,2)
        lat = (rel * nrm.unsqueeze(1)).sum(dim=-1).abs()
        lon = (rel * head.unsqueeze(1)).sum(dim=-1)
        Q = (self.ego_w + w_o).unsqueeze(-1) / 2 + self.ped_lat
        S = ((self.ego_l + l_o).unsqueeze(-1) / 2 + margin.unsqueeze(-1)
             + self.tau * spd.unsqueeze(1))
        x_ped = torch.where((lat <= Q) & (lon > 0),
                            (S - lon.clamp_min(0.0)).clamp_min(0.0),
                            torch.zeros_like(lon))

        x = torch.where(is_ped.unsqueeze(-1), x_ped, x_circ)        # (B,N,T)
        per_t = (nu.unsqueeze(-1) * x).sum(dim=1)                   # (B,T)

        if self.stop_guard:                                         # off by default
            per_t = per_t * ((spd - self.vg0) / (self.vg1 - self.vg0)).clamp(0.0, 1.0)
        if weight is not None:
            per_t = per_t * (weight[..., 0] if weight.dim() == 3 else weight)

        return self.loss_weight * per_t.mean()


# ════════════════════════════════════════════════════════════════════════════
# Collision loss v7. PlanCollisionLoss and PlanCollisionLossV4 above are left
#   untouched; this class is selected only through loss_plan_col.type in the config
#   together with the head flag plan_col_v7=True.
#
#
#   L = (1/T) Σ_k [ Σ_{m∈V} ν_m Σ_i Σ_j max(0, D − ‖c_{k,i} − o_{m,k,j}‖)
#                 + Σ_{m∈P} ν_m Σ_i Σ_j 1[s_ij>0] max(0, D − sqrt(s_ij² + sg(ℓ_ij)²)) ]
#
#   Differences with respect to v4:
#     (1) the multi-disc approximation becomes disc *filling* (r = w/2, n = ceil(l/w)),
#         replacing the 3-disc covering radius, whose larger radius pushed the ego
#         into the neighbouring lane.
#     (2) the safety distance is a single shared D = 3 m for every object, replacing
#         the sum of radii plus a per-class margin.
#     (3) pedestrians lose the corridor: same hinge, same D and same disc filling as
#         vehicles, but with dist = sqrt(s^2 + sg(l)^2), i.e. only the lateral component
#         is stop-gradient. The value stays isotropic while the requested motion is
#         purely longitudinal.
#         Only the forward gate 1[s>0] remains (disable it with ped_gate=False).
#         sg(l) makes the field non-conservative on purpose: grad L is not the gradient
#         of any scalar function.
#         Sign: c = p + off*h, so ds/dp = -h and dx/dp = +(s/dist)*h - deceleration only.
#     (4) agg is exposed as a config argument ('pairsum' by default, 'min' also
#         available). Both fire on the same frames and differ only in magnitude and direction.
# ════════════════════════════════════════════════════════════════════════════
@LOSSES.register_module()
class PlanCollisionLossV7(nn.Module):
    """Planning collision constraint (v7): disc filling + fixed D = 3 m + longitudinal pedestrian gradient.

    Args:
        loss_weight (float): w_c. In a linear hinge this equals the displacement
            authority in metres (delta* = T * g = w_c).
        dis_thresh (float): D [m], shared by every object (pedestrians included, no
            per-class margin).
        agg (str): 'pairsum' sums every firing disc pair, 'min' keeps the nearest one.
        ped_gate (bool): pedestrian forward gate 1[s>0]. With False, pedestrians behind
            the ego would produce an acceleration signal.
        agent_thresh (float): agent confidence threshold (0.5, unchanged).
        n_max (int): maximum number of discs per object.
        perception_detach (bool): detach the agent predictions (centre, future, yaw,
            score, dims). Default True.
        ego_size (tuple): (W, L) [m].
        gt_tangent_min (float): lower bound on the GT tangent ||dg|| [m]; below it the
            fallback direction (0,1) is used.
        eps (float): sqrt stabilisation constant.
    """

    # Class table - indices into the B2D class_names list
    #   0 car · 1 van · 2 truck · 3 bicycle · 4 traffic_sign · 5 traffic_cone
    #   6 traffic_light · 7 pedestrian · 8 others
    #   Classes 4, 6 and 8 get nu=0 (excluded): roadside fixtures and overhead
    #   structures are not planning constraints.
    TARGET_CLS = (0, 1, 2, 3, 5, 7)     # car van truck bicycle cone pedestrian
    PED_CLS = 7

    def __init__(self,
                 loss_weight=0.5,
                 dis_thresh=3.0,
                 agg='pairsum',
                 ped_gate=True,
                 agent_thresh=0.5,
                 n_max=6,
                 perception_detach=True,
                 ego_size=(1.8367, 4.8924),
                 gt_tangent_min=0.25,
                 eps=1e-12):
        super(PlanCollisionLossV7, self).__init__()
        assert agg in ('pairsum', 'min')
        self.loss_weight = loss_weight
        self.D = dis_thresh
        self.agg = agg
        self.ped_gate = ped_gate
        self.agent_thresh = agent_thresh
        self.n_max = n_max
        self.perception_detach = perception_detach
        self.ego_w, self.ego_l = ego_size
        self.gt_tan_min = gt_tangent_min
        self.eps = eps
        ne, re = self._fit(torch.tensor(self.ego_w, dtype=torch.float64),
                           torch.tensor(self.ego_l, dtype=torch.float64))
        self.ego_n, self.ego_r = int(ne), float(re)

    # -- disc filling rule: r = short side / 2, n = ceil(long side / short side) ------
    def _fit(self, w, l):
        short = l < w
        r = torch.where(short, l / 2, w / 2)
        n = torch.where(short, torch.ones_like(l),
                        torch.ceil(l / w.clamp_min(1e-6) - 1e-9))
        return n.clamp(1, self.n_max), r

    def _offsets(self, l, n, r, slots):
        """Variable n -> fixed slots (n_max) plus a validity mask; disc centres are evenly spaced along the long axis."""
        a = (l / 2 - r).unsqueeze(-1)
        j = slots.view(*([1] * l.dim()), -1).to(l.dtype)
        nn_ = n.unsqueeze(-1)
        t = torch.where(nn_ > 1, j / (nn_ - 1).clamp_min(1.0), torch.zeros_like(j))
        off = torch.where(nn_ > 1, -a + 2 * a * t, torch.zeros_like(j))
        return off, j < nn_

    def _gt_tangent(self, gt_cum):
        """GT tangent h_k (B,T,2). g_{-1} = origin. Falls back to (0,1) when ||dg|| < min.

        The GT enters the loss as a constant, so no detach is needed; because h does not
        depend on p, only the longitudinal component s of the pedestrian term is
        differentiated through p.
        """
        pad = torch.zeros_like(gt_cum[:, :1, :])
        d = torch.cat([gt_cum[:, :1, :] - pad, gt_cum[:, 1:, :] - gt_cum[:, :-1, :]], dim=1)
        n = d.norm(dim=-1, keepdim=True)
        fb = torch.tensor([0.0, 1.0], device=gt_cum.device, dtype=gt_cum.dtype).view(1, 1, 2)
        return torch.where(n >= self.gt_tan_min, d / n.clamp_min(1e-12), fb)

    def forward(self,
                ego_fut_preds,
                ego_fut_gt,
                agent_preds,
                agent_fut_preds,
                agent_score_preds,
                agent_fut_cls_preds,
                agent_wl,
                agent_yaw,
                weight=None):
        """Forward function. The argument contract is identical to v4 (shared head branch).

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2] per-step offsets (unchanged convention).
            ego_fut_gt (Tensor): [B, fut_ts, 2] GT offsets, a constant used to derive h.
            agent_preds (Tensor): [B, num_agent, 2] current centres (m).
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2] ★offset★.
            agent_score_preds (Tensor): [B, num_agent, num_cls] after sigmoid.
            agent_fut_cls_preds (Tensor): [B, num_agent, fut_mode].
            agent_wl (Tensor): [B, num_agent, 2] = (w, l) [m].
            agent_yaw (Tensor): [B, num_agent] - the rot value from denormalize_bbox
                (BEV heading = -yaw - pi/2).
            weight (Tensor, optional): [B, fut_ts] valid-timestep mask.
        """
        if self.perception_detach:
            agent_preds = agent_preds.detach()
            agent_fut_preds = agent_fut_preds.detach()
            agent_score_preds = agent_score_preds.detach()
            agent_fut_cls_preds = agent_fut_cls_preds.detach()
            agent_wl = agent_wl.detach()
            agent_yaw = agent_yaw.detach()

        ego = ego_fut_preds.cumsum(dim=-2)                          # (B,T,2) absolute
        B, T, _ = ego.shape
        N = agent_preds.shape[1]
        dev, dtp = ego.device, ego.dtype

        head = self._gt_tangent(ego_fut_gt.cumsum(dim=-2))          # (B,T,2) constant
        nrm = torch.stack([-head[..., 1], head[..., 0]], dim=-1)    # left normal

        # -- ego disc filling (evenly spaced along the heading h) --
        a_e = self.ego_l / 2 - self.ego_r
        eoff = (torch.zeros(1, device=dev, dtype=dtp) if self.ego_n == 1 else
                torch.linspace(-a_e, a_e, self.ego_n, device=dev, dtype=dtp))
        ec = ego.unsqueeze(2) + eoff.view(1, 1, -1, 1) * head.unsqueeze(2)      # (B,T,ne,2)

        # -- agent best-mode absolute trajectory --
        conf, cls_idx = agent_score_preds.max(dim=-1)               # (B,N)
        best = agent_fut_cls_preds.argmax(dim=-1)                   # (B,N)
        gid = best.view(B, N, 1, 1, 1).expand(-1, -1, 1, T, 2)
        fut = agent_fut_preds.gather(2, gid).squeeze(2).cumsum(dim=-2)          # (B,N,T,2)
        tgt = agent_preds.unsqueeze(2) + fut                        # (B,N,T,2)

        # -- class table -> validity weight nu_m --
        #   Do not assign 1e6 into the agent tensor in place (that would mutate the head output).
        w_o, l_o = agent_wl[..., 0], agent_wl[..., 1]               # (B,N)
        is_tgt = torch.zeros_like(w_o, dtype=torch.bool)
        for c in self.TARGET_CLS:
            is_tgt = is_tgt | (cls_idx == c)
        is_ped = cls_idx == self.PED_CLS
        nu = ((conf >= self.agent_thresh) & is_tgt).to(dtp)         # (B,N)

        # -- object disc filling (the same rule applies to pedestrians) --
        n_o, r_o = self._fit(w_o, l_o)
        slots = torch.arange(self.n_max, device=dev)
        off, valid = self._offsets(l_o, n_o, r_o, slots)            # (B,N,n_max)
        th = -agent_yaw - math.pi / 2                               # BEV heading
        dvec = torch.stack([torch.cos(th), torch.sin(th)], dim=-1)  # (B,N,2)
        occ = tgt.unsqueeze(3) + off.view(B, N, 1, self.n_max, 1) * dvec.view(B, N, 1, 1, 2)

        # -- disc-pair vectors delta = o - c   (B,N,T,ne,n_max,2) --
        dl = occ.unsqueeze(3) - ec.unsqueeze(1).unsqueeze(4)
        d_iso = dl.norm(dim=-1)                                     # isotropic distance
        # (s, l) decomposition for pedestrians: only l is detached, so the value matches
        #   d_iso while the gradient flows through s alone
        s = (dl * head.view(B, 1, T, 1, 1, 2)).sum(dim=-1)
        lat = (dl * nrm.view(B, 1, T, 1, 1, 2)).sum(dim=-1)
        d_ped = torch.sqrt(s ** 2 + lat.detach() ** 2 + self.eps)

        d = torch.where(is_ped.view(B, N, 1, 1, 1), d_ped, d_iso)
        h = (self.D - d).clamp_min(0.0)
        h = h * valid.view(B, N, 1, 1, self.n_max).to(dtp)          # drop invalid slots
        if self.ped_gate:
            gate = torch.where(is_ped.view(B, N, 1, 1, 1), (s > 0).to(dtp),
                               torch.ones_like(h))
            h = h * gate

        if self.agg == 'pairsum':
            x = h.flatten(3).sum(dim=-1)                            # (B,N,T)
        else:
            x = h.flatten(3).max(dim=-1).values

        per_t = (nu.unsqueeze(-1) * x).sum(dim=1)                   # (B,T)
        if weight is not None:
            per_t = per_t * (weight[..., 0] if weight.dim() == 3 else weight)

        return self.loss_weight * per_t.mean()
