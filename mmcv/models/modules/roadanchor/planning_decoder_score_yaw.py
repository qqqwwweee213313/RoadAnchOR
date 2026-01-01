# ============================================================================
# planning_decoder_score_yaw.py
# ----------------------------------------------------------------------------
# Planning decoder with a reference-line score embedding and a heading head.
#   It extends PlanningDecoderScore, which itself extends the plain
#   PlanningDecoder; neither of those is modified.
#
# [difference with respect to PlanningDecoderScore]
#   (1) __init__ : self.yaw_head = MLPLayer(dim, 2*dim, future_steps*2)
#         The heading head that is commented out in planning_decoder.py is
#         enabled here. There is no velocity head: the dataset has no reliable
#         aligned speed ground truth.
#   (2) forward  : traj = torch.cat([loc, yaw], -1)  → (bs, R, M, T, ★4★)
#         The reference formulation emits [px, py, cos(t), sin(t), vx, vy] and
#         supervises heading directly; we use position 2 + heading 2 = 4 channels.
#
#   Everything else (logic, ordering, shapes, the score injection point) is
#   unchanged; no new hyper-parameter is introduced (future_steps and dim are reused).
#
# [channel contract - consumers must respect this]
#   traj[..., 0:2] = position (cumulative, ego frame, metres) <- driving,
#                    evaluation and visualisation use only this
#   traj[..., 2:4] = heading (cos, sin), raw regression <- supervision, dumps
#                    and visualisation only
# ============================================================================
from typing import Optional

import torch
from torch import Tensor

from .layers.fourier_embedding import FourierEmbedding
from .layers.mlp_layer import MLPLayer
from .planning_decoder import PlanningDecoder, safe_atan2


class PlanningDecoderScoreYaw(PlanningDecoder):
    """PlanningDecoder + reference-line score embedding + heading (yaw) head.

    The arguments are exactly those of PlanningDecoder (no extra hyper-parameter).
    """

    def __init__(self, *args, **kwargs) -> None:
        super(PlanningDecoderScoreYaw, self).__init__(*args, **kwargs)
        # score (1 scalar channel) -> dim, matching the r_pos_emb convention.
        self.score_emb = FourierEmbedding(1, self.dim, 64)
        # Heading head, same structure and width as loc_head.
        self.yaw_head = MLPLayer(self.dim, 2 * self.dim, self.future_steps * 2)
        print("\033[96m PlanningDecoderScoreYaw: r_emb += score_emb(ref_score), "
              "traj = [loc(2), yaw(cos,sin)] enabled (no velocity head).\033[0m")

    def forward(
        self,
        reference_data,
        reference_mask,
        enc_emb,
        enc_key_padding_mask,
        latent_feature_alignment=False,
        ref_score: Optional[Tensor] = None,
    ):
        """
        reference_data : (bs, R, P, 2)  selected reference centerlines (metric, ego frame)
        reference_mask : (bs, R)        True = padding
        ref_score      : (bs, R) or None - predicted score of the selected centerlines, in [0,1]

        Returns
        -------
        traj : (bs, R, M, T, ★4★)  [x, y, cosθ, sinθ]
        pi   : (bs, R, M)
        """
        # M2M inputs
        bs, R, P, C = reference_data.shape

        if self.wo_ref_embed_flag is False:
            # compute the vectors
            reference_vector = torch.zeros_like(reference_data)
            reference_vector[:, :, :-1, :] = reference_data[:, :, 1:, :] - reference_data[:, :, :-1, :]
            reference_vector[:, :, -1, :] = reference_vector[:, :, -2, :]

            # compute the orientation (angle of the direction vector)
            r_orientation = safe_atan2(reference_vector[:, :, :, 1], reference_vector[:, :, :, 0])  # (bs, R, P)

            r_feature = torch.cat(
                [
                    reference_data,
                    reference_vector,
                    torch.stack([r_orientation.cos(), r_orientation.sin()], dim=-1),
                ],
                dim=-1,
            )  # (bs, R, P, 6)
        else:
            device = reference_data.device
            r_feature = torch.zeros(bs, R, P, 6).to(device)

        reference_valid_mask = ~reference_mask

        bs, R, P, C = r_feature.shape
        r_valid_mask = reference_valid_mask.unsqueeze(-1).expand(bs, R, P)
        r_valid_mask = r_valid_mask.view(bs * R, P)
        r_feature = r_feature.reshape(bs * R, P, C)

        # Input: Reference pos, Reference vector, Reference orientation
        r_emb = self.r_encoder(r_feature, r_valid_mask).view(bs, R, -1)

        if self.wo_ref_embed_flag is False:
            r_pos = torch.cat([reference_data[:, :, 0], r_orientation[:, :, 0, None]], dim=-1)
            r_emb = r_emb + self.r_pos_emb(r_pos)

        # Score injection: add the per-centerline score to the reference embedding.
        if ref_score is not None:
            r_emb = r_emb + self.score_emb(ref_score.unsqueeze(-1).to(r_emb.dtype))  # (bs,R,dim)

        r_emb = r_emb.unsqueeze(2).repeat(1, 1, self.num_mode, 1)

        m_emb = self.m_emb.repeat(bs, R, 1, 1)

        q = self.q_proj(torch.cat([r_emb, m_emb], dim=-1))

        if latent_feature_alignment:
            with torch.no_grad():
                for blk in self.decoder_blocks:
                    q = blk(
                        q,
                        enc_emb,
                        tgt_key_padding_mask=reference_mask,
                        memory_key_padding_mask=enc_key_padding_mask,
                        m_pos=self.m_pos,
                    )
                    assert torch.isfinite(q).all()

                loc = self.loc_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
                yaw = self.yaw_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
                pi = self.pi_head(q).squeeze(-1)
        else:
            for blk in self.decoder_blocks:
                q = blk(
                    q,
                    enc_emb,
                    tgt_key_padding_mask=reference_mask,
                    memory_key_padding_mask=enc_key_padding_mask,
                    m_pos=self.m_pos,
                )
                assert torch.isfinite(q).all()

            loc = self.loc_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
            yaw = self.yaw_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
            pi = self.pi_head(q).squeeze(-1)

        # Position 2ch + heading 2ch (no velocity channels on this axis).
        traj = torch.cat([loc, yaw], dim=-1)

        if latent_feature_alignment:
            return traj, pi, r_emb
        else:
            return traj, pi
