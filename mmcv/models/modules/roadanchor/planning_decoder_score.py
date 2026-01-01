# ============================================================================
# planning_decoder_score.py
# ----------------------------------------------------------------------------
# Subclass of PlanningDecoder; the base class is not modified.
#
# [the single difference]
#   A per-centerline score embedding is added to the reference-line embedding:
#     before: r_emb = r_encoder(geometry 6ch) + r_pos_emb([x0, y0, theta0])
#     now:    r_emb = r_encoder(geometry 6ch) + r_pos_emb([x0, y0, theta0]) + score_emb(score)
#   The score comes from the goal-conditioned score head (in [0,1], shape (bs,R)).
#   With ref_score=None the behaviour is identical to the base class.
#
# [why forward is copied]
#   The injection point sits in the middle of forward (right after r_emb is built),
#   so it cannot be hooked. The parent body is copied verbatim and three lines are
#   inserted; logic, ordering and shapes are otherwise unchanged.
# ============================================================================
from typing import Optional

import torch
from torch import Tensor

from .layers.fourier_embedding import FourierEmbedding
from .planning_decoder import PlanningDecoder, safe_atan2


class PlanningDecoderScore(PlanningDecoder):
    """PlanningDecoder + reference-line score embedding.

    The arguments are exactly those of PlanningDecoder (no extra hyper-parameter).
    """

    def __init__(self, *args, **kwargs) -> None:
        super(PlanningDecoderScore, self).__init__(*args, **kwargs)
        # score (1 scalar channel) -> dim, matching the r_pos_emb convention.
        self.score_emb = FourierEmbedding(1, self.dim, 64)
        print("\033[96m PlanningDecoderScore: r_emb += score_emb(ref_score) enabled. \033[0m")

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

        # The only addition: inject the per-centerline score into the reference embedding.
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
            pi = self.pi_head(q).squeeze(-1)

        traj = torch.cat([loc], dim=-1)

        if latent_feature_alignment:
            return traj, pi, r_emb
        else:
            return traj, pi
