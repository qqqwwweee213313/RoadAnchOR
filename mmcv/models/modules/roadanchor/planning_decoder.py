from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .layers.embedding import PointsEncoder
from .layers.fourier_embedding import FourierEmbedding
from .layers.mlp_layer import MLPLayer

def safe_atan2(y, x, eps=1e-6):
    """ONNX / TensorRT friendly safe atan2."""
    # magnitude of the vector
    r = torch.sqrt(x*x + y*y + eps)
    
    # return 0 for very small vectors
    angle = torch.where(
        r < eps, 
        torch.zeros_like(r),
        torch.atan2(y, torch.clamp(x, min=eps))  # keep x away from zero
    )
    return angle

class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout) -> None:
        super().__init__()
        self.dim = dim

        self.r2r_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.m2m_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.cross_attn_weights = None

    def forward(
        self,
        tgt,
        memory,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
    ):
        """
        tgt: (bs, R, M, dim)
        tgt_key_padding_mask: (bs, R)
        """
        bs, R, M, D = tgt.shape

        tgt = tgt.transpose(1, 2).reshape(bs * M, R, D)
        tgt2 = self.norm1(tgt)
        tgt2 = self.r2r_attn(
            tgt2, tgt2, tgt2, key_padding_mask=tgt_key_padding_mask.repeat(M, 1)
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt_tmp = tgt.reshape(bs, M, R, D).transpose(1, 2).reshape(bs * R, M, D)
        tgt_valid_mask = ~tgt_key_padding_mask.reshape(-1)
        tgt_valid = tgt_tmp[tgt_valid_mask]
        tgt2_valid = self.norm2(tgt_valid)
        tgt2_valid, _ = self.m2m_attn(
            tgt2_valid + m_pos, tgt2_valid + m_pos, tgt2_valid
        )
        tgt_valid = tgt_valid + self.dropout2(tgt2_valid)
        tgt = torch.zeros_like(tgt_tmp)
        tgt[tgt_valid_mask] = tgt_valid

        tgt = tgt.reshape(bs, R, M, D).view(bs, R * M, D)
        tgt2 = self.norm3(tgt)
        # tgt2 = self.cross_attn(
        #     tgt2, memory, memory, key_padding_mask=memory_key_padding_mask
        # )[0]
        tgt2, cross_weights = self.cross_attn(
            tgt2, memory, memory, 
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,  # ✅ Get attention weights
            average_attn_weights=True
        )
        
        self.cross_attn_weights = cross_weights.detach()  # (bs, R*M, N)

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm4(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt.reshape(bs, R, M, D)

        return tgt
class DecoderLayer_d0(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout) -> None:
        super().__init__()
        self.dim = dim

        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.cross_attn_weights = None

    def forward(
        self,
        tgt,
        memory,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
    ):
        """
        tgt: (bs, R, M, dim)
        tgt_key_padding_mask: (bs, R)
        m_pos: (1, M, dim) - mode positional embedding (unused in this simplified version)
        
        Combines R2R and M2M attention into a single self-attention over R*M tokens
        """
        bs, R, M, D = tgt.shape

        # Flatten (R, M) into a single sequence dimension: (bs, R*M, D)
        tgt = tgt.view(bs, R * M, D)
        
        # Create padding mask for R*M tokens
        # If a reference line is padding (True), all its M modes are padding
        if tgt_key_padding_mask is not None:
            # (bs, R) -> (bs, R, M) -> (bs, R*M)
            tgt_key_padding_mask_expanded = tgt_key_padding_mask.unsqueeze(-1).expand(bs, R, M).reshape(bs, R * M)
        else:
            tgt_key_padding_mask_expanded = None
        
        # Self-attention over all R*M reference line-mode combinations
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            tgt2, tgt2, tgt2, 
            key_padding_mask=tgt_key_padding_mask_expanded
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        # Cross-attention with scene context
        tgt2 = self.norm2(tgt)
        tgt2, cross_weights = self.cross_attn(
            tgt2, memory, memory, 
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        self.cross_attn_weights = cross_weights.detach()  # (bs, R*M, N)

        tgt = tgt + self.dropout2(tgt2)
        
        # Feed-forward network
        tgt2 = self.norm3(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        
        # Reshape back to (bs, R, M, D)
        tgt = tgt.view(bs, R, M, D)

        return tgt
class PlanningDecoder(nn.Module):
    def __init__(
        self,
        num_mode,
        decoder_depth,
        dim,
        num_heads,
        mlp_ratio,
        dropout,
        future_steps,
        wo_ref_embed_flag=False,
    ) -> None:
        super().__init__()

        self.num_mode = num_mode
        self.future_steps = future_steps

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(decoder_depth)
            ]
        )

        # set to True to run without any reference-line information
        self.wo_ref_embed_flag = wo_ref_embed_flag
        
        # print wo_ref_embed_flag with cyan color
        print (f"\033[96m wo_ref_embed_flag: {self.wo_ref_embed_flag} \033[0m")
        if self.wo_ref_embed_flag is True:
            print (f"\033[96m Centerline-guided Lat Query Initialization is NOT used in PlanningDecoder. \033[0m")
        else:
            print (f"\033[96m Centerline-guided Lat Query Initialization is used in PlanningDecoder. \033[0m")

        if self.wo_ref_embed_flag is False:
            self.r_pos_emb = FourierEmbedding(3, dim, 64)

        self.r_encoder = PointsEncoder(6, dim)

        self.q_proj = nn.Linear(2 * dim, dim)

        self.m_emb = nn.Parameter(torch.Tensor(1, 1, num_mode, dim))
        self.m_pos = nn.Parameter(torch.Tensor(1, num_mode, dim))
        
        self.loc_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        # self.yaw_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        # self.vel_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.pi_head = MLPLayer(dim, dim, 1)
        
        self.dim = dim

        nn.init.normal_(self.m_emb, mean=0.0, std=0.01)
        nn.init.normal_(self.m_pos, mean=0.0, std=0.01)

    def forward(self, reference_data, reference_mask, enc_emb, enc_key_padding_mask, latent_feature_alignment=False):
        # M2M inputs
        bs, R, P, C = reference_data.shape
        
        if self.wo_ref_embed_flag is False:
            # compute the vectors
            reference_vector = torch.zeros_like(reference_data) 
            reference_vector[:, :, :-1, :] = reference_data[:, :, 1:, :] - reference_data[:, :, :-1, :] # TODO: static indexing once the reference-line length is fixed
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
            
        # set to True to run without any reference-line information
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