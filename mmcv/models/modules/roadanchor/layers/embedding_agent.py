# mmcv/models/modules/roadanchor/layers/embedding_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# ============================================
# Pure PyTorch replacement for natten
# ============================================
class NeighborhoodAttention1D(nn.Module):
    """
    Neighborhood Attention 1D (Pure PyTorch Implementation)
    
    Each position attends only to the tokens inside a kernel_size window
    (Sliding window attention)
    """
    def __init__(
        self,
        dim,
        kernel_size=7,
        dilation=1,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        # kernel_size must be odd (symmetric around the centre)
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.dilation = dilation or 1
        self.window_size = self.kernel_size
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, C) - Batch, Length, Channel
        
        Returns:
            out: (B, L, C)
        """
        B, L, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, L, head_dim)
        
        # ============================================
        # Neighborhood Attention (Sliding Window)
        # ============================================
        q = q * self.scale
        
        # Padding for boundary handling
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size - 1 - pad_l
        
        # Pad K, V
        k_padded = F.pad(k, (0, 0, pad_l, pad_r), mode='constant', value=0)
        v_padded = F.pad(v, (0, 0, pad_l, pad_r), mode='constant', value=0)
        
        # Compute attention for each position
        attn_outputs = []
        for i in range(L):
            # Get local window
            start_idx = i
            end_idx = i + self.kernel_size
            
            k_local = k_padded[:, :, start_idx:end_idx, :]  # (B, H, K, D)
            v_local = v_padded[:, :, start_idx:end_idx, :]  # (B, H, K, D)
            
            q_i = q[:, :, i:i+1, :]  # (B, H, 1, D)
            
            # Attention scores
            attn = (q_i @ k_local.transpose(-2, -1))  # (B, H, 1, K)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            # Weighted sum
            out_i = (attn @ v_local)  # (B, H, 1, D)
            attn_outputs.append(out_i)
        
        # Concatenate all positions
        out = torch.cat(attn_outputs, dim=2)  # (B, H, L, D)
        out = out.transpose(1, 2).reshape(B, L, C)  # (B, L, C)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


# ============================================
# Optimised variant (faster)
# ============================================
class NeighborhoodAttention1D_Fast(nn.Module):
    """
    Faster implementation using unfold
    """
    def __init__(
        self,
        dim,
        kernel_size=7,
        dilation=1,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.dilation = dilation or 1
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, C)
        Returns:
            out: (B, L, C)
        """
        B, L, C = x.shape
        
        # QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, D)
        
        q = q * self.scale
        
        # ============================================
        # Use unfold for efficient sliding window
        # ============================================
        pad = (self.kernel_size - 1) // 2
        
        # Reshape for unfold: (B*H, D, L)
        k = k.transpose(2, 3).reshape(B * self.num_heads, self.head_dim, L)
        v = v.transpose(2, 3).reshape(B * self.num_heads, self.head_dim, L)
        
        # Pad
        k = F.pad(k, (pad, pad), mode='constant', value=0)
        v = F.pad(v, (pad, pad), mode='constant', value=0)
        
        # Unfold: extract sliding windows
        # (B*H, D, L) -> (B*H, D*K, L)
        k_unfold = F.unfold(
            k.unsqueeze(-1),  # (B*H, D, L, 1)
            kernel_size=(self.kernel_size, 1),
            padding=0
        ).reshape(B * self.num_heads, self.head_dim, self.kernel_size, L)
        
        v_unfold = F.unfold(
            v.unsqueeze(-1),
            kernel_size=(self.kernel_size, 1),
            padding=0
        ).reshape(B * self.num_heads, self.head_dim, self.kernel_size, L)
        
        # Reshape back
        k_unfold = k_unfold.permute(0, 3, 2, 1)  # (B*H, L, K, D)
        v_unfold = v_unfold.permute(0, 3, 2, 1)  # (B*H, L, K, D)
        
        q = q.reshape(B * self.num_heads, L, 1, self.head_dim)
        
        # Attention
        attn = (q @ k_unfold.transpose(-2, -1))  # (B*H, L, 1, K)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Weighted sum
        out = (attn @ v_unfold).squeeze(2)  # (B*H, L, D)
        
        # Reshape
        out = out.reshape(B, self.num_heads, L, self.head_dim)
        out = out.transpose(1, 2).reshape(B, L, C)
        
        # Projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


# ============================================
# NATSequenceEncoder (modified)
# ============================================
class NATSequenceEncoder(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=32,
        mlp_ratio=3,
        kernel_size=[3, 3, 5],
        depths=[2, 2, 2],
        num_heads=[2, 4, 8],
        out_indices=[0, 1, 2],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        use_fast_version=False,  # use the optimised variant
    ) -> None:
        super().__init__()

        self.embed = ConvTokenizer(in_chans, embed_dim)
        self.num_levels = len(depths)
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.out_indices = out_indices
        self.use_fast_version = use_fast_version

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size[i],
                dilations=None,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                use_fast_version=use_fast_version,
            )
            self.levels.append(level)

        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        n = self.num_features[-1]
        self.lateral_convs = nn.ModuleList()
        for i_layer in self.out_indices:
            self.lateral_convs.append(
                nn.Conv1d(self.num_features[i_layer], n, 3, padding=1)
            )

        self.fpn_conv = nn.Conv1d(n, n, 3, padding=1)

    def forward(self, x):
        """x: [B, C, T]"""
        x = self.embed(x)

        out = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                out.append(x_out.permute(0, 2, 1).contiguous())

        laterals = [
            lateral_conv(out[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(out) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                scale_factor=(laterals[i - 1].shape[-1] / laterals[i].shape[-1]),
                mode="linear",
                align_corners=False,
            )

        out = self.fpn_conv(laterals[0])

        return out[:, :, -1]


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=32, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 1)  # B, C, L -> B, L, C
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv1d(
            dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_fast_version=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        
        # fast vs standard
        AttnClass = NeighborhoodAttention1D_Fast if use_fast_version else NeighborhoodAttention1D
        
        self.attn = AttnClass(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_fast_version=True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_fast_version=use_fast_version,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class PointsEncoder(nn.Module):
    def __init__(self, feat_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_mlp = nn.Sequential(
            nn.Linear(feat_channel, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.encoder_channel),
        )

    def forward(self, x, mask=None):
        """
        x : B M 3
        mask: B M
        -----------------
        feature_global : B C
        """

        bs, n, _ = x.shape
        device = x.device

        x_valid = self.first_mlp(x[mask])  # B n 256
        x_features = torch.zeros(bs, n, 256, device=device)
        x_features[mask] = x_valid

        pooled_feature = x_features.max(dim=1)[0]
        x_features = torch.cat(
            [x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1
        )

        x_features_valid = self.second_mlp(x_features[mask])
        res = torch.zeros(bs, n, self.encoder_channel, device=device)
        res[mask] = x_features_valid

        res = res.max(dim=1)[0]
        return res