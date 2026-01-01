import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output



class LocalConvAttention1D(nn.Module):
    """
    Local attention using depthwise separable convolutions as replacement for NeighborhoodAttention1D
    """
    def __init__(
        self,
        dim,
        kernel_size=7,
        dilation=1,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else (self.head_dim ** -0.5)
        self.kernel_size = kernel_size
        self.dilation = dilation if dilation is not None else 1
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Local convolution for positional encoding
        padding = (kernel_size - 1) * self.dilation // 2
        self.local_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, 
            padding=padding, groups=dim, dilation=self.dilation
        )
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, L, C = x.shape
        
        # Generate QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, L, head_dim
        
        # Add local positional information through convolution
        x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)  # B, L, C
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, num_heads, L, L
        
        # Apply local mask to simulate neighborhood attention
        mask = self._create_local_mask(L, self.kernel_size, x.device)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)  # B, L, C
        
        # Combine with local convolution features
        out = out + x_conv
        
        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out
    
    def _create_local_mask(self, seq_len, kernel_size, device):
        """Create a local attention mask"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        half_k = kernel_size // 2
        
        for i in range(seq_len):
            start = max(0, i - half_k)
            end = min(seq_len, i + half_k + 1)
            mask[i, start:end] = 1
            
        return mask.unsqueeze(0).unsqueeze(0)  # 1, 1, L, L


class LocalSequenceEncoder(nn.Module):
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
    ) -> None:
        super().__init__()

        self.embed = ConvTokenizer(in_chans, embed_dim)
        self.num_levels = len(depths)
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = LocalBlock(
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
            )
            self.levels.append(level)

        self.norms = nn.ModuleList([norm_layer(self.num_features[i]) for i in range(self.num_levels)])

        n = self.num_features[-1]
        # Fixed 3 lateral convs for TensorRT - always process indices [0, 1, 2]
        self.lateral_convs = nn.ModuleList([
            nn.Conv1d(self.num_features[0], n, 3, padding=1),
            nn.Conv1d(self.num_features[1], n, 3, padding=1),
            nn.Conv1d(self.num_features[2], n, 3, padding=1)
        ])

        self.fpn_conv = nn.Conv1d(n, n, 3, padding=1)

    def forward(self, x):
        """x: [B, C, T]"""
        x = self.embed(x)

        out = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = self.norms[idx]
                x_out = norm_layer(xo)
                out.append(x_out.permute(0, 2, 1).contiguous())

        laterals = [
            lateral_conv(out[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(out) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                scale_factor=2.0,
                mode="linear",
                align_corners=False,
            )
        
        fpn_out = self.fpn_conv(laterals[0])
        
        return fpn_out[:, :, -1]


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=32, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        # Always create norm layer - no conditional logic for TensorRT
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 1)  # B, C, L -> B, L, C
        x = self.norm(x)  # Always apply norm (either real norm or Identity)
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
        # Fixed feature dimensions - no conditional assignments for TensorRT
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features
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


class LocalLayer(nn.Module):
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
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = LocalConvAttention1D(
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


class LocalBlock(nn.Module):
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
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                LocalLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=(dilations[i] if dilations is not None else None),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    norm_layer=norm_layer,
                    act_layer=act_layer,
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
        self.feat_channel = feat_channel
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
        x : B M 3 (or B M C)
        mask: B M (torch.int32, 1=valid, 0=invalid)
        -----------------
        feature_global : B C
        """
        bs, n, c = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones((bs, n), device=device, dtype=torch.int32)
            
        # reshape so that the MLP is applied to every point
        # BatchNorm1d expects (N, C), so batch and sequence dims are merged
        x_reshaped = x.reshape(bs * n, c)
        
        # first MLP
        x_features = self.first_mlp(x_reshaped)
        x_features = x_features.reshape(bs, n, -1) # restore the original shape

        # cast the int32 mask to float and use multiplication (0=invalid, 1=valid)
        mask_float = mask.unsqueeze(-1).float()  # (B, M, 1)
        masked_x_features = x_features * mask_float
        
        # global feature by max pooling
        # invalid points are 0, so add a small value in case everything is 0
        pooled_feature = masked_x_features.max(dim=1)[0]
        
        # concatenate the global feature with the per-point features
        x_features = torch.cat(
            [x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1
        )
        
        # reshape again for the second MLP
        x_features_reshaped = x_features.reshape(bs * n, -1)
        res = self.second_mlp(x_features_reshaped)
        res = res.reshape(bs, n, self.encoder_channel)

        # mask once more (multiplication)
        masked_res = res * mask_float
        
        # final global feature
        res = masked_res.max(dim=1)[0]
        
        return res