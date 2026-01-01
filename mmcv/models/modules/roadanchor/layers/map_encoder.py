import torch
import torch.nn as nn

from .embedding import PointsEncoder
from .fourier_embedding import FourierEmbedding

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

class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.polygon_channel = polygon_channel
        self.polygon_encoder = PointsEncoder(self.polygon_channel, dim)

    def forward(self, map_data, map_mask) -> torch.Tensor:
        bs, N, P, C = map_data.shape
        point_position = map_data # (bs, N, P=20, 2)
        
        if map_mask is not None:
            valid_mask = map_mask.unsqueeze(-1).expand(-1, -1, P)  # (bs, N, P)
            valid_mask = valid_mask.reshape(bs * N, P)  # (bs*N, P)
        else:
            valid_mask = None
            
        point_vector = torch.zeros_like(point_position[:, :, :, :2])
        point_vector[:, :, :-1, :] = point_position[:, :, 1:, :] - point_position[:, :, :-1, :]
        point_vector[:, :, -1, :] = point_vector[:, :, -2, :]
        
        point_orientation = safe_atan2(point_vector[:, :, :, 1], point_vector[:, :, :, 0])


        polygon_feature = torch.cat(
            [
                point_position,
                point_vector,
                torch.stack([point_orientation.cos(), point_orientation.sin()], dim=-1),
            ],
            dim=-1,
        ) # (bs, N, P, C=6)

        bs, N, P, C = polygon_feature.shape
        polygon_feature = polygon_feature.reshape(bs * N, P, C)        

        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, N, -1)       

        return x_polygon
