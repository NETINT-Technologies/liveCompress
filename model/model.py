import torch.nn as nn
from compressai.models.google import ScaleHyperprior
from compressai.models.utils import conv, deconv


class liveModel(ScaleHyperprior):    
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)
        self.g_a = nn.Sequential(
            conv(4, M),
        )
        
        self.g_s = nn.Sequential(
            deconv(M, 4),
        )
    
    @property
    def downsampling_factor(self) -> int:
        return 2**4    
    
    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}
        
