import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)

class ResidualBlock(nn.Module):    # 残差块
    """ Simple Residual block without any resampling
    """
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            norm_layer(dim), 
            act_layer(), 
            conv3x3(dim, dim), 
            norm_layer(dim), 
            act_layer(), 
            conv3x3(dim, dim), 
        )
    
    def forward(self, x):
        return self.conv(x) + x
    
class GPoolResidualBlock(nn.Module):   # 带全局池化偏置的残差块
    """ Residual block with global pooling bias as used in KatoGo, but a simplier version
    """
    def __init__(self, dim, c_pool, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(dim), 
            act_layer(),
            conv3x3(dim, dim),
        )

        self.c_pool = c_pool
        self.norm = norm_layer(c_pool)
        self.act = act_layer()
        self.gAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.gMaxPool = nn.AdaptiveMaxPool2d((1, 1))    # only two kinds of global pooling
        self.gpool_fc = nn.Linear(2 * c_pool, dim-c_pool)

        self.conv2 = nn.Sequential(
            norm_layer(dim), 
            act_layer(), 
            conv3x3(dim, dim),
        )
    
    def forward(self, x):
        out = self.conv1(x)
        pool_out, out = out[:, :self.c_pool], out[:, self.c_pool:]  # dim -> c_pool, dim-c_pool   # 把通道切成两半, 用一半的全局池化去偏置另一半
        gpool = self.act(self.norm(pool_out))
        gpool = torch.cat([self.gAvgPool(gpool), self.gMaxPool(gpool)], dim=1).permute(0, 2, 3, 1)  # B, 1, 1, 2*c_pool
        gpool = self.gpool_fc(gpool).permute(0, 3, 1, 2)   # B, dim-c_pool, 1, 1
        out = torch.cat([pool_out, out + gpool], dim=1)

        return self.conv2(out) + x
