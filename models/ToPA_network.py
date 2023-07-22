import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResidualBlock, conv3x3 #, TransformerLayer, PatchEmbed
from .utils import dict_to_cpu

class ToPANetwork(nn.Module):  # 其他模块主要会调用这个类
    def __init__(self, config):
        super().__init__()
        self.network = nn.DataParallel(Network(config)) # 多卡时数据并行
    
    def forward(self, state):
        policy, value, _, _ = self.network(state)
        return policy, value
    
    def forward_with_auxiliary(self, state):
        policy, value, auxiliary_outputs, _ = self.network(state)
        return policy, value, auxiliary_outputs
    
    def forward_with_auxiliary_and_attention_map(self, state):
        policy, value, auxiliary_outputs, attn_map = self.network(state)
        return policy, value, auxiliary_outputs, attn_map
    
    def get_weights(self): # 获取权值
        return dict_to_cpu(self.state_dict())
    
    def set_weights(self, weights): # 设置权值
        self.load_state_dict(weights)

class Network:  # 选择网络类型
    def __new__(cls, config):
        if config.network_type == "Conv":
            return ConvNetwork(
                board_size=config.board_size, 
                input_dim=config.input_dim, 
                num_features=config.num_features, 
                num_blocks=config.num_blocks, 
                auxiliary_output_channels=config.auxiliary_output_channels,
            )

class ConvNetwork(nn.Module):
    def __init__(self, *, board_size, input_dim, num_features, num_blocks, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, 
                 auxiliary_output_channels):
        super().__init__()
        self.conv_1 = nn.Sequential(
            conv3x3(in_channels=input_dim, out_channels=num_features), 
            norm_layer(num_features), 
            act_layer(),
        )
        self.trunk = nn.ModuleList(  # 主干
            [ResidualBlock(num_features, act_layer=act_layer, norm_layer=norm_layer) for _ in range(num_blocks)]
        )
        self.policy_conv = nn.Sequential(  # 策略头的1x1卷积层
            nn.Conv2d(num_features, out_channels=4, kernel_size=1, stride=1),
            norm_layer(4),
            act_layer(),
        )
        self.policy_fc = nn.Linear(board_size**2 * 4, board_size**2 + 1) # 策略头的全连接

        self.value_conv = nn.Sequential(  # 价值头的1x1卷积层
            nn.Conv2d(num_features, out_channels=2, kernel_size=1, stride=1),
            norm_layer(2),
            act_layer(),
        )
        self.value_fc = nn.Sequential(   # 价值头, 两层全连接
            nn.Linear(board_size**2 * 2, 64),
            act_layer(),
            nn.Linear(64, 1),
        )
        self.auxiliary_branch = AuxiliaryBranch(  # 辅助分支, 暂时叫这个名
            num_features=num_features, 
            act_layer=act_layer, 
            norm_layer=norm_layer, 
            auxiliary_output_channels=auxiliary_output_channels, 
        )
    
    def forward(self, x):
        x = self.conv_1(x)

        for block in self.trunk[:-1]: # 前n-1个block
            x = block(x)
        
        auxiliary_outputs, x, attention_map = self.auxiliary_branch(x) # 进入辅助分支
        auxiliary_outputs = torch.sigmoid(auxiliary_outputs)
        
        x = self.trunk[-1](x) # 最后一个block

        #------policy head-------
        policy = self.policy_conv(x).flatten(1)  # 压平再进入全连接层
        policy = self.policy_fc(policy)    # 不做softmax, 直接输出logits

        #-------value head-------
        value = self.value_conv(x).flatten(1)
        value = self.value_fc(value)
        value = torch.tanh(value)   # tanh限制输出范围(-1, 1)
        
        return policy, value, auxiliary_outputs, attention_map
    
# 耿贝尔softmax, 重参数化技巧使离散随机采样可微
def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits) #, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class AuxiliaryBranch(nn.Module):  # 辅助分支, 暂时叫这个名
    def __init__(self, num_features, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, auxiliary_output_channels=10, 
                 attention_drop=0.1):
        super().__init__()
        self.conv_1 = nn.Sequential(
            conv3x3(num_features, 16 * auxiliary_output_channels),  # 给每个auxiliary_output 16个通道
            norm_layer(16 * auxiliary_output_channels),
            act_layer(),
        )

        self.group_conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16 * auxiliary_output_channels, 
                out_channels=16 * auxiliary_output_channels, 
                kernel_size=3,
                padding=1,
                groups=auxiliary_output_channels,  # 使用分组卷积, 使每个auxiliary_output分开计算
            ),
            norm_layer(16 * auxiliary_output_channels),
            act_layer(),
        )

        self.group_conv_2 = nn.Conv2d(   # 1x1卷积输出auxiliary_output_channels个平面, 依然使用分组卷积
            in_channels=16 * auxiliary_output_channels, 
            out_channels=auxiliary_output_channels, 
            kernel_size=1, 
            groups=auxiliary_output_channels,
        )
        self.norm = norm_layer(num_features)
        self.act = act_layer()

        self.attention_fc = nn.Linear(num_features, 16 * auxiliary_output_channels)
        self.attn_drop = nn.Dropout(attention_drop)  # Attention Dropout
        
        self.reduce_conv_attn = nn.Conv2d(  # 降低通道数
            in_channels=16 * auxiliary_output_channels, 
            out_channels=num_features,
            kernel_size=1, 
        )

    def forward(self, x):
        B = x.shape[0]
        auxiliary_branch = self.conv_1(x)
        auxiliary_outputs = torch.sigmoid(self.group_conv_2(self.group_conv_1(auxiliary_branch)))  # 辅助输出

        global_pooled = F.adaptive_avg_pool2d(self.act(self.norm(x)), (1, 1)).flatten(1)  # 全局池化

        attention = self.attn_drop(self.attention_fc(global_pooled))        # 计算注意力

        # hard gumbel-softmax
        attention = gumbel_softmax(attention.reshape(B, -1, 16), tau=1, hard=True, dim=1)  # [B, aux_out_channels, 16]
        attention = attention.sum(-1)

        # 扩展到对应维度
        attention_expanded = attention.unsqueeze(-1).repeat(1, 1, 16).reshape(B, -1, 1, 1)  # [B, aux_out_channels, 1] -> [B, aux_out_channels, 16] -> [B, aux_out_channels*16, 1, 1]

        attentioned_auxiliary_branch = auxiliary_branch * attention_expanded  # element-wise product, 维度=1的部分会自动填充, 相当于channel-wise product
        
        out = self.reduce_conv_attn(attentioned_auxiliary_branch)

        out += x  # residual connection
        
        return auxiliary_outputs, out, attention

