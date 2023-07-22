import torch
import torch.nn as nn

from .layers import ResidualBlock, conv3x3 #, TransformerLayer, PatchEmbed
from .utils import dict_to_cpu

class AlphaZeroNetwork(nn.Module):  # 其他模块主要会调用这个类
    def __init__(self, config):
        super().__init__()
        self.network = nn.DataParallel(Network(config)) # 多卡时数据并行
    
    def forward(self, state):
        policy, value = self.network(state)
        return policy, value
    
    def get_weights(self): # 获取权值
        return dict_to_cpu(self.state_dict())
    
    def set_weights(self, weights): # 设置权值
        self.load_state_dict(weights)

class Network:  # 选择网络类型
    def __new__(cls, config):
        if config.network_type == "Conv":
            return ConvNetwork(
                config.board_size, 
                config.input_dim, 
                config.num_features, 
                config.num_blocks, 
            )
        elif config.network_type == "Transformer":
            return TransformerNetwork(
                config.board_size,
                config.input_dim, 
                config.embed_dim, 
                config.patch_size, 
                config.patch_stride, 
                config.num_heads, 
                config.depth, 
            )
        elif config.network_type == "Hybrid":  # Conv+Transformer的混合模型
            return HybridNetwork(
                config.board_size,
                config.input_dim, 
                config.num_features, 
                config.embed_dim, 
                config.patch_size, 
                config.patch_stride, 
                config.num_heads, 
                config.depths,
            )
        else:
            raise NotImplementedError(
                'network_type should be "Conv" or "Transformer" or "Hybrid".'
            )

class ConvNetwork(nn.Module):
    def __init__(self, board_size, input_dim, num_features, num_blocks, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv_1 = nn.Sequential(
            conv3x3(in_channels=input_dim, out_channels=num_features), 
            norm_layer(num_features), 
            act_layer(),
        )
        self.trunk = nn.Sequential(  # 主干
            *[ResidualBlock(num_features, act_layer=act_layer, norm_layer=norm_layer) for _ in range(num_blocks)], 
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
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.trunk(x)  # 主干

        #------policy head-------
        policy = self.policy_conv(x).flatten(1)  # 压平再进入全连接层
        policy = self.policy_fc(policy)    # 不做softmax, 直接输出logits

        #-------value head-------
        value = self.value_conv(x).flatten(1)
        value = self.value_fc(value)
        value = torch.tanh(value)   # tanh限制输出范围(-1, 1)
        
        return policy, value
"""
class TransformerNetwork(nn.Module):
    def __init__(self, board_size, input_dim, embed_dim, patch_size, stride, num_heads, depth, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        #---------patch embedding---------
        self.embed = PatchEmbed(board_size, patch_size, stride, input_dim, embed_dim)
        self.extra_tokens = nn.Parameter(torch.zeros(1, 2, embed_dim))      # 用于策略和价值的两个特殊token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed.num_patches + 2, embed_dim))  # N+2个可学习的位置编码
        #-----------transformer-----------
        self.transformer = nn.Sequential(
            *[TransformerLayer(embed_dim, num_heads, act_layer=act_layer, norm_layer=norm_layer) for _ in range(depth)]
        )
        self.norm = norm_layer(embed_dim)
        #------policy and value heads---------
        self.policy_head = nn.Linear(embed_dim, board_size**2 + 1) # 策略头
        self.value_head = nn.Sequential( # 价值头, 两层全连接
            nn.Linear(embed_dim, 64),
            act_layer(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        x = self.embed(x) # BCHW -> BNC  # 图像到序列

        extra_tokens = self.extra_tokens.expand(x.shape[0], -1, -1)  # B2C  # expand将两个特殊token复制batch_size份
        x = torch.cat((extra_tokens, x), dim=1)   # B N+2 C  # 拼接在其他正常token的最前面, 得到N+2个token
        x = self.transformer(x + self.pos_embed)  # 位置编码可以不用expand, 因为pytorch的广播机制(https://blog.csdn.net/flyingluohaipeng/article/details/125107959)
        x = self.norm(x)
        
        policy_token, value_token = x[:, 0], x[:, 1]  # 取前两个token
        policy = self.policy_head(policy_token) # 策略

        value = self.value_head(value_token) # 价值
        value = torch.tanh(value) # tanh限制输出范围(-1, 1)

        return policy, value

class HybridNetwork(nn.Module):  # 卷积和Transformer混合的网络
    def __init__(self, board_size, input_dim, num_features, embed_dim, patch_size, stride, num_heads, depths, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        conv_depth, transformer_depth = depths
        #-----------CNN backbone----------
        self.backbone = nn.Sequential(
            conv3x3(in_channels=input_dim, out_channels=num_features), 
            nn.BatchNorm2d(num_features), 
            nn.ReLU(),
            *[ResidualBlock(num_features) for _ in range(conv_depth)],
        )
        #---------patch embedding---------
        self.embed = PatchEmbed(board_size, patch_size, stride, num_features, embed_dim)
        self.extra_tokens = nn.Parameter(torch.zeros(1, 2, embed_dim))      # 用于策略和价值的两个特殊token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed.num_patches + 2, embed_dim))  # N+2个可学习的位置编码
        #-----------transformer-----------
        self.transformer = nn.Sequential(
            *[TransformerLayer(embed_dim, num_heads, act_layer=act_layer, norm_layer=norm_layer) for _ in range(transformer_depth)]
        )
        self.norm = norm_layer(embed_dim)
        #------policy and value heads---------
        self.policy_head = nn.Linear(embed_dim, board_size**2 + 1) # 策略头
        self.value_head = nn.Sequential(  # 价值头, 两层
            nn.Linear(embed_dim, 64),
            act_layer(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        x = self.backbone(x) # 多了一个卷积结构backbone
        x = self.embed(x)    # BCHW -> BNC  # 图像到序列

        extra_tokens = self.extra_tokens.expand(x.shape[0], -1, -1)  # B2C  # expand将两个特殊token复制batch_size份
        x = torch.cat((extra_tokens, x), dim=1)   # B N+2 C  # 拼接在其他正常token的最前面, 得到N+2个token
        x = self.transformer(x + self.pos_embed)  # 位置编码可以不用expand, 因为pytorch的广播机制(https://blog.csdn.net/flyingluohaipeng/article/details/125107959)
        x = self.norm(x)
        
        policy_token, value_token = x[:, 0], x[:, 1]  # 取前两个token
        policy = self.policy_head(policy_token)  # 策略

        value = self.value_head(value_token)  # 价值
        value = torch.tanh(value) # tanh限制输出范围(-1, 1)

        return policy, value
"""