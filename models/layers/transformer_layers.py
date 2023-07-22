import torch
import torch.nn as nn

# modify from pytorch image models(timm) https://github.com/rwightman/pytorch-image-models/tree/master/timm/models

class PatchEmbed(nn.Module):   # 切块+投影
    """ 2D Board to overlapping or non-overlapping Patch Embeddings
    """
    def __init__(self, board_size=19, patch_size=4, stride=3, in_chans=10, embed_dim=256, norm_layer=None, flatten=True):
        super().__init__()
        self.num_patches = ( ( board_size - patch_size + stride ) // stride ) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.flatten = flatten
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Attention(nn.Module):    # 多头自注意力
    """ Multi-head self-attention layer
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)  # B N 3C -> B N QKV Heads HeadDim
        qkv = qkv.permute(2, 0, 3, 1, 4)  # QKV B Heads N HeadDim
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B Heads N N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B Heads N HeadDim -> B N Heads HeadDim -> B N C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):   # 多层感知机
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TransformerLayer(nn.Module):   # Transformer编码器层
    """ Transformer encoder layer
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # residual connection
        x = x + self.mlp(self.norm2(x))
        return x
