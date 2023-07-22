import torch
import torch.nn.functional as F

def soft_target_cross_entropy(x, target):
    """ 软标签交叉熵损失 """
    loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
    return loss.mean()
