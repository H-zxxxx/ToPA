import math
from .nodes import Node, RT_Node
import numpy as np

# 计算PUCT
def PUCT(parent: Node, child: Node, c_puct=1.0):
    #Q = -child.get_value()  # 子节点的价值对于自己来说是负的
    Q = child.value_soft
    U = c_puct * child.prior * 10
    U *= (math.sqrt(parent.visit_count) / (child.visit_count + 1))
    return Q + U

# 计算πUCT
def piUCT(parent: RT_Node, child: RT_Node, c_piuct=1.0, anti_exploit=True):
    Q = -child.get_value()  # 子节点的价值对于自己来说是负的
    U = c_piuct * child.pi
    U *= (math.sqrt(parent.visit_count) / (child.visit_count + 1))
    if anti_exploit:
        return -Q + U
    return Q + U


#----------------Maximum Entropy Policies----------------

# f_tau
def soft_indmax(x, temp=1.0):
    """ 在深度学习里一般把这个称为softmax, 这里沿用MaxEnt RL里的叫法soft-indmax
    """
    x = x / temp
    e_x = np.exp(x - np.max(x))  # 减最大值是为了防止溢出, 不会影响计算结果
    return e_x / np.sum(e_x)

# F_tau
def log_sum_exp(x, temp=1.0):
    """ MaxEnt RL里定义的softmax, 为了不引起歧义, 这里命名为log_sum_exp
    """
    x = x / temp
    max_x = np.max(x)
    log_sum_exp_x = np.log(np.sum(np.exp(x - max_x))) + max_x  # 同上, 加减最大值是为了防止溢出, 不会影响计算结果
    
    return temp * log_sum_exp_x

# 试一个新的
def raw_soft_UCT(parent: Node, child: Node, c_suct=1.0):
    Q_soft = -child.value_soft
    U = c_suct * child.prior
    U *= (math.sqrt(parent.visit_count) / (child.visit_count + 1))
    return Q_soft + U

# E2W in MENTS
def E2W(parent: Node, temperature=1.0, exploration_param=0.1):
    """ 经验指数权重, Empircal Exponential Weight
        不同于UCT是确定性策略, E2W是随机策略, 返回的是选择各个动作权重(概率)
    """
    num_legal_actions = len(parent.children)
    parent_visit_count = parent.visit_count
    value_softs, action_index = parent.get_value_softs()
    explore_term = exploration_param / np.sqrt(parent_visit_count + 1)
    
    f_tau_Q_soft = soft_indmax(value_softs, temp=-temperature) # 子节点价值取负
    exploit_term = (1 - explore_term * num_legal_actions) * f_tau_Q_soft  # 最后一个*号是element-wise product
    
    weights = explore_term + exploit_term

    # 归一化
    weights = min_max_normalize(weights)
    weights = weights / weights.sum()

    return weights, action_index

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val == min_val:
        return x
    normalized_x = (x - min_val) / (max_val - min_val)
    return normalized_x

# Proposed E2WP
def E2WP(parent: Node, len_action_space, temperature=1.0, exploration_param=1.0):
    """ 带先验概率的经验指数权重, Empircal Exponential Weight with Prior Probability
        不同于UCT是确定性策略, E2WP是随机策略, 返回的是选择各个动作权重(概率)
    """
    num_legal_actions = len(parent.children)
    parent_visit_count = parent.visit_count
    value_softs, priors, action_index = parent.get_value_softs_and_priors()

    decay_rate = exploration_param * num_legal_actions / (len_action_space * (np.sqrt(parent_visit_count + 1)))
    explore_term = decay_rate * priors
    
    f_tau_Q_soft = soft_indmax(value_softs, temp=-temperature) # 子节点价值取负
    exploit_term = (1 - decay_rate) * f_tau_Q_soft  # 最后一个*号是element-wise product
    
    weights = explore_term + exploit_term

    # 归一化
    # weights = min_max_normalize(weights)
    weights = weights / weights.sum()  # 概率和不严格=1会报错

    return weights, action_index

# 反利用E2WP
def AE2WP(parent: Node, len_action_space, temperature=1.0, exploration_param=0.1, anti_exploit=True):
    """ 带先验概率的经验指数权重, Empircal Exponential Weight with Prior Probability
        不同于UCT是确定性策略, E2WP是随机策略, 返回的是选择各个动作权重(概率)
    """
    num_legal_actions = len(parent.children)
    parent_visit_count = parent.visit_count
    value_softs, priors, action_index = parent.get_value_softs_and_target_policy()

    decay_rate = exploration_param * num_legal_actions / (len_action_space * ((parent_visit_count + 1) ** 0.5))
    explore_term = decay_rate * priors

    if not anti_exploit:
        value_softs = - value_softs

    f_tau_Q_soft = soft_indmax(value_softs, temp=temperature) # 子节点价值取负, anti_exploit的话再取负, 负负得正相当于不取了

    exploit_term = (1 - decay_rate) * f_tau_Q_soft  # 最后一个*号是element-wise product
    
    weights = explore_term + exploit_term

    # 归一化
    # weights = min_max_normalize(weights)
    weights = weights / weights.sum()

    return weights, action_index
