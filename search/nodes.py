import numpy as np
import collections

class Node:
    """ Node of the MCTS
    """
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior  # 先验概率
        self.value_sum = 0
        self.children = {}
        self.state = None

        self.value_soft = 0
    
    def expanded(self): # 是否被扩展过
        return len(self.children) > 0
    
    def get_value(self): # 获取价值
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, legal_actions, policy, state): # 扩展
        self.state = state
        
        policy_dict = {idx: p for idx, p in enumerate(policy) if idx in legal_actions}
        for action, p in policy_dict.items():
            self.children[action] = Node(p) # 按照此p作为prior新建子节点
        
    def expand_soft(self, legal_actions, policy, log_policy, state): # 扩展for E2W
        self.expand(legal_actions, policy, state)
        
        log_policy_dict = {idx: p for idx, p in enumerate(log_policy) if idx in legal_actions}
        for action, log_p in log_policy_dict.items():
            self.children[action].value_soft = 0 # -log_policy作为value_soft的初始化
            #print("log_P:", log_p)
    
    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction): # 添加探索噪声
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac  # n为狄利克雷噪声, frac一般取0.25
    
    def update_prior(self, prior_probs): # 更新子节点的prior
        for action, p in enumerate(prior_probs):
            if action in self.children:
                prior = 0.5 * self.children[action].prior + 0.5 * p
                self.children[action].prior = prior
    
    def get_value_softs(self):
        value_softs = np.zeros(len(self.children))
        action_index = []

        for i, (action, child) in enumerate(self.children.items()):
            value_softs[i] = child.value_soft
            action_index.append(action)
        
        return value_softs, action_index

    
    def get_value_softs_and_priors(self):
        value_softs = np.zeros(len(self.children))
        priors = np.zeros(len(self.children))
        action_index = []

        for i, (action, child) in enumerate(self.children.items()):
            value_softs[i] = child.value_soft
            priors[i] = child.prior
            action_index.append(action)
        
        return value_softs, priors, action_index
    

class RT_Node(Node):  # Replay Tree Node, 继承Node
    """ Replay Tree Node
    """
    def __init__(self, pi):
        super().__init__(0)
        self.target_policy = None  # 目标策略, 用于神经网络训练
        self.pi = pi  # 增加π

        self.observation = None
        self.value = 0
    
    def update_policy(self, target_policy): # 更新目标策略及子节点的π
        self.target_policy = target_policy
        
        for action, pi in enumerate(target_policy): # 更新子节点的π
            if action in self.children:
                self.children[action].pi = pi

    # 重写expand方法
    def expand(self, legal_actions, policy, state): # 扩展
        self.state = state
        self.target_policy = policy  # 增加一个policy的赋值
        
        policy = {idx: pi for idx, pi in enumerate(policy) if idx in legal_actions}
        for action, pi in policy.items():
            self.children[action] = ST_Node(pi) # 按照此pi作为pi新建子节点
    
    # 重写add_exploration_noise方法
    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction): # 添加探索噪声
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].pi = self.children[a].pi * (1 - frac) + n * frac  # n为狄利克雷噪声, frac一般取0.25


class ST_Node(Node):  # Short-term Memory Tree Node, 继承Node
    """ Short-term Memory Tree Node
    """
    def __init__(self, pi):
        super().__init__(0)
        self.target_policy = None  # 目标策略, 用于神经网络训练
        self.pi = pi  # 增加π

        self.observation = None
        self.value = 0
    
    def update_policy(self, target_policy): # 更新目标策略及子节点的π
        self.target_policy = target_policy
        
        for action, pi in enumerate(target_policy): # 更新子节点的π
            if action in self.children:
                self.children[action].pi = pi

    # 重写expand方法
    def expand(self, legal_actions, policy, state): # 扩展
        self.state = state
        self.target_policy = policy  # 增加一个policy的赋值
        
        policy = {idx: pi for idx, pi in enumerate(policy) if idx in legal_actions}
        for action, pi in policy.items():
            self.children[action] = ST_Node(pi) # 按照此pi作为pi新建子节点
    
    # 重写add_exploration_noise方法
    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction): # 添加探索噪声
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].pi = self.children[a].pi * (1 - frac) + n * frac  # n为狄利克雷噪声, frac一般取0.25
    
    def get_value_softs_and_target_policy(self):
        value_softs = np.zeros(len(self.children))
        target_policy = np.zeros(len(self.children))
        action_index = []

        for i, (action, child) in enumerate(self.children.items()):
            value_softs[i] = child.value_soft
            target_policy[i] = target_policy[action]
            action_index.append(action)
        
        return value_softs, target_policy, action_index
