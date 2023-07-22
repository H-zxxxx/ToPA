import numpy as np
import torch

from .nodes import Node
from .select_action import PUCT  # 计算UCB

from .select_action import E2W, log_sum_exp, E2WP, raw_soft_UCT, soft_indmax


class MCTS:
    def __init__(self, config):
        self.config = config
    
    def run( # 运行
        self, 
        root: Node, # 根节点改成了外部赋予
        model, 
        env, 
        add_exploration_noise = True, # 添加探索噪声
        num_simulations = None,  # 可指定搜索次数, 若不指定按config里的值
    ):
        assert root.expanded(), 'root must has been expanded.' # 根节点必须已经扩展
        
        if add_exploration_noise: # 添加狄利克雷噪声
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha, 
                exploration_fraction=self.config.root_exploration_fraction, 
            )
        max_tree_depth = 0
        
        if num_simulations == None:
            num_simulations = self.config.num_simulations

        # 开始搜索
        for _ in range(num_simulations):  # 搜索num_simulations次
            node = root
            search_path = [node]
            current_tree_depth = 0
            
            # E2W
            if self.config.select_action1 == "E2W":

                #----------选择阶段------------
                while node.expanded():
                    current_tree_depth += 1
                    action, node = self.select_child_E2W(node)
                    search_path.append(node)  # 找到叶子节点为止
                
                #--------评估和扩展阶段---------
                parent = search_path[-2]  # 叶子节点的父节点
                new_state, legal_actions, observation, done = env.step(parent.state, action) # 调用环境

                if not done: # 没到终局
                    policy, log_policy, value = self.evaluate2(observation, model)  # 评估
                    node.expand_soft(legal_actions, policy, log_policy, new_state)  # 扩展soft

                else: # 到了终局
                    if node.state == None:
                        node.state = new_state
                    value = 1 if env.getPlayer(node.state) == env.getWinner(node.state) \
                            else -1  # 不再扩展, 直接用输赢作为价值
            
                #------------回溯阶段-----------
                self.soft_backpropagate(search_path, value)
            
            # PUCT
            elif self.config.select_action1 == "PUCT":
                #----------选择阶段------------
                while node.expanded():
                    current_tree_depth += 1
                    action, node = self.select_child_PUCT(node)
                    search_path.append(node)  # 找到叶子节点为止
            
                #--------评估和扩展阶段---------
                parent = search_path[-2]  # 叶子节点的父节点
                new_state, legal_actions, observation, done = env.step(parent.state, action) # 调用环境
            
                if not done: # 没到终局
                    # policy, value = self.evaluate(observation, model) # 评估
                    # node.expand(legal_actions, policy, new_state)  # 扩展
                    policy, log_policy, value = self.evaluate2(observation, model)  # 评估
                    node.expand_soft(legal_actions, policy, log_policy, new_state)  # 扩展soft

                else: # 到了终局
                    if node.state == None:
                        node.state = new_state
                    value = 1 if env.getPlayer(node.state) == env.getWinner(node.state) \
                           else -1  # 不再扩展, 直接用输赢作为价值
            
                #------------回溯阶段-----------
                if env.getPlayer(node.state) == 1:
                    value = -value 
                self.backpropagate(search_path, value)
                self.soft_backpropagate(search_path, value)
                
                """Q_values, Q_sft_values = [], []
                for child in root.children.values():
                    Q_values.append(child.get_value())
                    Q_sft_values.append(child.value_soft)
                
                #print("Q_values:", Q_values)
                #print("Q_sft_values:", Q_sft_values)"""
            
            elif self.config.select_action1 == "softUCT":
                #----------选择阶段------------
                while node.expanded():
                    current_tree_depth += 1
                    action, node = self.select_child_PUCT(node)
                    search_path.append(node)  # 找到叶子节点为止
            
                #--------评估和扩展阶段---------
                parent = search_path[-2]  # 叶子节点的父节点
                new_state, legal_actions, observation, done = env.step(parent.state, action) # 调用环境
            
                if not done: # 没到终局
                    policy, log_policy, value = self.evaluate2(observation, model)  # 评估
                    node.expand_soft(legal_actions, policy, log_policy, new_state)  # 扩展soft

                else: # 到了终局
                    if node.state == None:
                        node.state = new_state
                    value = 1 if env.getPlayer(node.state) == env.getWinner(node.state) \
                           else -1  # 不再扩展, 直接用输赢作为价值
            
                #------------回溯阶段-----------
                self.soft_backpropagate(search_path, value)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        return root, max_tree_depth
    
    @staticmethod
    def evaluate(observation, model): # 调用模型评估状态
        observation = torch.from_numpy(observation).unsqueeze(0).to(next(model.parameters()).device)
        policy_logits, value = model(observation[:, 0:6])
        policy = torch.softmax(policy_logits, dim=-1)

        policy = policy.detach().cpu().numpy()
        value = value.detach().cpu().numpy()

        return policy[0], value[0][0]
    
    @staticmethod
    def evaluate2(observation, model): # 调用模型评估状态
        observation = torch.from_numpy(observation).unsqueeze(0).to(next(model.parameters()).device)
        policy_logits, value = model(observation[:, 0:6])
        policy = torch.softmax(policy_logits, dim=-1)
        log_policy = torch.log_softmax(policy_logits, dim=-1)

        policy = policy.detach().cpu().numpy()
        log_policy = log_policy.detach().cpu().numpy()
        value = value.detach().cpu().numpy()

        return policy[0], log_policy[0], value[0][0]
    
    def select_child_PUCT(self, node): # 选择子节点
        max_ucb = max(  # 计算最大的PUCT值
            PUCT(node, child, self.config.c_puct)
            for action, child in node.children.items()
        )
        action = np.random.choice(  # 从最大PUCT的的节点里(可能不止一个)随机选一个
            [
                action
                for action, child in node.children.items()
                if PUCT(node, child, self.config.c_puct) == max_ucb
            ]
        )
        return action, node.children[action]
    
    def select_child_softUCT(self, node):
        raw_soft_ucts = []
        actions = []
        # 遍历子节点
        for action, child in node.children.items():
            raw_soft_ucts.append(raw_soft_UCT(node, child, c_suct=1.0))
            actions.append(action)
            
        soft_ucts = soft_indmax(np.array(raw_soft_ucts))
        action = np.random.choice(actions, p=soft_ucts)

        return action, node.children[action]

    def select_child_E2W(self, node):
        weights, action_index = E2WP(node, len(self.config.action_space), self.config.temperature, self.config.exploration_param)
        action = np.random.choice(action_index, p=weights)
        return action, node.children[action]

    @staticmethod
    def backpropagate(search_path, value): # (硬)贝尔曼回溯
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value
    
    def soft_backpropagate(self, search_path, value): # 软贝尔曼回溯
        last_node = search_path[-1]
        last_node.value_soft = value
        last_node.visit_count += 1

        softs = []

        for node in reversed(search_path[:-1]):
            value_softs, _ = node.get_value_softs()
            node.value_soft = log_sum_exp(value_softs, temp=-self.config.temperature) # 对于父节点来说, 温度取负
            softs.append(node.value_soft)
            node.visit_count += 1
        
        #print("softs on search path:", softs, value)

    def get_new_root(  # 获取新的根节点
        self, 
        old_root, 
        fall_action, 
        model, 
        env, 
    ):
        done = False

        # 创建新的根节点
        if old_root == None:
            new_root = Node(0)
            root_state, legal_actions, observation = env.reset() # 获取空棋盘
            if self.config.select_action1 == "E2W":
                policy, log_policy, value = self.evaluate2(observation, model) # 评估
                new_root.expand_soft(legal_actions, policy, log_policy, root_state)  # 扩展soft
            else:
                policy, value = self.evaluate(observation, model)  # 评估
                new_root.expand(legal_actions, policy, root_state) # 扩展
        
        # 或从旧树上继承
        else:
            assert old_root.expanded(), 'old_root must has been expanded.'
            new_root = old_root.children[fall_action]

            if not new_root.expanded(): # 如果新根节点还没被扩展
                new_state, legal_actions, observation, done = env.step(old_root.state, fall_action)
                if self.config.select_action1 == "E2W":
                    policy, log_policy, value = self.evaluate2(observation, model) # 评估
                    new_root.expand_soft(legal_actions, policy, log_policy, new_state)  # 扩展soft
                else:
                    policy, value = self.evaluate(observation, model) # 评估
                    new_root.expand(legal_actions, policy, new_state) # 扩展
                
        return new_root, done
    
    def run_with_state(self, state, model, env, add_exploration_noise=False, num_simulations=None):  # 指定状态的run
        root = Node(0)
        legal_actions, observation = env.getLegalAction(state), env.encode(state)

        if self.config.select_action1 == "E2W":
            policy, log_policy, value = self.evaluate2(observation, model) # 评估
            root.expand_soft(legal_actions, policy, log_policy, state)  # 扩展soft
        else:
            policy, value = self.evaluate(observation, model)
            root.expand(legal_actions, policy, state)
        
        return self.run(root, model, env, add_exploration_noise, num_simulations)
    
    # 新增select_action, 同self-play里的
    @staticmethod
    def select_action(node, temperature): # 选择动作
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"  # 把全部子节点的访问次数n放到一个列表里, 数据类型指定为int32
        )
        actions = [action for action in node.children.keys()] # 每个子节点对应的动作放到一个列表里
        if temperature == 0:               # 如果温度参数为零, 直接选最大
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):  # 如果温度参数=无穷大, 所有动作等概率
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature) # 访问次数n的(1/温度)次方
            visit_count_distribution = visit_count_distribution / sum(   # 除以总和得到概率分布
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution) # 按此概率选动作

        return action