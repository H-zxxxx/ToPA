import copy
import time

import numpy as np
import ray
import torch

import envs

from search.nodes import ST_Node
from search.select_action import piUCT, log_sum_exp, AE2WP # 计算πUCT, F_tau

@ray.remote
class ShorttermMemoryTree:  # 短期记忆树
    def __init__(self, initial_checkpoint, initial_root, config):
        self.config = config
        self.game_action_history = {}  # 自对弈长程路径的动作字典
        self.num_played_games = 0
        #self.num_played_steps = 0
        
        # Fix random generator seed
        np.random.seed(self.config.seed)
        
        self.env = envs.GoEnv(self.config) # 环境
        self.root = None
        self.init_root()  # 初始化根节点
        
        
    def get_root(self):  # 获取根节点
        return self.root
    
    def init_root(self): # 初始化根节点
        self.root = ST_Node(0)
        state, legal_acts, observation = self.env.reset()
        init_policy = np.ones_like(self.config.action_space) / len(self.config.action_space)
        self.root.expand(legal_acts, init_policy, state)
        self.root.observation = observation
    
    def reset_root(self): # 重置根节点
        init_policy = self.root.target_policy # 保留原根节点的目标策略
        self.root = ST_Node(0)
        state, legal_acts, observation = self.env.reset()
        self.root.expand(legal_acts, init_policy, state)
        self.root.observation = observation
    

    #--------------- for self-play actors ----------------

    def get_node_policy(self, action_history):  # 获取某个节点的π (通过动作历史来定位该节点)
        node = self.root
        
        if len(action_history) == 0:
            return node.target_policy
        
        for act in action_history:
            node = node.children[act]
            
        return node.target_policy
    
    def step_forward(self, target_policy, action_history): # 前进一步
        reach_leaf = False  # 是否到达叶节点
        node = self.root
        assert node.expanded()
        
        if len(action_history) > 0:
            for act in action_history[:-1]:
                try:
                    node = node.children[act]  # 处理异常, 树满时可能会出现
                except KeyError:
                    return None, None
                
            parent = node    # 根据动作历史, 先找到该节点的父节点

            try:  # 处理异常
                node = node.children[action_history[-1]]  # 再找到该节点
            except KeyError:
                return None, None
            
            # 通过父节点获取子节点状态
            new_state, legal_acts, observation, done = self.env.step(parent.state, action_history[-1])

        # 更新目标策略
        if node.expanded():
            node.update_policy(target_policy)
        
        # 如果节点还没扩展, 扩展之
        else:
            # 根据父节点的状态获得当前状态
            node.expand(legal_acts, target_policy, new_state)
            node.observation = observation
            reach_leaf = True  # 到达叶节点了

        # 选择子节点
        if self.config.select_action2 == "E2W":
            action, child = self.select_child_AE2WP(node)
        else:
            action, child = self.select_child_AeUCT(node)

        prior_probability = child.target_policy # 用节点上一次的目标策略作为这次的先验概率

        return action, prior_probability, reach_leaf
    
    def select_child_AeUCT(self, node):
        if np.random.choice([0, 1], p=[0.5, 0.5]) == 0:  # 50%的概率按策略选, 50%按πUCT选
            action, child = self.select_child(node)
        else:
            action, child = self.select_child_random(node)
        return action, child
    
    def select_child_AE2WP(self, node):    # AE2WP本来就是随机策略, 不用在引入随机选择了
        weights, action_index = AE2WP(node, len(self.config.action_space), self.config.temperature, self.config.exploration_param, self.anti_exploit)
        action = np.random.choice(action_index, p=weights)
        return action, node.children[action]
    
    def update_selfplay_path(self, value, action_history, shared_storage=None): # 更新整条路径
        root = self.root
        root.value_sum += value
        node = root
        self_play_path = [node]  # 通过动作历史找到自对弈路径

        for act in action_history:
            node = node.children[act]
            self_play_path.append(node)

        if self.config.select_action2 == "E2W":
            self.soft_backpropagate(search_path=self_play_path, value=value)  # 软回溯
        else:
            self.backpropagate(search_path=self_play_path, value=value)   # 原本的回溯
                
        self.game_action_history[self.num_played_games] = action_history

        self.num_played_games += 1
        self.num_played_steps += (len(action_history) + 1)

        if len(self.game_action_history) > self.config.st_memory_tree_size:  # 超过了stm tree的容量, 清除掉整棵树
            self.reset_root()
            self.game_action_history = {}

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)
    
    @staticmethod
    def backpropagate(search_path, value): # (硬)贝尔曼回溯
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value
    
    def soft_backpropagate(self, search_path, value): # 软贝尔曼回溯
        last_node = search_path[-1]
        #print(value)
        value = (value + 1) * 2.0
        last_node.value_soft = value
        last_node.visit_count += 1
        for node in reversed(search_path[:-1]):
            value_softs, _ = node.get_value_softs()
            node.value_soft = log_sum_exp(-value_softs, temp=self.config.temperature) # 对于父节点来说, 价值取负
            #print(node.value_soft)
            node.visit_count += 1

    def select_child(self, node: ST_Node): # 选择子节点
        max_ucb = max(  # 计算最大的πUCT值
            piUCT(node, child, self.config.c_puct, self.config.anti_exploit)
            for action, child in node.children.items()
        )
        action = np.random.choice(  # 从最大πUCT的的节点里(可能不止一个)随机选一个
            [
                action
                for action, child in node.children.items()
                if piUCT(node, child, self.config.c_piUCT, self.config.anti_exploit) == max_ucb
            ]
        )
        return action, node.children[action]
    
    def select_child_random(self, node: ST_Node):  # 选择子节点(随机)
        p = node.target_policy
        action = np.random.choice(self.config.action_space, p=p) # 按此概率选动作

        return action, node.children[action]