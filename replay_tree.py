# replay-tree, 原本用于替代replay buffer, 是short-term memory tree的前身, 即将弃用. 

import copy
import time

import numpy as np
import ray
import torch

import envs

from search.nodes import RT_Node
from search.select_action import piUCT # 计算πUCT

@ray.remote
class ReplayTree:  # 回放树
    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.game_action_history = {}  # 自对弈长程路径的动作字典
        # self.game_branch = {}          # 自对弈长程路径的起始字典
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        
        # Fix random generator seed
        np.random.seed(self.config.seed)
        
        self.env = envs.GoEnv(self.config) # 环境
        self.root = None
        self.init_root()  # 初始化根节点
        
        #---------新增: 用于树滑动----------
        self.new_root = None
        self.new_game_action_history = {}
        
    def get_root(self):  # 获取根节点
        return self.root
    
    def init_root(self): # 初始化根节点
        self.root = RT_Node(0)
        state, legal_acts, observation = self.env.reset()
        init_policy = np.ones_like(self.config.action_space) / len(self.config.action_space)
        self.root.expand(legal_acts, init_policy, state)
        self.root.observation = observation
    
    def reset_root(self): # 重置根节点
        init_policy = self.root.target_policy # 保留原根节点的目标策略
        self.root = RT_Node(0)
        state, legal_acts, observation = self.env.reset()
        self.root.expand(legal_acts, init_policy, state)
        self.root.observation = observation
    
    #--------------新增: 用于树滑动(tree sliding)--------------

    def reset_new_root(self, root_target_policy):
        self.new_root = RT_Node(0)
        state, legal_acts, observation = self.env.reset()
        self.new_root.expand(legal_acts, root_target_policy, state)
        self.new_root.observation = observation
    
    def get_trees(self):
        return self.root, self.game_action_history, self.new_root, self.new_game_action_history
    
    #--------------- for self-play actors ----------------

    def get_node_policy(self, action_history):  # 获取某个节点的π (通过动作历史来定位该节点)
        node = self.new_root or self.root
        
        if len(action_history) == 0:
            return node.target_policy
        
        for act in action_history:
            node = node.children[act]
            
        return node.target_policy
    
    def step_forward(self, target_policy, action_history): # 前进一步
        node = self.new_root or self.root
        assert node.expanded()
        
        if len(action_history) > 0:
            for act in action_history[:-1]:
                if act not in node.children:
                    return None, None
                node = node.children[act]
            parent = node         # 根据动作历史, 先找到该节点的父节点
            if action_history[-1] not in node.children:
                return None, None
            node = node.children[action_history[-1]]  # 再找到该节点
            new_state, legal_acts, observation, done = self.env.step(parent.state, action_history[-1])

        # 更新目标策略
        if node.expanded():
            node.update_policy(target_policy)
        
        # 如果节点还没扩展, 扩展之
        else:
            # 根据父节点的状态获得当前状态
            node.expand(legal_acts, target_policy, new_state)
            node.observation = observation

        # 选择子节点
        if np.random.choice([0, 1], p=[0.5, 0.5]) == 0:  # 50%的概率按策略选, 50%按πUCT选
            action, child = self.select_child(node)
        else:
            action, child = self.select_child_random(node)

        prior_probability = child.target_policy # 用节点上一次的目标策略作为这次的先验概率

        return action, prior_probability
    
    def update_selfplay_path(self, value, action_history, shared_storage=None): # 更新整条路径
        root = self.new_root or self.root
        root.value = value
        root.value_sum += value
        
        node = root
        for act in action_history:
            value = -value
            node = node.children[act]
            # node.value = value   # 更新子节点的价值
            # 改成用动量
            node.value = value
            node.value_sum += value
        
        if self.new_root == None:
            game_action_history = self.game_action_history
        else:
            game_action_history = self.new_game_action_history
        
        game_action_history[self.num_played_games] = action_history

        self.num_played_games += 1
        self.num_played_steps += (len(action_history) + 1)

        if self.config.replay_buffer_size / 2 < len(game_action_history):
            #del_id = self.num_played_games - len(self.game_action_history)
            #del self.game_action_history[del_id]
            # 直接重置整棵树以及game_action_history
            # self.reset_root()
            # self.game_action_history = {}
            
            # 使用树滑动
            if self.new_root == None:
                self.reset_new_root(root_target_policy=self.root.target_policy)
            else:
                new_root_init_policy = self.root.target_policy
                self.root = self.new_root
                self.reset_new_root(root_target_policy=new_root_init_policy)
                
                self.game_action_history = self.new_game_action_history
                self.new_game_action_history = {}

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)


    def select_child(self, node: RT_Node): # 选择子节点
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
    
    def select_child_random(self, node: RT_Node):  # 选择子节点(随机)
        p = node.target_policy
        action = np.random.choice(self.config.action_space, p=p) # 按此概率选动作

        return action, node.children[action]
    
    #--------------- for the trainer actor ----------------

    def get_batch(self):  # 获取batch用于神经网络训练
        observation_batch, policy_batch, value_batch = [], [], []

        sample_n_games = self.sample_n_games
        make_target = self.make_target
        
        if len(self.new_game_action_history) > 0:
            sample_probs = np.array([len(self.game_action_history), len(self.new_game_action_history)])
            sample_probs = sample_probs / sample_probs.sum()
            if np.random.choice([0, 1], p=sample_probs) == 0:
                sample_n_games = self.sample_n_games_in_new_tree
                make_target = self.make_target_in_new_tree

        for game_id, action_history in sample_n_games(self.config.batch_size // 8): # 采样batch_size局游戏
            action_pos = self.sample_position(action_history) # 每局游戏采样一个盘面
            
            # observation, policy, value = self.make_target(game_id, action_pos) # 制作标签
            # 改成强制数据增强
            for augment_index in range(8):
                observation, policy, value = make_target(game_id, action_pos, augment_index) # 制作标签

                observation_batch.append(observation) # 添加到batch
                policy_batch.append(policy)
                value_batch.append(value)

        observation_batch = np.array(observation_batch) # 转成numpy数组
        policy_batch = np.array(policy_batch)
        value_batch = np.array(value_batch)

        return observation_batch, policy_batch, value_batch

    def sample_n_games(self, n_games): # 采样n局游戏
        selected_games = np.random.choice(list(self.game_action_history.keys()), n_games)
        ret = [(game_id, self.game_action_history[game_id])
                for game_id in selected_games]
        return ret
    
    def sample_position(self, action_history): # 采样一个盘面
        position = np.random.choice(len(action_history))
        return position

    def make_target(self, game_id, action_pos, augment_index=None): # 制作标签
        node = self.root
        action_history = self.game_action_history[game_id]
        # for i in range(0, action_pos+1):
        for i in range(0, action_pos):
            node = node.children[action_history[i]]

        target_observation = node.observation
        target_policy = node.target_policy
        # target_value = -node.get_value()
        target_value = node.value
        
        if augment_index == None:
            augment_index = np.random.randint(8) # 随机选一种数据增强
            
        target_observation, target_policy, target_value = self.data_augment(
            target_observation, target_policy, target_value, augment_index, 
        )
        return target_observation, target_policy, target_value

    def data_augment(self, observation, policy, value, index):   # 数据增强x8, 旋转和翻转, index取0~7
        if index == 0:
            return observation, policy, value

        policy_pos, policy_pass = policy[:-1], policy[-1] # 策略拆成位置和停着
        policy_pos = policy_pos.reshape(self.config.board_size, -1) # reshape成二维
        
        #----------------flip----------------
        if index >= 4:
            observation = np.flip(observation, axis=-1) # 翻转最后一维, 即左右翻转
            policy_pos = np.flip(policy_pos, axis=-1)
            index -= 4
        #---------------rotate----------------
        observation = np.rot90(observation, k=index, axes=(-2, -1)) # 旋转后两维
        policy_pos = np.rot90(policy_pos, k=index, axes=(-2, -1))
        policy_pos = policy_pos.flatten() # 压回一维
        policy = np.append(policy_pos, policy_pass) # 把停着拼回来

        return observation, policy, value
    
    
    #---------------------for new tree---------------------

    def sample_n_games_in_new_tree(self, n_games): # 采样n局游戏
        selected_games = np.random.choice(list(self.new_game_action_history.keys()), n_games)
        ret = [(game_id, self.new_game_action_history[game_id])
                for game_id in selected_games]
        return ret

    def make_target_in_new_tree(self, game_id, action_pos, augment_index=None): # 制作标签
        node = self.new_root
        action_history = self.new_game_action_history[game_id]
        # for i in range(0, action_pos+1):
        for i in range(0, action_pos):
            node = node.children[action_history[i]]

        target_observation = node.observation
        target_policy = node.target_policy
        # target_value = -node.get_value()
        target_value = node.value
        
        if augment_index == None:
            augment_index = np.random.randint(8) # 随机选一种数据增强
            
        target_observation, target_policy, target_value = self.data_augment(
            target_observation, target_policy, target_value, augment_index, 
        )
        return target_observation, target_policy, target_value
    
    