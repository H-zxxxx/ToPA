import math
import time

import numpy as np
import ray
import torch

import envs, models, search
from search.select_action import E2W, E2WP

#-----------------Alphazero的自对弈--------------------

@ray.remote
class SelfPlay:
    def __init__(self, initial_checkpoint, config, seed, render=False):
        self.config = config
        self.env = envs.GoEnv(self.config)
        
        # Fix random generator
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initial the network
        self.model = models.AlphaZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])  # !改回用最新的权值
        self.model.to(self.config.device)
        self.model.eval()

        self.render = render  # 是否显示棋盘

    def continuous_self_play(self, shared_storage, replay_tree, test_mode=False): # 持续自对弈
        print("start continuous self-play")
        while ray.get(
            shared_storage.get_info.remote("training_step") # 实际的training_step小于设定值
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")     # 且还没有terminate
        ): 
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights"))) # 更新模型参数  # !改回用最新的权值
            
            self.self_play_in_replay_tree(shared_storage, replay_tree) # 自对弈一局

            # 调整训练/自对弈比率
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio_min # ratio改成区间, 小于最小值才sleep
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    # print("adjust ratio: self_play wait for 0.5s")
                    time.sleep(0.5)

    
    # 在ReplayTree中自对弈一局
    def self_play_in_replay_tree(self, shared_storage, replay_tree): 
        mcts = search.MCTS(self.config)
        action_history = []

        with torch.no_grad(): # 不追踪梯度
            start = time.time()
            # 获取初始根节点
            root, done = mcts.get_new_root(old_root=None, fall_action=None, model=self.model, env=self.env)
            init_prior_probs = ray.get(replay_tree.get_node_policy.remote(action_history=[]))
            root.update_prior(prior_probs=init_prior_probs)

            while not done: # 直到游戏结束
                # 运行树搜索
                num_legal_acts = len(self.env.getLegalAction(root.state))
                root, max_tree_depth = mcts.run(root, self.model, self.env, add_exploration_noise=True, num_simulations=self.config.num_simulations)
                if self.config.select_action1 == "E2W":
                    weights, action_index = E2WP(root, len(self.config.action_space), -self.config.temperature, exploration_param=0)
                    target_policy = np.zeros(len(self.config.action_space))
                    for action, weight in zip(action_index, weights):
                        target_policy[action] = weight
                else:
                    target_policy = self.statistics_to_policy(root, self.config.action_space)
                action, prior_probs = ray.get(
                    replay_tree.step_forward.remote(target_policy, action_history)  # 在ReplayTree上前进一步
                )
                # print("action =", action)
                if action is None:
                    return
                    
                action_history.append(action)

                if self.render:
                    if self.env.getPlayer(root.state) == self.config.black:
                        print("player: BLACK")
                    else:
                        print("player: WHITE")
                
                    print("num_root_child:", len(root.children))
                    print("max_tree_depth:", max_tree_depth)
                    if self.config.select_action1 == "E2W":
                        print("root_value:", root.value_soft)
                    else:
                        print("root_value:", root.get_value())
                    softs = []
                    for child in root.children.values():
                        softs.append(round(child.value_soft, 2))
                    print("value_softs:", softs)

                # 从旧树上继承子树的根节点
                root, done = mcts.get_new_root(old_root=root, fall_action=action, model=self.model, env=self.env)
                if prior_probs is not None:
                    root.update_prior(prior_probs)

                shared_storage.set_info.remote("real_time_played_steps")
                
                
                if self.render:
                    print("action:", action)
                    print("step:", self.env.getStep(root.state))
                
                    self.env.render(root.state) # 显示棋盘
                    print("done:", done)
                    if done:
                        print("score:", self.env.getScore(root.state))
                        print("winner:", self.env.getWinner(root.state))
                    print('')
                
            # 获取价值
            winner = self.env.getWinner(root.state)
            if winner == self.config.black:
                value = 1
            else:
                value = -1
            replay_tree.update_selfplay_path.remote(value, action_history, shared_storage) # 对局历史保存到replay_tree
            end = time.time()
            print("run time:%.4fs" % (end - start))
    
    
    def self_play_in_st_memory_tree(self, shared_storage, st_memory_tree, replay_buffer):  # 在短期记忆树中自对弈一局
        mcts = search.MCTS(self.config)
        game_history = GameHistory()
        action_history = []

        with torch.no_grad(): # 不追踪梯度
            start = time.time()
            # 获取初始根节点
            root, done = mcts.get_new_root(old_root=None, fall_action=None, model=self.model, env=self.env)
            init_prior_probs = ray.get(st_memory_tree.get_node_policy.remote(action_history=[]))
            root.update_prior(prior_probs=init_prior_probs)
            reach_leaf = False  # 是否达到叶节点的标志
            game_step = 0       # 游戏步数

            while not done: # 直到游戏结束
                # 运行树搜索
                root, max_tree_depth = mcts.run(root, self.model, self.env, add_exploration_noise=True, num_simulations=self.config.num_simulations)

                """if self.config.select_action1 == "E2W":
                    weights, action_index = E2WP(root, len(self.config.action_space), self.config.temperature, self.config.exploration_param)
                    target_policy = np.zeros(len(self.config.action_space))
                    for action, weight in zip(action_index, weights):
                        target_policy[action] = weight
                else:"""
                target_policy = self.statistics_to_policy(root, self.config.action_space)
                
                if not reach_leaf:  # 没到叶节点
                    action, prior_probs, reach_leaf = ray.get(
                        st_memory_tree.step_forward.remote(target_policy, action_history)  # 在StMemoryTree上前进一步
                    )
                    action_history.append(action)  # 添加到动作历史

                else:  # 到了叶节点
                    action = self.select_action(root, temperature=self.config.epsilon_by_frame(game_step))

                game_step += 1  # 游戏步数+1

                # print("action =", action)
                if action is None:
                    return
                
                # 添加到对局历史
                game_history.observation_history.append(self.env.encode(root.state))
                game_history.player_history.append(self.env.getPlayer(root.state))
                game_history.policy_history.append(target_policy)

                if self.render:
                    if self.env.getPlayer(root.state) == self.config.black:
                        print("player: BLACK")
                    else:
                        print("player: WHITE")

                # 从旧树上继承子树的根节点
                root, done = mcts.get_new_root(old_root=root, fall_action=action, model=self.model, env=self.env)
                if prior_probs is not None:
                    root.update_prior(prior_probs)

                shared_storage.set_info.remote("real_time_played_steps")
                
                if self.render:
                    print("num_root_child:", len(root.children))
                    print("max_tree_depth:", max_tree_depth)
                    if self.config.select_action1 == "E2W":
                        print("root_value:", root.value_soft)
                    else:
                        print("root_value:", root.get_value())
                    print("action:", action)
                    print("step:", self.env.getStep(root.state))
                    self.env.render(root.state) # 显示棋盘
                    print("done:", done)
                    if done:
                        print("score:", self.env.getScore(root.state))
                        print("winner:", self.env.getWinner(root.state))
                    print('')
                
            # 获取价值
            winner = self.env.getWinner(root.state)
            game_history.store_value_history(winner)
            
            value = 1 if self.env.getPlayer(root.state) == self.env.getWinner(root.state) \
                        else -1
            st_memory_tree.update_selfplay_path.remote(value, action_history, shared_storage) # 回溯st_memory_tree的自对弈路径
            replay_buffer.save_game.remote(game_history, shared_storage)  # 对局历史保存到replay_buffer

            end = time.time()
            print("run time:%.4fs" % (end - start))
        
    
    @staticmethod
    def statistics_to_policy(root, action_space):  # 统计数据转成策略
        sum_visits = sum(child.visit_count for child in root.children.values())
        
        return np.array([
                   root.children[a].visit_count / sum_visits
                   if a in root.children
                   else 0
                   for a in action_space
               ])
    
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
        

    @torch.no_grad()
    def policy_evaluate(self, n_games=10, num_simulations=210, shared_storage_worker=None): # 最新模型和最强模型对弈10局
        mcts = search.MCTS(self.config)

        train_model = self.model
        weights = ray.get(shared_storage_worker.get_info.remote("weights"))
        train_model.set_weights(weights)

        evaluate_model = models.AlphaZeroNetwork(self.config).to(self.config.device)
        evaluate_model.set_weights(ray.get(shared_storage_worker.get_info.remote("best_weights")))
        evaluate_model.eval()

        BLACK, WHITE = 1, 2
        color = BLACK

        win_num, lose_num = 0, 0
        model_iteration = ray.get(shared_storage_worker.get_info.remote("model_iteration"))

        for i in range(n_games):
            state, _, _ = self.env.reset()
            done = False
            if color == BLACK:
                model = {BLACK: train_model, WHITE: evaluate_model}
            else:
                model = {BLACK: evaluate_model, WHITE: train_model}
            
            while not done:
                player = self.env.getPlayer(state)
                root, max_tree_depth = mcts.run_with_state(state, model[player], self.env, False, num_simulations)
                action = self.select_action(root, temperature=0.12)
                state, _, _, done = self.env.step(state, action)

                if self.render:
                    print("player:", player, end='')
                    print(", num_root_child:", len(root.children), end='')
                    print(", max_tree_depth:", max_tree_depth, end='')
                    if self.config.select_action1 == "E2W" or "softUCT":
                        print(", root_value:", root.value_soft)
                    else:
                        print(", root_value:", root.get_value())
                    print("action:", action, end='')
                    print(", step:", self.env.getStep(state))
                    self.env.render(state) # 显示棋盘
                    print("done:", done)
                    if done:
                        print("score:", self.env.getScore(state), end='')
                        print(", winner:", self.env.getWinner(root.state))
                    print('')
            
            winner = self.env.getWinner(state)
            print("simulate round: {}".format(i+1),", winner is :", winner, ', model player is :', color)
            info1 = "simulate round: {}, winner is : {}, model player is : {}\n".format(i+1, winner, color)

            if winner == color:
                win_num += 1
            else:
                lose_num += 1
            
            print(win_num, ":", lose_num)
            color = BLACK + WHITE - color
        
        win_ratio = win_num / n_games
        
        print("model iteration: {}, win: {}, lose: {}".format(model_iteration, win_num, lose_num))
        info2 = "evaluate_score:{}, win: {}, lose: {}\n".format(model_iteration, win_num, lose_num)

        if win_ratio >= 0.7: # 胜率>=0.55, 更新最强模型
            shared_storage_worker.set_info.remote("model_iteration", model_iteration + 1)
            shared_storage_worker.set_info.remote("best_weights", weights)
        
        return win_ratio, info1, info2

class GameHistory: # 保存游戏历史
    def __init__(self):
        self.observation_history = []
        self.player_history = []
        self.policy_history = []
        self.value_history = []
        # self.action_history = []  # 新增: 动作历史

    """
    def store_policy_history(self, root, action_space): # apply at each game step
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.policy_history.append(
            np.array([
                root.children[a].visit_count / sum_visits
                if a in root.children and root.children[a].visit_count > 1  # 只访问一次的节点也剪掉
                else 0
                for a in action_space
            ])
        )
    """
    def store_value_history(self, winner): # apply at the end of the game
        self.value_history = [[1] if player == winner else [-1] \
            for player in self.player_history]
        print("winner:", winner, end='')
        print(", game length:", len(self.player_history))
    
    def __len__(self):  # 定义类的长度
        return len(self.player_history)


