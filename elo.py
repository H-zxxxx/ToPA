import random
import numpy as np
import os
import pickle
from scipy.optimize import minimize
import enum
""""
from GoEnv.environment import GoEnv
from self_play import MCTS
from configure import Config
from model import AlphaZeroNetwork, AlphaZeroNetwork2
"""

from envs import GoEnv
from search import MCTS
from configs import ToPAConfig as Config
from models import AlphaZeroNetwork

import argparse
parser = argparse.ArgumentParser(description="ToPA")
parser.add_argument('--gpu', type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

def simulate_game(black_model, white_model, config, go_env, render=True):
    game_state, _, _ = go_env.reset()
    done = False
    mcts = MCTS(config)

    BLACK = 1
    WHITE = 2
    models = {
        BLACK: black_model,
        WHITE: white_model,
    }
    
    while not done:
        next_player = go_env.getPlayer(game_state)
        #next_action = agents[next_player].select_action(game_state)
        root, max_tree_depth = mcts.run_with_state(game_state, models[next_player], go_env, False, config.num_simulations)
        next_action = mcts.select_action(root, temperature=0.12)
        game_state, _, _ , done = go_env.step(game_state, next_action)

        if render:
            print("player:", next_player, end='')
            print(", num_root_child:", len(root.children), end='')
            print(", max_tree_depth:", max_tree_depth, end='')
            if config.select_action1 == "E2W":
                print(", root_value:", root.value_soft)
            else:
                print(", root_value:", root.get_value())
            print("action:", next_action, end='')
            print(", step:", go_env.getStep(game_state))
            go_env.render(game_state) # 显示棋盘
            print("done:", done)
            if done:
                print("score:", go_env.getScore(game_state), end='')
                print(", winner:", go_env.getWinner(root.state))
            print('')

    winner = go_env.getWinner(game_state)
    return winner

   
def nll_results(ratings, winners, losers):
    all_ratings = np.concatenate([np.ones(1), ratings])
    winner_ratings = all_ratings[winners]
    loser_ratings = all_ratings[losers]
    log_p_wins = np.log(winner_ratings / (winner_ratings + loser_ratings))
    log_likelihood = np.sum(log_p_wins)
    return -1 * log_likelihood


def calculate_ratings(models, num_games,config, go_env):
    num_models = len(models)
    model_ids = list(range(num_models))

    winners = np.zeros(num_games, dtype=np.int32)
    losers = np.zeros(num_games, dtype=np.int32)

    for i in range(num_games):
        print("Game %d / %d..." % (i + 1, num_games))
        black_id, white_id = random.sample(model_ids, 2)
        print("black id: ", black_id, ", white id: ", white_id)

        with open(os.path.join(config.test_results_path, "test_record.txt"), "a") as f:
            f.write("black id: " + str(black_id) + ", white id: " + str(white_id))

        winner = simulate_game(models[black_id], models[white_id],config, go_env)

        with open(os.path.join(config.test_results_path, "test_record.txt"), "a") as f:
            f.write("\nwinner: " + str(winner))
            f.write("\n\n")
        
        if winner == 1:
            winners[i] = black_id
            losers[i] = white_id
        else:
            winners[i] = white_id
            losers[i] = black_id

    guess = np.ones(num_models - 1)          # 第0个model作为elo的 base. 且 elo = 0
    bounds = [(1e-8, None) for _ in guess]   # None is used to specify no bound.
    result = minimize(
        nll_results, guess,
        args=(winners, losers),
        bounds=bounds)
    assert result.success          
                                    
    abstract_ratings = np.concatenate([np.ones(1), result.x])
    elo_ratings = 400.0 * np.log10(abstract_ratings)
    # min_rating = np.min(elo_ratings)

    return elo_ratings


def fab_models(model_paths, config):
    models = []
    agents = []
    num_agents = len(model_paths)
    print("\nnum_agents:\n",num_agents)
    for i in range(num_agents):
        model = AlphaZeroNetwork(config).to(config.device)
        models.append(model)
    
    for i, path in enumerate(model_paths):
        if os.path.exists(path):
            print("agent{}_path is exists.\n".format(i))
            with open(path, "rb") as f:
                    model_weights = pickle.load(f)
                    models[i].set_weights( model_weights["weights"])
    """
    for i in range(num_agents):
        agent = MCTS(config, go_env, models[i])
        agents.append(agent)
    """
    return models
 
if __name__ == '__main__':
  
    config = Config()
    config.num_simulation = 210
    go_env = GoEnv(config)

    os.makedirs(config.test_results_path, exist_ok=True)

    num_games = 1000
    """
    model_paths = [ "./save_weight/AZ_10_19_best_policy_1.model",
                    "./save_weight/ToPA_10_16_best_policy_11.model", "./save_weight/ToPA_10_18_best_policy_11.model", 
                    "./save_weight/AZ_10_19_best_policy_11.model", "./save_weight/AZ_10_19_best_policy_16.model",
                    "./save_weight/AZ_10_19_best_policy_22.model", "./save_weight/AZ_10_19_best_policy_24.model",
                    "./save_weight/AZ_10_19_best_policy_34.model", 
                    "./save_weight/AZ_10_22_best_policy_11.model", "./save_weight/AZ_10_22_best_policy_17.model", 
                    "./save_weight/ToPA_10_25_best_policy_22.model", "./save_weight/ToPA_10_25_best_policy_31.model", 
                    "./save_weight/ToPA_10_25_best_policy_34.model",
    ]
    
    model_paths = ["results/2023-03-21--17-32-28/best_policy_1.model", 
                   "results/2023-03-21--17-32-28/best_policy_3.model", 
                   "results/2023-03-21--17-32-28/best_policy_5.model", 
                   "results/2023-03-21--17-32-28/best_policy_7.model", 
                   "results/2023-03-21--17-32-28/best_policy_9.model", 
                   "results/2023-03-21--17-32-28/best_policy_11.model", 
                   "results/2023-03-21--17-32-28/best_policy_12.model", 
                   "results/2023-03-21--17-32-28/best_policy_13.model", 
                   "results/2023-03-21--17-32-28/best_policy_14.model", 
                   "results/2023-03-21--17-32-28/best_policy_15.model", 
                   "results/2023-03-21--17-32-28/best_policy_16.model"]
    
    model_paths = ["results/2023-03-31--18-02-33/best_policy_1.model", 
                   "results/2023-03-31--18-02-33/best_policy_10.model",
                   "results/2023-03-31--18-02-33/best_policy_15.model",
                   "results/2023-03-31--18-02-33/best_policy_20.model", 
                   "results/2023-03-31--18-02-33/best_policy_24.model"]
    
    model_paths = ["results/2023-04-06--21-36-26/best_policy_1.model", 
                   "results/2023-04-06--21-36-26/best_policy_2.model",
                   "results/2023-04-06--21-36-26/best_policy_3.model",
                   "results/2023-04-06--21-36-26/best_policy_4.model",
                   "results/2023-04-06--21-36-26/best_policy_5.model", 
                   "results/2023-04-06--21-36-26/best_policy_9.model", 
                   "results/2023-04-06--21-36-26/best_policy_13.model"]
    """
    #model_paths = ["results/2023-05-04--09-59-22/best_policy_" + str(i+1) + ".model" for i in range(39)]
    #model_paths = ["results/2023-05-21--13-18-16/best_policy_" + str(i+1) + ".model" for i in range(19)]
    #model_paths = ["results/2023-06-09--11-16-15/best_policy_" + str(i+1) + ".model" for i in range(10)]
    model_paths = ["results/2023-06-20--01-51-02/best_policy_" + str(i+1) + ".model" for i in range(9)]

    models = fab_models(model_paths, config)

    with open(os.path.join(config.test_results_path, "test_record.txt"), "a") as f:
        f.write("model_paths: \n")
        f.write(str(model_paths))
        f.write("\n\n")

    cal_ratings = calculate_ratings(models, num_games, config, go_env)
    print("per agent elo is :", cal_ratings)
    # 写入测试记录
    with open(os.path.join(config.test_results_path, "test_record.txt"), "a") as f:
        f.write("per agent elo is: \n")
        f.write(str(cal_ratings))
        f.write("\n\n")
    
    f.close()