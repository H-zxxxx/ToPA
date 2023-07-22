import copy
import math
import os
import pickle
import sys
import time

import numpy as np
import ray
import torch

#from torch.utils.tensorboard import SummaryWriter

import models
import replay_buffer
import replay_tree
import self_play
import shared_storage
import trainer
import configs

import argparse
parser = argparse.ArgumentParser(description="ToPA")
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--num_simulations', type=int, default=210)
parser.add_argument('--evaluate_num', type=int, default=1500)
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

class ToPA:
    def __init__(self, config):
        self.config = config

        # Fix random generator seed   # 设置随机数生成器的种子
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        self.num_gpus = torch.cuda.device_count()
        print("num_gpus:", self.num_gpus)
        ray.init(num_gpus=self.num_gpus, ignore_reinit_error=True)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {  # checkpoint是一个字典
            "weights": None,             # 最新的权重
            "best_weights": None,        # 最强的权重
            "model_iteration": 0,        # 第几代模型
            "optimizer_state": None, 
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,       # 下完一局才更新
            "real_time_played_steps": 0, # 实时的步数, 每下一步都更新一次
            "terminate": False,
        }
        self.replay_tree = {}
        
        if self.config.init_model:
            if os.path.exists( self.config.init_model_path):
                with open( self.config.init_model_path, "rb") as f:
                    model_weights = pickle.load(f)
                    self.checkpoint["weights"] = model_weights["weights"]
                    self.checkpoint["best_weights"] = model_weights["best_weights"]
                    print(" load model success...\n")
        else:
            cpu_actor = CPUActor.remote()
            cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
            self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))
            self.checkpoint["best_weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_play_workers = None
        self.evaluate_worker = None
        self.training_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        

    def train(self, log_in_tensorboard=True):
        if log_in_tensorboard or self.config.save_model:
            os.makedirs(self.config.results_path, exist_ok=True)
        
        # Manage GPUs
        num_gpus_per_worker = self.num_gpus / (self.config.num_workers + 2)
        if 1 < num_gpus_per_worker:
            num_gpus_per_worker = math.floor(num_gpus_per_worker)
        print("num_gpus_per_worker:", num_gpus_per_worker)
        
        # Initialize workers
        self.training_worker = trainer.AlphaZeroTrainer.options(
            num_gpus=num_gpus_per_worker, 
        ).remote(self.checkpoint, self.config)
        
        self.shared_storage_worker = shared_storage.SharedStorage.remote(
            self.checkpoint, self.config, 
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = replay_tree.ReplayTree.remote(
            self.checkpoint, self.replay_tree, self.config, 
        )

        self.self_play_workers = [
            self_play.SelfPlay.options(
                num_cpus=0, 
                num_gpus=num_gpus_per_worker, 
            ).remote(
                self.checkpoint, self.config, self.config.seed + seed, # +seed使每个自对弈线程随机数生成器不一样
                render = False 
                #render = True if seed == 0 else False, 
            )
            for seed in range(self.config.num_workers)
        ]

        self.evaluate_worker = self_play.SelfPlay.options(
            num_cpus=0, 
            num_gpus=num_gpus_per_worker, 
        ).remote(
            self.checkpoint, self.config, self.config.seed + self.config.num_workers, 
            True, 
        )
        
        # launch workers
        [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_play_worker in self.self_play_workers
        ]
        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        
        if log_in_tensorboard:
            self.logging_loop(num_gpus_per_worker)
        
        
    def logging_loop(self, num_gpus):
        counter = 0
        keys = [
            "training_step", 
            "num_played_games", 
            "num_played_steps", 
            "real_time_played_steps", 
            "total_loss", 
            "policy_loss", 
            "value_loss", 
            "lr", 
            "model_iteration", 
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                if counter % 10 == 0:
                    print("counter:", counter)
                    print(info, "\n")
                
                    # 写入训练记录
                    with open(os.path.join(self.config.results_path, "train_record.txt"), "a") as f:
                        f.write(("counter:{}\n").format(counter))
                        f.write(str(info) + "\n")
                
                # 将损失 用tensorboard画出来(暂未实现)
                # writer.add_scalars(...)

                if (counter + 1) % self.config.evaluate_num == 0:
                    win_ratio, _, info2 = ray.get(self.evaluate_worker.policy_evaluate.remote(shared_storage_worker=self.shared_storage_worker))  # 启动评估的线程
                    with open(os.path.join(self.config.results_path, "train_record.txt"), "a") as f:
                        f.write(info2)

                    pickle.dump(
                        ray.get(self.shared_storage_worker.get_info.remote(["weights", "best_weights"])),
                        open(os.path.join(self.config.results_path, "current_policy_" + str(info["num_played_steps"]) + ".model"), "wb")
                    )

                    if win_ratio >= 0.7:
                        print("New best policy!!!!!!!!")
                        
                        model_iteration = ray.get(self.shared_storage_worker.get_info.remote("model_iteration"))
                        pickle.dump(
                            ray.get(self.shared_storage_worker.get_info.remote(["weights","best_weights"])),
                            open(os.path.join(self.config.results_path, "best_policy_{}.model".format(model_iteration)), "wb")
                        )

                counter += 1
                time.sleep(1)

        except KeyboardInterrupt:
            pass
        
        self.terminate_workers()

        if self.config.save_model:
            # Persist replay buffer to disk
            print("\n\nPersisting replay buffer games to disk...")
            pickle.dump(
                {
                    "root": self.root,
                    "game_action_history": self.game_action_history,
                    "new_root": self.new_root,
                    "new_game_action_history": self.new_game_action_history,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                },
                open(os.path.join(self.config.results_path, "replay_tree.pkl"), "wb"),
            )

    def terminate_workers(self):
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            (   self.root, 
                self.game_action_history,
                self.new_root, 
                self.new_game_action_history
            ) = ray.get(self.replay_buffer_worker.get_trees.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
    
    def test(self):
        pass
    
    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        # Load checkpoint
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self.checkpoint = torch.load(checkpoint_path)
                print(f"\nUsing checkpoint from {checkpoint_path}")
            else:
                print(f"\nThere is no model saved in {checkpoint_path}.")

        # Load replay buffer
        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                with open(replay_buffer_path, "rb") as f:
                    replay_buffer_infos = pickle.load(f)
                self.replay_tree = replay_buffer_infos["buffer"]
                self.checkpoint["num_played_steps"] = replay_buffer_infos[
                    "num_played_steps"
                ]
                self.checkpoint["num_played_games"] = replay_buffer_infos[
                    "num_played_games"
                ]
                print(f"\nInitializing replay buffer with {replay_buffer_path}")

            else:
                print(
                    f"Warning: Replay buffer path '{replay_buffer_path}' doesn't exist.  Using empty buffer."
                )
                self.checkpoint["training_step"] = 0
                self.checkpoint["num_played_steps"] = 0
                self.checkpoint["num_played_games"] = 0


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.AlphaZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary

if __name__ == '__main__':
    config = configs.ToPAConfig()
    config.num_simulations = opt.num_simulations
    config.evaluate_num = opt.evaluate_num
    agent = ToPA(config)
    #agent.load_model("results/2023-02-13--19-09-25/model.checkpoint", "results/2023-02-13--19-09-25/replay_tree.pkl")
    agent.train()

