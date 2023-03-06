import copy
import time

import numpy as np
import ray

import torch
import torch.nn.functional as F

import models


class Trainer:
    def __init__(self, config):
        self.config = config

        # Fix random generator seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
                
    def continuous_update_weights(self, replay_buffer, shared_storage): # 持续更新网络权值
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1: # 等一局对弈完才开始训练
            time.sleep(0.1)
        print("now update weights")
        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            #total_loss, policy_loss, value_loss = self.update_weights(batch)
            total_loss, policy_loss, value_loss, aux_loss = self.update_weights_with_auxiliary(batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "policy_loss": policy_loss,
                }
            )
            # Managing the self-play / training ratio  # 调整自对弈/训练比率
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio_max # ratio改成区间, 大于最大值才sleep
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    # print("adjust ratio: training wait for 0.5s")
                    time.sleep(0.5)

    def update_weights(self, batch): # 更新一次网络权值
        observation_batch, target_policy, target_value = batch

        device = next(self.model.parameters()).device
        observation_batch = torch.from_numpy(observation_batch).float().to(device)
        target_policy = torch.from_numpy(target_policy).float().to(device)
        target_value = torch.from_numpy(target_value).float().to(device)

        # Predict  # 模型预测
        policy_logits, value = self.model(observation_batch)
        # Compute loss  # 计算损失
        policy_loss, value_loss = self.loss_function(policy_logits, value.view(-1), target_policy, target_value) # 计算策略、价值损失
        loss = policy_loss + value_loss * self.config.value_loss_weight  # 总损失, 可以调节价值损失的占比
        # Optimize  # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return loss.item(), policy_loss.item(), value_loss.item()
    
    def update_weights_with_auxiliary(self, batch): # 更新一次网络权值
        observation_batch, target_policy, target_value = batch

        device = next(self.model.parameters()).device
        auxiliary_targets = torch.from_numpy(observation_batch[:, 6:]).float().to(device)
        observation_batch = torch.from_numpy(observation_batch[:, 0:6]).float().to(device)
        target_policy = torch.from_numpy(target_policy).float().to(device)
        target_value = torch.from_numpy(target_value).float().to(device)

        # Predict  # 模型预测
        policy_logits, value, auxiliary_outputs = self.model.forward_with_auxiliary(observation_batch)
        # Compute loss  # 计算损失
        policy_loss, value_loss, aux_loss = self.loss_function_with_auxiliary(
            policy_logits, 
            value.view(-1), 
            auxiliary_outputs,
            target_policy, 
            target_value, 
            auxiliary_targets, 
        ) # 计算策略、价值损失
        loss = policy_loss + value_loss * self.config.value_loss_weight + aux_loss * 0.9 # 总损失, 可以调节价值损失的占比
               
        # Optimize  # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return loss.item(), policy_loss.item(), value_loss.item(), aux_loss.item()
    
    def update_lr(self): # 更新学习率
        if self.optimizer.param_groups[0]["lr"] == self.config.lr_final:
            return  # 达到最终学习率了, 直接返回

        lr = self.config.lr_init * self.config.lr_decay_rate ** ( # 计算当前学习率
            self.training_step / self.config.lr_decay_steps
        )
        if lr < self.config.lr_final: # 达到最终学习率后不再衰减
            lr = self.config.lr_final
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(policy_logits, value, target_policy, target_value): # 损失函数
        policy_loss = models.loss.soft_target_cross_entropy(policy_logits, target_policy) # 软标签交叉熵损失
        value_loss = F.mse_loss(value, target_value) # 均方损失
        return policy_loss, value_loss
    
    @staticmethod
    def loss_function_with_auxiliary(
        policy_logits, 
        value, 
        auxiliary_outputs,
        target_policy, 
        target_value,
        auxiliary_targets,
    ): # 损失函数
        policy_loss = models.loss.soft_target_cross_entropy(policy_logits, target_policy) # 软标签交叉熵损失
        value_loss = F.mse_loss(value, target_value) # 均方损失
        auxiliary_loss = F.binary_cross_entropy(auxiliary_outputs, auxiliary_targets)
        return policy_loss, value_loss, auxiliary_loss


@ray.remote
class AlphaZeroTrainer(Trainer):
    def __init__(self, initial_checkpoint, config):
        super().__init__(config)
        
        # Initialize the network
        self.model = models.AlphaZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(self.config.device)
        self.model.train()
        
        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.l2_const,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.l2_const,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )
        
        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )


@ray.remote
class TruncateGoTrainer(Trainer):
    def __init__(self, initial_checkpoint, config):
        super().__init__(config)
        
        # Initialize the network
        self.model = models.TruncateNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(self.config.device)
        self.model.train()
        
        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.l2_const,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.l2_const,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )
        
        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )
    
    def continuous_update_weights(self, replay_buffer, shared_storage): # 持续更新网络权值
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1: # 等一局对弈完才开始训练
            time.sleep(0.1)
        
        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            total_loss, U_loss, H_loss = self.update_weights(batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "U_loss": U_loss,
                    "H_loss": H_loss,
                }
            )
            # Managing the self-play / training ratio  # 调整自对弈/训练比率
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio_max # ratio改成区间, 大于最大值才sleep
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    # print("adjust ratio: training wait for 0.5s")
                    time.sleep(0.5)

    def update_weights(self, batch): # 更新一次网络权值
        observation_batch, target_U, target_H = batch

        device = next(self.model.parameters()).device
        observation_batch = torch.from_numpy(observation_batch).float().to(device)
        target_U = torch.from_numpy(target_U).float().to(device)
        target_H = torch.from_numpy(target_H).float().to(device)

        # Predict  # 模型预测
        U, H = self.model(observation_batch)
        # Compute loss  # 计算损失
        U_loss, H_loss = self.loss_function(U, H, target_U, target_H) # 计算策略、价值损失
        loss = U_loss + H_loss * self.config.H_loss_weight  # 总损失, 可以调节H损失的占比
        # Optimize  # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return loss.item(), U_loss.item(), H_loss.item()
    

    @staticmethod
    def loss_function(U, H, target_U, target_H): # 损失函数
        U_loss = F.mse_loss(U, target_U)  # 均方损失
        H_loss = F.mse_loss(H, target_H)  # 均方损失
        return U_loss, H_loss

