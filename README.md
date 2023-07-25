# ToPA强化学习算法

1. 我们提出ToPA算法（Monte Carlo Tree Search on both Planning and Acting），不仅在用于改进策略的规划方法（planning）使用树搜索，并将其扩展到实际自对弈（acting）中，整个算法可以视为“MCTS中的MCTS”。ToPA仍然是一个类AlphaZero算法。
2. ToPA算法与最大熵蒙特卡洛树搜索（MENTS）结合，称为METPA（Maximum Entropy Tree Search on both Planning and Acting）算法，证明了最大熵算法和类AlphaZero算法结合的可行性。

本程序使用C++（环境部分）和python（算法部分）编写，并使用ray库实现分布式和并行化。

## 训练
./train.sh

## 测试
./test.sh

若要更改测试模型，请在elo.py中更改。

## 各部分说明
### envs
强化学习环境

#### go_env
围棋环境（固定棋盘尺度支持9x9、19x19，使用9时更节省内存）
使用C++编写（再由environment.py调用动态库）

#### go_env_variable
围棋环境（棋盘尺度1x1~19x19，占用内存按19x19）
使用C++编写（再由environment.py调用动态库）

### models
网络结构
包括ResBlock, Transformer等层的定义和总体网络结构

### search
mcts：搜索算法
nodes：节点定义
select_action：动作选择算法

### configs
一般参数设置

### replay buffer
经验回放池

### replay tree
经验回放树，可看作树结构的replay buffer，是短期记忆树的前身

### st_memory_tree
短期记忆树，replay tree的改进

### self_play
自对弈worker，不断地进行自对弈，并在每一步都调用search.mcts来获得决策

### trainer
网络训练器，设置优化器、损失函数、学习率变化、训练流程等

### shared_storage
实现各个worker之间的信息共享

### main_ToPA
算法主程序，采用并行结构。N个self_play worker + 1个策略评估worker + 1个replay buffer（或replay tree/st_memory_tree）worker + 一个Trainer worker + 一个shared_storage worker之间协同工作
达到设定训练步数自动停止训练

### elo
用于测试，评估训练出来的模型elo评分，结果保存到test_results中

更多详情见设计文档（design docs）。
