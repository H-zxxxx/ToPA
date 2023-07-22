import datetime
import os

import torch
import math

#--------------一般设置---------------

class Config:   # 不同gpu跟cpu性能, 要调节的参数 1、batch_size; 2、num_workers; 3、ratio

    def __init__(self):
        self.seed = 0   # numpy和torch的随机数种子
        
    #----------环境相应参数------------
        self.board_size = 9              # 棋盘大小, 现在支持1～19任意尺度了, 不用重新编译
        self.encode_state_channels = 17  # 己方1,2,3及以上气连通块, 对方1,2,3及以上气连通块,上一个历史落子点,非法落子,己方真眼,己方活棋块
                                         # 有三种可选； 9个特征；10个特征；后面加了一个13特征
        self.komi = 7.5                  # 贴目
        self.black = 1
        self.white = 2
        self.max_step = 120              # 棋局最大步数
        self.legal_no_eye = True         # True表示合法动作不含己方真眼
        self.ban_pass_until = 100        # 100步之前禁止停着(除非无处可下)

        self.add_color_plain = False      # 增加一个颜色平面, 表示当前玩家的颜色

        #----------影响域参数-----------
        self.effect_type = 'inver'       # 影响域类型  # 可选'inver'和'exp'
        self.effect_scale_factor = 1.0   # 影响域参数a
        self.effect_pow_factor = 0.5     # 影响域参数b
    
    #-------------储存区参数-------------
        self.replay_buffer_size = 10000   # 是局数, 不是步数!
    
    #-------------自对弈参数-------------
        self.num_workers = 8              # 自对弈的线程数
        self.c_puct = 3
        self.num_simulations = 210           # 搜索次数
        self.root_dirichlet_alpha = 0.03  # 狄利克雷噪声
        self.root_exploration_fraction = 0.25
        self.action_space = list(range(self.board_size**2 + 1)) # 动作空间, 棋盘各点+停着

    #--------------网络模型--------------
        self.input_dim = 6 #self.encode_state_channels  # 输入通道数
        self.auxiliary_output_channels = self.encode_state_channels - self.input_dim

        if self.add_color_plain:
            self.input_dim += 1   # 颜色平面+1通道

        self.network_type = "Conv"     # 网络类型, 可选"Conv", "Transformer"和"Hybrid"

        #-----------卷积部分------------
        self.num_features = 96          # 特征图通道数
        self.num_blocks = 6              # 残差块数
        
        #--------Transformer部分---------
        self.depth = 6                   # 网络深度
        self.patch_size = 3              # 切块的大小
        self.patch_stride = 2            # 切块的滑动步长, 等于patch_size时切块无重叠
        self.embed_dim = 192             # 嵌入维度, 即每个patch表示的向量维度
        self.num_heads = 6               # 多头注意力的头数
        
        #------Hybrid(混合模型)部分-------
        self.depths = [5, 3]             # [conv_depth, transformer_depth]
        # 其余参数同上
    
    #--------------训练参数--------------
        self.optimizer = "Adam"          # 优化器, 可选"SGD", "Adam"
        self.momentum = 0.9              # 动量参数, 仅用于SGD优化器
        self.l2_const = 1e-4             # 网络参数的2范数惩罚项
        self.checkpoint_interval = 6     # 每训练6次更新一次自对弈模型
        self.batch_size = 2048

        #---------学习率指数衰减----------
        self.lr_init = 5.5e-4            # 初始学习率
        self.lr_decay_rate = 0.5         # 设为1是恒定学习率
        self.lr_decay_steps = 50000      # lr_decay_steps步时衰减到初始值的lr_decay_rate倍
        self.lr_final = self.lr_init / 4 # 最终学习率, 达到此学习率后不再下降
        
        self.training_steps = int(1e6)   # 总训练步数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./results", datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True           # 是否保存模型

        self.test_results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./test_results", datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        
    
    #----------------评估----------------
        self.evaluate_num = 1500         # 间隔 1500 个计数 评估一次

        self.init_model = False
        #self.init_model_path = "./results/2022-11-17--16-50-10/best_policy_15.model" # 下了5505局的模型
        #self.init_model_path = "./results/2023-02-15--17-19-33/best_policy_12.model"
        #self.init_model_path = "./results/2023-02-16--10-52-21/best_policy_5.model"
        self.init_model_path = "./results/2023-02-16--19-09-51/best_policy_6.model"
    
    #------------控制温度系数-------------
    def epsilon_by_frame(self, game_step):  # 温度系数衰减
        epsilon_start = 1.0
        epsilon_final = 0.65 * epsilon_start
        epsilon_decay = 10
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * game_step / epsilon_decay)
    

#--------------AlphaZero的设置---------------

class ToPAConfig(Config):
    def __init__(self):
        super().__init__()

    #--------------训练参数--------------
        self.value_loss_weight = 1     # 价值损失的比重
        self.buffer_tree = True

        #--------调整训练/自对弈比率------- (改成区间)
        self.ratio = True
        self.ratio_max = 0.400           # 比率上限
        self.ratio_min = 0.200           # 比率下限

        self.c_piUCT = 3   # piUCT的c
        self.temperature = 0.2
        self.exploration_param = 1.0

        self.select_action1 = "E2W"  # "E2W" or "PUCT" of "softUCT"
        self.select_action2 = "piUCT" # "piUCT" or "E2W"
        self.anti_exploit = True

        self.use_st_memory_tree = True   # 使用短期记忆树
        self.st_memory_tree_size = 3000  # 3000局
