
from audioop import add
from ctypes import *
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

import platform

BLACK = 1
WHITE = 2

# 构造一个跟c语言里一模一样的结构体来传输GoState
class c_GoState(ctypes.Structure):     # 围棋状态, 见go_env.h的GoState结构  # 基类为ctypes.Structure
    pass

def create_c_structs():
    """ 创建c语言结构体
    """
    class c_Info(ctypes.Structure):    # 棋盘格点信息, 见board.h的Info结构
        pass
    class c_Block(ctypes.Structure):   # 连通块, 见board.h的Block结构
        pass
    class c_Board(ctypes.Structure):   # 棋盘, 见board.h的Board结构
        pass

    BOARD_SIZE = 19
    MAX_COORD = BOARD_SIZE * BOARD_SIZE  # 最大坐标数
    MAX_BLOCK = 256         # 最大连通块数
    
    MAX_HISTORY_DIM = 1     # 最大历史棋盘数 (要跟./cpp_src/go_env.h里的GoState::MAX_HISTORY_DIM保持一致)

    c_Info._fields_ = [('color', c_uint8), ('id', c_int16), ('next', c_int16), ('last_placed', c_uint16)]
    c_Block._fields_ = [('color', c_uint8), ('start', c_int16), ('num_stones', c_int16), ('liberties', c_int16)]
    c_Board._fields_ = [('infos', c_Info * MAX_COORD), ('blocks', c_Block * MAX_BLOCK), ('next_player', c_int16),
                ('step_count', c_uint16), ('last_move1', c_int16), ('last_move2', c_int16), ('removed_block_ids', c_int*4),
                ('num_block_removed', c_int16), ('ko_location', c_int16), ('ko_color', c_uint8), ('ko_age', c_int16)]
    c_GoState._fields_ = [('_boards', c_Board * MAX_HISTORY_DIM), ('_terminate', c_bool)]


class GoEnv:
    """ 围棋环境
    """
    def __init__(self, 
            config=None, 
            #------------新增以下默认值, 使之在没有config的情况下也能运行-----------
            board_size=9, 
            encoded_dim=10, 
            history_dim=1, 
            max_step=120, 
            komi=7.5, 
            legal_no_eye=True, 
            ban_pass_until=100, 
            add_color_plain=True, 
            effect_type='exp', 
            effect_scale_factor=1.0, 
            effect_pow_factor=0.5, 
        ):
        
        self.board_size = board_size              # 棋盘大小
        self.encoded_dim = encoded_dim            # 编码的特征平面数M
        self.history_dim = history_dim            # 每个状态包含的历史棋盘数N
        self.max_step = max_step                  # 棋局最大步数
        self.komi = komi                          # 贴目数
        self.no_eye = legal_no_eye                # 合法动作不含 己方真眼
        self.ban_pass_until = ban_pass_until      # 若干步禁止停着
        self.add_color_plain = add_color_plain    # 是否增加一个颜色平面

        # ------------影响域参数---------------
        self.effect_type = effect_type                   # 影响域类型   # 有指数型'exp' 和 倒数型'inver'
        self.effect_scale_factor = effect_scale_factor   # 影响域参数a
        self.effect_pow_factor = effect_pow_factor       # 影响域参数b

        if config:
            self.config = config
            self.board_size = config.board_size              # 棋盘大小
            self.encoded_dim = config.encode_state_channels  # 编码的特征平面数M
            #self.history_dim = 1                             # 每个状态包含的历史棋盘数N
            self.max_step = config.max_step                  # 棋局最大步数
            self.komi = config.komi                          # 贴目数
            self.no_eye = config.legal_no_eye                # 合法动作不含 己方真眼
            self.ban_pass_until = config.ban_pass_until      # 若干步禁止停着
            self.add_color_plain = config.add_color_plain    # 是否增加一个颜色平面        
        
        # ------------影响域参数---------------
            self.effect_type = config.effect_type                   # 影响域类型   # 有指数型'exp' 和 倒数型'inver'
            self.effect_scale_factor = config.effect_scale_factor   # 影响域参数a
            self.effect_pow_factor = config.effect_pow_factor       # 影响域参数b
        
        create_c_structs()                # 构建c语言结构体

        # ------------加载动态库---------------
        
        # 新增: 根据系统选择dll或者so

        if "Windows" in platform.platform():  # Windows加载dll
            if int(platform.python_version().split(".")[1]) >= 8: # python>=3.8的版本要加winmode=0才能加载dll
                self.lib = ctypes.CDLL("./envs/go_env_variable/go_env_var.dll", winmode=0)   # 加载动态库
            else:
                self.lib = ctypes.CDLL("./envs/go_env_variable/go_env_var.dll")
        
        elif "Linux" in platform.platform():  # Linux加载so
            self.lib = ctypes.cdll.LoadLibrary("./envs/go_env_variable/go_env_var.so")
        
        else:
            raise NotImplementedError('Only support Windows and Linux system.')
        
        # ----------指定动态库函数的参数列表和返回值----------
        
        self.c_init = self.lib.Init
        self.c_init.argtypes = [c_int, c_int, c_int , c_int, c_float]
        self.c_init.restype = c_bool
        self.c_init(self.board_size, self.history_dim, self.encoded_dim, self.max_step, self.komi)   # 初始化动态库
        
        self.c_reset = self.lib.Reset    # 重置状态
        self.c_reset.argtypes = [POINTER(c_GoState)]
        self.c_reset.restype = c_bool
        
        self.c_step = self.lib.Step      # 下一步, 返回棋局是否结束
        self.c_step.argtypes = [POINTER(c_GoState), POINTER(c_GoState), c_int]
        self.c_step.restype = c_bool
        
        self.c_encode = self.lib.Encode  # 编码特征平面   # 己方1,2,3及以上气连通块, 对方1,2,3及以上气连通块, 上一个历史落子点, 非法落子, 己方真眼, 己方活棋块
        self.c_encode.argtypes = [POINTER(c_GoState), ndpointer(c_float)]
        self.c_encode.restype = c_bool
        
        self.c_getScore = self.lib.getScore  # 获取盘面差
        self.c_getScore.argtypes = [POINTER(c_GoState)]
        self.c_getScore.restype = c_float
        
        self.c_isTerminated = self.lib.isTerminated
        self.c_isTerminated.argtypes = [POINTER(c_GoState)]
        self.c_isTerminated.restype = c_bool
        
        self.c_getTerritory = self.lib.getTerritory  # 获取盘面差及归属, 返回值是盘面差
        self.c_getTerritory.argtypes = [POINTER(c_GoState), ndpointer(c_float)]
        self.c_getTerritory.restype = c_float

        self.c_setEffectionDomainParam = self.lib.setEffectionDomainParam   # 设置影响域参数
        self.c_setEffectionDomainParam.argtypes = [c_int, c_double, c_double]
        self.c_setEffectionDomainParam.restype = c_bool
        self._setEffectionParams(self.effect_type, self.effect_scale_factor, self.effect_pow_factor)
        
        self.c_get_H_effection = self.lib.getSumedEffectionDomain  # 获取基于影响域的盘面归属
        self.c_get_H_effection.argtypes = [POINTER(c_GoState), ndpointer(c_float)]
        self.c_get_H_effection.restype = None
        
        self.c_getLegalAction = self.lib.getLegalAction  # 获取合法动作集, 返回值是动作数
        self.c_getLegalAction.argtypes = [POINTER(c_GoState), ndpointer(c_int)]
        self.c_getLegalAction.restype = c_int
        
        self.c_getLegalNoEye = self.lib.getLegalNoEye    # 获取合法动作集(不含己方真眼), 返回值是动作数    # 真眼的定义见board.cc中的isTrueEye()函数
        self.c_getLegalNoEye.argtypes = [POINTER(c_GoState), ndpointer(c_int)]
        self.c_getLegalNoEye.restype = c_int
        
        self.c_getBoard = self.lib.getBoard   # 获取棋盘
        self.c_getBoard.argtypes = [POINTER(c_GoState), ndpointer(c_int)]
        self.c_getBoard.restype = None
        
        self.c_getPlayer = self.lib.getPlayer  # 获取下一个玩家
        self.c_getPlayer.argtypes = [POINTER(c_GoState)]
        self.c_getPlayer.restype = c_int
        
        self.c_getStep = self.lib.getStep   # 获取步数
        self.c_getStep.argtypes = [POINTER(c_GoState)]
        self.c_getStep.restype = c_int
    
    def reset(self):
        """ 重置 """
        state = c_GoState()
        self.c_reset(state)
        legal_actions = self.getLegalAction(state)
        observation = self.encode(state)
        return state, legal_actions, observation
    
    def step(self, state, action):
        """ 下一步 """
        new_state = c_GoState()
        done = self.c_step(state, new_state, action)
        legal_actions = self.getLegalAction(new_state)
        observation = self.encode(new_state)
        return new_state, legal_actions, observation, done
        
    def encode(self, state):
        """ 编码特征平面, (M==10时)己方1,2,3及以上气连通块, 对方1,2,3及以上气连通块, 上一个历史落子点, 非法落子, 己方真眼, 己方活棋块
            返回(N*M, board_size, board_size)的numpy数组
        """
        encoded_state = np.zeros([self.history_dim * self.encoded_dim, self.board_size, self.board_size], dtype="float32")
        self.c_encode(state, encoded_state)
        # 增加一个颜色平面, 表示当前玩家, 黑子全1, 白子全0
        if self.add_color_plain:
            if self.getPlayer(state) == BLACK:
                color_plain = np.ones([1, self.board_size, self.board_size], dtype="float32")
            else:
                color_plain = np.zeros([1, self.board_size, self.board_size], dtype="float32")
            
            encoded_state = np.concatenate([encoded_state, color_plain], axis=0) # 拼接到通道上

        return encoded_state
    
    def getScore(self, state):
        """ 获取基于Tramp-Taylor规则的盘面差(含贴目)
        """
        return self.c_getScore(state)
     
    def get_H_TrompTaylor(self, state, flatten=True):
        """ 获取基于Tramp-Taylor规则的盘面归属(黑1.0, 中立0.0, 白-1.0)
        """
        H = np.zeros([self.board_size, self.board_size], dtype="float32")
        self.c_getTerritory(state, H)
        if flatten:
            H = H.flatten()
        return H
    
    def _setEffectionParams(self, type, scale_factor, pow_factor):
        """ 设置影响域类型及参数a, b
        """
        if type == 'exp':
            effect_type = 1
        elif type == 'inver':
            effect_type = 2
        else:
            raise NotImplementedError("effection_type should be 'exp' or 'inver'.")

        self.c_setEffectionDomainParam(effect_type, scale_factor, pow_factor)

    def get_H_effection(self, state, flatten=True):
        """ 获取基于影响域的盘面归属(黑1.0, 中立0.0, 白-1.0)
        """
        H_e = np.zeros([self.board_size, self.board_size], dtype="float32")
        self.c_get_H_effection(state, H_e)
        if flatten:
            H_e = H_e.flatten()
        return H_e

    def getWinner(self, state):
        """ 获取赢家, 动态库里没有专门获取赢家的函数, 可以通过盘面差是否>0来判断
        """
        return BLACK if self.getScore(state) > 0 else WHITE

    def getLegalAction(self, state):
        """ 返回合法动作, 数组长度等于动作数
            或返回不含 己方真眼 的合法动作, 数组长度等于动作数
            真眼的定义见board.cc中的isTrueEye()函数
        """
                
        legal_action = np.zeros([self.board_size * self.board_size + 1], dtype='int32')
        if self.no_eye:  # 不填真眼
            num_action = self.c_getLegalNoEye(state, legal_action)
        else:
            num_action = self.c_getLegalAction(state, legal_action)
        
        if self.getStep(state) < self.ban_pass_until and num_action > 1:
            return legal_action[:num_action - 1]  # -1去掉停着

        return legal_action[:num_action] # 切片, 只保留前num_action个动作
    
    def getPlayer(self, state):
        """ 获取下一个玩家 """
        return self.c_getPlayer(state)
    
    def getStep(self, state):
        """ 获取游戏步数 """
        return self.c_getStep(state)

    def render(self, state):     
        """ 显示棋盘 """
        board = np.zeros([self.board_size * self.board_size], dtype='int32')
        self.c_getBoard(state, board)
        board_str = []
        for color in board:
            if color == WHITE:
                ch = 'O' # 白棋
            elif color == BLACK:
                ch = 'X' # 黑棋
            else:
                ch = '.' # 空点
            board_str.append(ch)
        for i in range(self.board_size):
            for j in range(self.board_size):
                print(board_str[i * self.board_size + j], end=' ')
            print('')


if __name__ == '__main__':
    env = GoEnv()
    state, _, _ = env.reset()
    env.render(state)
