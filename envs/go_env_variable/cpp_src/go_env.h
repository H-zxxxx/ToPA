// go_env.h -- 定义状态GoState, 封装go_env接口

#ifndef GO_ENV_H_
#define GO_ENV_H_

#include "board.h"
#include "board_feature.h"
#include <cstdio>
#include <cstring>


namespace GoState {
  
const int MAX_HISTORY_DIM = 1;  // 最大历史棋盘数(含当前棋盘)

// 围棋状态GoState
// 包含N个历史棋盘(含当前棋盘)
typedef struct {
    Board _boards[MAX_HISTORY_DIM];
    bool _terminated;
} GoState;

}

using State = GoState::GoState;
// using Action = Coord;

extern "C" {
// 接口声明
// 按C语言风格编译, 否则ctypes无法识别

// 初始化环境参数
bool Init(int board_size, int history_dim, int encode_dim, int max_step, float komi);

// 重置状态
bool Reset(State* state);

// 下一步
// 棋盘: action==0~360, Pass: 361(或-1)
bool Step(const State* state, State* new_state, Coord action);
bool Step_(State* state, Coord action);

// 检查动作合法性
bool checkAction(const State* state, Coord action);

// 棋局是否结束
bool isTerminated(const State* state);

// 编码成特征平面
bool Encode(const State* state, float* encode_state);

// 只获取盘面差(含贴目)
float getScore(const State* state);

// 获取盘面差和具体归属
float getTerritory(const State* state, float* territory);

// 获取所有合法动作
int getLegalAction(const State* state, int* actions);

// 获取不含己方真眼的合法动作
int getLegalNoEye(const State* state, int* actions);

// 显示棋盘
void Show(const State* state);

// 获取盘面
void getBoard(const State* state, int* stones);

// 获得下一个玩家(1:黑, 2:白)
Stone getPlayer(const State* state);

// 获取步数
int getStep(const State* state);

// 设置影响域类型和参数a, b
bool setEffectionDomainParam(int type, double scale_factor, double pow_factor);

// 基于影响域的盘面归属
void getSumedEffectionDomain(const State* state, float* sum_ed);

}

#endif