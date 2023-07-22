#include "go_comm.h"

int BOARD_SIZE = GoComm::UPPER_BOARD_SIZE;     // 实际棋盘尺寸, 可变, 默认19
Coord MAX_COORD = BOARD_SIZE * BOARD_SIZE;     // 实际最大坐标数, 可变