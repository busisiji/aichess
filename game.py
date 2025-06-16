"""棋盘游戏控制"""

import numpy as np
import copy
import time
from config import CONFIG
from collections import deque   # 这个队列用来判断长将或长捉
import random


# 列表来表示棋盘，红方在上，黑方在下。使用时需要使用深拷贝
state_list_init = [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
                   ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
                   ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]


# deque来存储棋盘状态，长度为4
state_deque_init = deque(maxlen=4)
for _ in range(4):
    state_deque_init.append(copy.deepcopy(state_list_init))


# 构建一个字典：字符串到数组的映射，函数：数组到字符串的映射
string2array = dict(红车=np.array([1, 0, 0, 0, 0, 0, 0]), 红马=np.array([0, 1, 0, 0, 0, 0, 0]),
                    红象=np.array([0, 0, 1, 0, 0, 0, 0]), 红士=np.array([0, 0, 0, 1, 0, 0, 0]),
                    红帅=np.array([0, 0, 0, 0, 1, 0, 0]), 红炮=np.array([0, 0, 0, 0, 0, 1, 0]),
                    红兵=np.array([0, 0, 0, 0, 0, 0, 1]), 黑车=np.array([-1, 0, 0, 0, 0, 0, 0]),
                    黑马=np.array([0, -1, 0, 0, 0, 0, 0]), 黑象=np.array([0, 0, -1, 0, 0, 0, 0]),
                    黑士=np.array([0, 0, 0, -1, 0, 0, 0]), 黑帅=np.array([0, 0, 0, 0, -1, 0, 0]),
                    黑炮=np.array([0, 0, 0, 0, 0, -1, 0]), 黑兵=np.array([0, 0, 0, 0, 0, 0, -1]),
                    一一=np.array([0, 0, 0, 0, 0, 0, 0]))


def array2string(array):
    return list(filter(lambda string: (string2array[string] == array).all(), string2array))[0]


# 改变棋盘状态
def change_state(state_list, move):
    """move : 字符串'0010'"""
    copy_list = copy.deepcopy(state_list)
    y, x, toy, tox = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    copy_list[toy][tox] = copy_list[y][x]
    copy_list[y][x] = '一一'
    return copy_list


# 打印盘面，可视化用到
def print_board(_state_array):
    # _state_array: [10, 9, 7], HWC
    board_line = []
    for i in range(10):
        for j in range(9):
            board_line.append(array2string(_state_array[i][j]))
        print(board_line)
        board_line.clear()


# 列表棋盘状态到数组棋盘状态
def state_list2state_array(state_list):
    _state_array = np.zeros([10, 9, 7])
    for i in range(10):
        for j in range(9):
            _state_array[i][j] = string2array[state_list[i][j]]
    return _state_array


# 拿到所有合法走子的集合，2086长度，也就是神经网络预测的走子概率向量的长度
# 第一个字典：move_id到move_action
# 第二个字典：move_action到move_id
# 例如：move_id:0 --> move_action:'0010'
def get_all_legal_moves():
    _move_id2move_action = {}
    _move_action2move_id = {}
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # 士的全部走法
    advisor_labels = ['0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
                      '9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']
    # 象的全部走法
    bishop_labels = ['2002', '0220', '2042', '4220', '0224', '2402', '4224', '2442',
                     '2406', '0624', '2446', '4624', '0628', '2806', '4628', '2846',
                     '7052', '5270', '7092', '9270', '5274', '7452', '9274', '7492',
                     '7456', '5674', '7496', '9674', '5678', '7856', '9678', '7896']
    idx = 0
    for l1 in range(10):
        for n1 in range(9):
            destinations = [(t, n1) for t in range(10)] + \
                           [(l1, t) for t in range(9)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
                    action = column[l1] + row[n1] + column[l2] + row[n2]
                    _move_id2move_action[idx] = action
                    _move_action2move_id[action] = idx
                    idx += 1

    for action in advisor_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    for action in bishop_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    return _move_id2move_action, _move_action2move_id


move_id2move_action, move_action2move_id = get_all_legal_moves()


# 走子翻转的函数，用来扩充我们的数据
def flip_map(string):
    new_str = ''
    for index in range(4):
        if index == 0 or index == 2:
            new_str += (str(string[index]))
        else:
            new_str += (str(8 - int(string[index])))
    return new_str


# 边界检查
def check_bounds(toY, toX):
    if toY in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and toX in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        return True
    return False


# 不能走到自己的棋子位置
def check_obstruct(piece, current_player_color):
    # 当走到的位置存在棋子的时候，进行一次判断
    if piece != '一一':
        if current_player_color == '红':
            if '黑' in piece:
                return True
            else:
                return False
        elif current_player_color == '黑':
            if '红' in piece:
                return True
            else:
                return False
    else:
        return True
def is_in_check(state_list, color):
    """
    判断当前玩家是否处于“被将军”的状态。
    即对方有一个合法走法可以吃掉己方的帅。
    """
    # 找到己方的帅的位置
    king_pos = None
    for y in range(10):
        for x in range(9):
            if (color == '红' and state_list[y][x] == '红帅') or \
               (color == '黑' and state_list[y][x] == '黑帅'):
                king_pos = (y, x)
                break
        if king_pos:
            break

    # 遍历所有敌方棋子，看看是否有能走到帅的位置
    enemy_color = '黑' if color == '红' else '红'
    for y in range(10):
        for x in range(9):
            piece = state_list[y][x]
            if enemy_color in piece:
                # 模拟这个棋子走到 king_pos 是否合法
                move_str = f"{y}{x}{king_pos[0]}{king_pos[1]}"
                next_state = change_state(state_list, move_str)
                if move_str in [move_id2move_action[m] for m in get_legal_moves(deque([next_state], maxlen=4), enemy_color)]:
                    return True
    return False
def is_move_into_check(state_list, move_str, current_player_color):
    """
    检查一个移动是否会导致送将（即移动后己方将帅处于被将军状态）
    """
    next_state = change_state(state_list, move_str)
    return is_in_check(next_state, current_player_color)

# 根据当前状态和动作过滤不合法的动作
def apply_rules_pruning(policy, state):
    """
    根据当前棋盘状态，将不合法动作的概率设置为0，并重新归一化
    """
    legal_moves = state.availables
    legal_mask = np.zeros_like(policy)
    legal_mask[legal_moves] = 1

    # 应用掩码并重新归一化
    policy *= legal_mask
    if policy.sum() > 0:
        policy /= policy.sum()
    else:
        # 如果所有动作都被屏蔽，回退到均匀分布
        policy = np.ones_like(policy) / len(policy)

    return policy



def filter_legal_moves(state_deque, moves, current_player_color):
    """
    过滤掉会导致送将或非法对脸的走法
    """
    state_list = state_deque[-1]
    legal_moves = []

    for move in moves:
        move_str = move_id2move_action[move]
        next_state = change_state(state_list, move_str)

        # 排除导致将帅对脸的走法
        if is_king_face_to_face(next_state):
            continue

        # 排除导致送将的走法
        if is_move_into_check(state_list, move_str, current_player_color):
            continue

        legal_moves.append(move)

    return legal_moves
# 得到当前盘面合法走子集合
# 输入状态队列不能小于10，current_player_color:当前玩家控制的棋子颜色
# 用来存放合法走子的列表，例如[0, 1, 2, 1089, 2085]
def get_legal_moves(state_deque, current_player_color):
    state_list = state_deque[-1]
    old_state_list = state_deque[-4]

    moves = []

    # 遍历棋盘
    for y in range(10):
        for x in range(9):
            if state_list[y][x] == '一一':
                continue

            # 黑车
            if state_list[y][x] == '黑车' and current_player_color == '黑':
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    while check_bounds(ny, nx):
                        if check_obstruct(state_list[ny][nx], current_player_color='黑'):
                            m = f"{y}{x}{ny}{nx}"
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        if state_list[ny][nx] != '一一':
                            break
                        ny += dy
                        nx += dx

            # 红车
            elif state_list[y][x] == '红车' and current_player_color == '红':
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    while check_bounds(ny, nx):
                        if check_obstruct(state_list[ny][nx], current_player_color='红'):
                            m = f"{y}{x}{ny}{nx}"
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        if state_list[ny][nx] != '一一':
                            break
                        ny += dy
                        nx += dx

            # 黑马
            elif state_list[y][x] == '黑马' and current_player_color == '黑':
                knight_moves = [
                    (-2, -1), (-2, 1),
                    (-1, -2), (-1, 2),
                    (1, -2), (1, 2),
                    (2, -1), (2, 1)
                ]
                for dy, dx in knight_moves:
                    ny, nx = y + dy, x + dx
                    if check_bounds(ny, nx):
                        eye_y, eye_x = y + dy // 2, x + dx // 2
                        if state_list[eye_y][eye_x] == '一一' and check_obstruct(state_list[ny][nx], current_player_color='黑'):
                            m = f"{y}{x}{ny}{nx}"
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

            # 红马
            elif state_list[y][x] == '红马' and current_player_color == '红':
                knight_moves = [
                    (-2, -1), (-2, 1),
                    (-1, -2), (-1, 2),
                    (1, -2), (1, 2),
                    (2, -1), (2, 1)
                ]
                for dy, dx in knight_moves:
                    ny, nx = y + dy, x + dx
                    if check_bounds(ny, nx):
                        eye_y, eye_x = y + dy // 2, x + dx // 2
                        if state_list[eye_y][eye_x] == '一一' and check_obstruct(state_list[ny][nx], current_player_color='红'):
                            m = f"{y}{x}{ny}{nx}"
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

            # 黑象
            elif state_list[y][x] == '黑象' and current_player_color == '黑':
                bishop_moves = [
                    (2, 2), (2, -2),
                    (-2, 2), (-2, -2)
                ]
                for dy, dx in bishop_moves:
                    ny, nx = y + dy, x + dx
                    if check_bounds(ny, nx) and ny >= 5:
                        eye_y, eye_x = y + dy // 2, x + dx // 2
                        if state_list[eye_y][eye_x] == '一一' and check_obstruct(state_list[ny][nx], current_player_color='黑'):
                            m = f"{y}{x}{ny}{nx}"
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

            # 红象
            elif state_list[y][x] == '红象' and current_player_color == '红':
                bishop_moves = [
                    (2, 2), (2, -2),
                    (-2, 2), (-2, -2)
                ]
                for dy, dx in bishop_moves:
                    ny, nx = y + dy, x + dx
                    if check_bounds(ny, nx) and ny <= 4:
                        eye_y, eye_x = y + dy // 2, x + dx // 2
                        if state_list[eye_y][eye_x] == '一一' and check_obstruct(state_list[ny][nx], current_player_color='红'):
                            m = f"{y}{x}{ny}{nx}"
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

            # 黑士
            elif state_list[y][x] == '黑士' and current_player_color == '黑':
                advisor_positions = {(7, 3), (7, 5), (8, 4), (9, 3), (9, 5)}
                for ty, tx in advisor_positions:
                    if check_obstruct(state_list[ty][tx], current_player_color='黑'):
                        m = f"{y}{x}{ty}{tx}"
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

            # 红士
            elif state_list[y][x] == '红士' and current_player_color == '红':
                advisor_positions = {(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)}
                for ty, tx in advisor_positions:
                    if check_obstruct(state_list[ty][tx], current_player_color='红'):
                        m = f"{y}{x}{ty}{tx}"
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

            # 黑帅
            elif state_list[y][x] == '黑帅' and current_player_color == '黑':
                king_positions = {(7, 3), (7, 4), (7, 5),
                                  (8, 3), (8, 4), (8, 5),
                                  (9, 3), (9, 4), (9, 5)}
                for ty, tx in king_positions:
                    if check_obstruct(state_list[ty][tx], current_player_color='黑'):
                        m = f"{y}{x}{ty}{tx}"
                        next_state = change_state(state_list, m)
                        if not is_king_face_to_face(next_state):
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

            # 红帅
            elif state_list[y][x] == '红帅' and current_player_color == '红':
                king_positions = {(0, 3), (0, 4), (0, 5),
                                  (1, 3), (1, 4), (1, 5),
                                  (2, 3), (2, 4), (2, 5)}
                for ty, tx in king_positions:
                    if check_obstruct(state_list[ty][tx], current_player_color='红'):
                        m = f"{y}{x}{ty}{tx}"
                        next_state = change_state(state_list, m)
                        if not is_king_face_to_face(next_state):
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

            # 黑炮
            elif state_list[y][x] == '黑炮' and current_player_color == '黑':
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                for dy, dx in directions:
                    step = 1
                    hit_flag = False
                    while True:
                        ny, nx = y + dy * step, x + dx * step
                        if not check_bounds(ny, nx):
                            break
                        if not hit_flag:
                            if state_list[ny][nx] != '一一':
                                hit_flag = True
                        else:
                            if state_list[ny][nx] != '一一':
                                if '红' in state_list[ny][nx]:
                                    m = f"{y}{x}{ny}{nx}"
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                            break
                        step += 1

            # 红炮
            elif state_list[y][x] == '红炮' and current_player_color == '红':
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                for dy, dx in directions:
                    step = 1
                    hit_flag = False
                    while True:
                        ny, nx = y + dy * step, x + dx * step
                        if not check_bounds(ny, nx):
                            break
                        if not hit_flag:
                            if state_list[ny][nx] != '一一':
                                hit_flag = True
                        else:
                            if state_list[ny][nx] != '一一':
                                if '黑' in state_list[ny][nx]:
                                    m = f"{y}{x}{ny}{nx}"
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                            break
                        step += 1

            # 黑兵
            elif state_list[y][x] == '黑兵' and current_player_color == '黑':
                if y > 0:
                    for dx in [-1, 0, 1]:
                        nx = x + dx
                        if check_bounds(y - 1, nx):
                            if check_obstruct(state_list[y - 1][nx], current_player_color='黑'):
                                m = f"{y}{x}{y-1}{nx}"
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)

            # 红兵
            elif state_list[y][x] == '红兵' and current_player_color == '红':
                if y < 9:
                    for dx in [-1, 0, 1]:
                        nx = x + dx
                        if check_bounds(y + 1, nx):
                            if check_obstruct(state_list[y + 1][nx], current_player_color='红'):
                                m = f"{y}{x}{y+1}{nx}"
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)

    # 检查是否进入非法对脸状态
    for move in list(moves):  # 使用副本避免修改迭代对象
        move_str = move_id2move_action[move]
        next_state = change_state(state_list, move_str)
        if is_king_face_to_face(next_state):
            moves.remove(move)

    # 检查是否走入“被将军”状态
    for move in list(moves):
        move_str = move_id2move_action[move]
        next_state = change_state(state_list, move_str)
        if is_in_check(next_state, current_player_color):
            moves.remove(move)

    # 过滤掉会导致送将或非法对脸的走法
    filtered_moves = []
    for move in moves:
        move_str = move_id2move_action[move]
        next_state = change_state(state_list, move_str)

        # 检查是否将帅对脸
        if is_king_face_to_face(next_state):
            continue

        # 检查是否送将
        if is_move_into_check(state_list, move_str, current_player_color):
            continue

        filtered_moves.append(move)

    return filtered_moves

# 检查重复局面（平局判负）
def is_repetition(state_deque):
    return state_deque.count(state_deque[-1]) >= 3
def is_king_face_to_face(state_list):
    """
    检查当前棋盘是否违反将帅不能直接照面的规则。

    :param state_list: 当前棋盘状态（二维列表）
    :return: True 表示违反规则，False 表示合法
    """
    k_pos = None
    K_pos = None

    # 找出黑帅和红帅的位置
    for y in range(10):
        for x in range(9):
            if state_list[y][x] == '黑帅':
                k_pos = (y, x)
            elif state_list[y][x] == '红帅':
                K_pos = (y, x)

    if not k_pos or not K_pos:
        return False  # 如果没有两个帅，则不违反规则

    k_y, k_x = k_pos
    K_y, K_x = K_pos

    # 只有在同一列时才需要检查
    if k_x != K_x:
        return False

    min_y, max_y = sorted([k_y, K_y])

    # 如果相邻，也视为违规
    if max_y - min_y <= 1:
        return True

    # 检查中间是否有其他棋子
    for y in range(min_y + 1, max_y):
        if state_list[y][k_x] == '一一':
            continue
        else:
            return False  # 中间有棋子，合法

    # 中间全是空位，说明将帅面对面且无遮挡，违规
    return True


# 棋盘逻辑控制
class Board(object):

    def __init__(self):
        self.state_list = copy.deepcopy(state_list_init)
        self.game_start = False
        self.winner = None
        self.state_deque = copy.deepcopy(state_deque_init)

    # 初始化棋盘的方法
    def init_board(self, start_player=1):   # 传入先手玩家的id
        # 增加一个颜色到id的映射字典，id到颜色的映射字典
        # 永远是红方先移动
        self.start_player = start_player

        if start_player == 1:
            self.id2color = {1: '红', 2: '黑'}
            self.color2id = {'红': 1, '黑': 2}
            self.backhand_player = 2
        elif start_player == 2:
            self.id2color = {2: '红', 1: '黑'}
            self.color2id = {'红': 2, '黑': 1}
            self.backhand_player = 1
        # 当前手玩家，也就是先手玩家
        self.current_player_color = self.id2color[start_player]     # 红
        self.current_player_id = self.color2id['红']
        # 初始化棋盘状态
        self.state_list = copy.deepcopy(state_list_init)
        self.state_deque = copy.deepcopy(state_deque_init)
        # 初始化最后落子位置
        self.last_move = -1
        # 记录游戏中吃子的回合数
        self.kill_action = 0
        self.game_start = False
        self.action_count = 0   # 游戏动作计数器
        self.winner = None

    @property
    # 获的当前盘面的所有合法走子集合
    def availables(self):
        return get_legal_moves(self.state_deque, self.current_player_color)

    # 从当前玩家的视角返回棋盘状态，current_state_array: [9, 10, 9]  CHW
    def current_state(self):
        _current_state = np.zeros([9, 10, 9])
        # 使用9个平面来表示棋盘状态
        # 0-6个平面表示棋子位置，1代表红方棋子，-1代表黑方棋子, 队列最后一个盘面
        # 第7个平面表示对手player最近一步的落子位置，走子之前的位置为-1，走子之后的位置为1，其余全部是0
        # 第8个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        _current_state[:7] = state_list2state_array(self.state_deque[-1]).transpose([2, 0, 1])  # [7, 10, 9]

        if self.game_start:
            # 解构self.last_move
            move = move_id2move_action[self.last_move]
            start_position = int(move[0]), int(move[1])
            end_position = int(move[2]), int(move[3])
            _current_state[7][start_position[0]][start_position[1]] = -1
            _current_state[7][end_position[0]][end_position[1]] = 1
        # 指出当前是哪个玩家走子
        if self.action_count % 2 == 0:
            _current_state[8][:, :] = 1.0

        return _current_state.copy()

    # 根据move对棋盘状态做出改变
    def do_move(self, move):
        self.game_start = True  # 游戏开始
        self.action_count += 1  # 移动次数加1
        move_action = move_id2move_action[move]
        start_y, start_x = int(move_action[0]), int(move_action[1])
        end_y, end_x = int(move_action[2]), int(move_action[3])
        state_list = copy.deepcopy(self.state_deque[-1])
        # 判断是否吃子
        if state_list[end_y][end_x] != '一一':
            # 如果吃掉对方的帅，则返回当前的current_player胜利
            self.kill_action = 0
            if self.current_player_color == '黑' and state_list[end_y][end_x] == '红帅':
                self.winner = self.color2id['黑']
            elif self.current_player_color == '红' and state_list[end_y][end_x] == '黑帅':
                self.winner = self.color2id['红']
        else:
            self.kill_action += 1
        # 更改棋盘状态
        state_list[end_y][end_x] = state_list[start_y][start_x]
        state_list[start_y][start_x] = '一一'
        self.current_player_color = '黑' if self.current_player_color == '红' else '红'  # 改变当前玩家
        self.current_player_id = 1 if self.current_player_id == 2 else 2
        # 记录最后一次移动的位置
        self.last_move = move
        self.state_deque.append(state_list)

    # 是否产生赢家
    def has_a_winner(self):
        """一共有三种状态，红方胜，黑方胜，平局"""
        if self.winner is not None:
            return True, self.winner
        elif self.kill_action >= CONFIG['kill_action']:  # 平局先手判负
            # return False, -1
            return True, self.backhand_player
        return False, -1

    # 检查当前棋局是否结束
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif self.kill_action >= CONFIG['kill_action']:  # 平局，没有赢家
            return True, -1
        return False, -1

    def get_current_player_color(self):
        return self.current_player_color

    def get_current_player_id(self):
        return self.current_player_id


# 在Board类基础上定义Game类，该类用于启动并控制一整局对局的完整流程，并收集对局过程中的数据，以及进行棋盘的展示
class Game(object):

    def __init__(self, board):
        self.board = board

    # 可视化
    def graphic(self, board, player1_color, player2_color):
        print('player1 take: ', player1_color)
        print('player2 take: ', player2_color)
        print_board(state_list2state_array(board.state_deque[-1]))

    # 用于人机对战，人人对战等
    def start_play(self, player1, player2, start_player=1, is_shown=1):
        if start_player not in (1, 2):
            raise Exception('start_player should be either 1 (player1 first) '
                            'or 2 (player2 first)')
        self.board.init_board(start_player)  # 初始化棋盘
        p1, p2 = 1, 2
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current_player = self.board.get_current_player_id()  # 红子对应的玩家id
            player_in_turn = players[current_player]  # 决定当前玩家的代理
            move = player_in_turn.get_action(self.board)  # 当前玩家代理拿到动作
            self.board.do_move(move)  # 棋盘做出改变
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    print("Game end. Winner is", players[winner])
                else:
                    print("Game end. Tie")
                return winner

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(self, player, is_shown=False, temp=1e-3, logger=None):
        self.board.init_board()     # 初始化棋盘, start_player=1
        p1, p2 = 1, 2
        states, mcts_probs, current_players = [], [], []
        # 开始自我对弈
        _count = 0
        while True:
            if _count % 20 == 0:
                start_time = time.time()
                move, move_probs = player.get_action(self.board,
                                                     temp=temp,
                                                     return_prob=1)
                result_msg = f'第{_count + 1}步，走一步要花: {time.time() - start_time}'
                if logger:
                    logger.info(result_msg)
            else:
                move, move_probs = player.get_action(self.board,
                                                     temp=temp,
                                                     return_prob=1)
            _count += 1
            # 保存自我对弈的数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            # 执行一步落子
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                # 从每一个状态state对应的玩家的视角保存胜负信息
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                else:
                    winner_z[:] = 0.0  # 平局情况下全部为 0
                # 重置蒙特卡洛根节点
                player.reset_player()
                if is_shown:
                    result_msg = "🤝 平局！" if winner == -1 else f"🏆 玩家 {winner} 获胜！"
                    print(result_msg)
                    if logger:
                        logger.info(result_msg)

                return winner, zip(states, mcts_probs, winner_z)


if __name__ == '__main__':
    # 测试array2string
    # _array = np.array([0, 0, 0, 0, 0, 0, 0])
    # print(array2num(_array))

    """# 测试change_state
    new_state = change_state(state_list_init, move='0010')
    for row in range(10):
        print(new_state[row])"""

    """# 测试print_board
    _state_list = copy.deepcopy(state_list_init)
    print_board(state_list2state_array(_state_list))"""

    """# 测试get_legal_moves
    moves = get_legal_moves(state_deque_init, current_player_color='黑')
    move_actions = []
    for item in moves:
        move_actions.append(move_id2move_action[item])
    print(move_actions)"""

    # 测试Board中的start_play
    # class Human1:
    #     def get_action(self, board):
    #         # print('当前是player1在操作')
    #         # print(board.current_player_color)
    #         # move = move_action2move_id[input('请输入')]
    #         move = random.choice(board.availables)
    #         return move
    #
    #     def set_player_ind(self, p):
    #         self.player = p
    #
    #
    # class Human2:
    #     def get_action(self, board):
    #         # print('当前是player2在操作')
    #         # print(board.current_player_color)
    #         # move = move_action2move_id[input('请输入')]
    #         move = random.choice(board.availables)
    #         return move
    #
    #     def set_player_ind(self, p):
    #         self.player = p
    #
    # human1 = Human1()
    # human2 = Human2()
    # game = Game(board=Board())
    # for i in range(20):
    #     game.start_play(human1, human2, start_player=2, is_shown=0)
    board = Board()
    board.init_board()




