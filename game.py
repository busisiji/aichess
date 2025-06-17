# -*- coding: utf-8 -*-
"""
棋盘游戏控制 - 完整封装版本（增强象棋规则）
"""

import time
import numpy as np
import copy
from collections import deque
from typing import List, Tuple, Dict

from config import CONFIG

# 颜色定义
PLAYER_COLORS = {'RED': '红', 'BLACK': '黑'}

# 初始棋盘布局
state_list_init = [
    ['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
    ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
    ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
    ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
    ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
    ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
    ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
    ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
    ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
    ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']
]

# 初始化棋盘状态队列
state_deque_init = deque(maxlen=4)
for _ in range(4):
    state_deque_init.append(copy.deepcopy(state_list_init))

# 棋子表示映射
string2array = {
    '红车': np.array([1, 0, 0, 0, 0, 0, 0]),
    '红马': np.array([0, 1, 0, 0, 0, 0, 0]),
    '红象': np.array([0, 0, 1, 0, 0, 0, 0]),
    '红士': np.array([0, 0, 0, 1, 0, 0, 0]),
    '红帅': np.array([0, 0, 0, 0, 1, 0, 0]),
    '红炮': np.array([0, 0, 0, 0, 0, 1, 0]),
    '红兵': np.array([0, 0, 0, 0, 0, 0, 1]),
    '黑车': np.array([-1, 0, 0, 0, 0, 0, 0]),
    '黑马': np.array([0, -1, 0, 0, 0, 0, 0]),
    '黑象': np.array([0, 0, -1, 0, 0, 0, 0]),
    '黑士': np.array([0, 0, 0, -1, 0, 0, 0]),
    '黑帅': np.array([0, 0, 0, 0, -1, 0, 0]),
    '黑炮': np.array([0, 0, 0, 0, 0, -1, 0]),
    '黑兵': np.array([0, 0, 0, 0, 0, 0, -1]),
    '一一': np.array([0, 0, 0, 0, 0, 0, 0])
}

def array2string(array) -> str:
    return list(filter(lambda s: (string2array[s] == array).all(), string2array))[0]


def change_state(state_list, move: str) -> List[List[str]]:
    y, x, toy, tox = map(int, move)
    new_state = copy.deepcopy(state_list)
    new_state[toy][tox] = new_state[y][x]
    new_state[y][x] = '一一'
    return new_state


def print_board(_state_array):
    board_line = []
    for i in range(10):
        for j in range(9):
            board_line.append(array2string(_state_array[i][j]))
        print(board_line)
        board_line.clear()


def state_list2state_array(state_list):
    _state_array = np.zeros([10, 9, 7])
    for i in range(10):
        for j in range(9):
            _state_array[i][j] = string2array[state_list[i][j]]
    return _state_array


def get_all_legal_moves():
    """
    生成所有合法走法
    返回：
        move_id2move_action: {int: str}
        move_action2move_id: {str: int}
    """
    move_id2move_action = {}
    move_action2move_id = {}

    row = [str(i) for i in range(9)]
    column = [str(i) for i in range(10)]

    advisor_labels = ['0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
                      '9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']
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
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2),
                             (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and 0 <= l2 < 10 and 0 <= n2 < 9:
                    action = column[l1] + row[n1] + column[l2] + row[n2]
                    if action not in move_action2move_id:
                        move_id2move_action[idx] = action
                        move_action2move_id[action] = idx
                        idx += 1

    for action in advisor_labels + bishop_labels:
        if action not in move_action2move_id:
            move_id2move_action[idx] = action
            move_action2move_id[action] = idx
            idx += 1

    return move_id2move_action, move_action2move_id


move_id2move_action, move_action2move_id = get_all_legal_moves()


def flip_map(string: str) -> str:
    return ''.join([
        c if i % 2 == 0 else str(8 - int(c)) for i, c in enumerate(string)
    ])


def check_bounds(y, x):
    return 0 <= y < 10 and 0 <= x < 9


def check_obstruct(piece, current_player_color):
    if piece == '一一':
        return True
    return (current_player_color == '红' and '黑' in piece) or (current_player_color == '黑' and '红' in piece)


def is_king_face_to_face(state_list):
    k_pos = K_pos = None
    for y in range(10):
        for x in range(9):
            if state_list[y][x] == '黑帅':
                k_pos = (y, x)
            elif state_list[y][x] == '红帅':
                K_pos = (y, x)

    if not k_pos or not K_pos:
        return False
    if k_pos[1] != K_pos[1]:
        return False

    min_y, max_y = sorted([k_pos[0], K_pos[0]])
    for y in range(min_y + 1, max_y):
        if state_list[y][k_pos[1]] != '一一':
            return False
    return True


def is_in_check(state_list, player_color):
    king_pos = None
    for y in range(10):
        for x in range(9):
            if f'{player_color}帅' in state_list[y][x]:
                king_pos = (y, x)
                break
        if king_pos:
            break

    opponent_color = '黑' if player_color == '红' else '红'
    check_count = 0

    for y in range(10):
        for x in range(9):
            piece = state_list[y][x]
            if piece.startswith(opponent_color):
                legal_moves = get_piece_legal_moves(state_list, y, x, opponent_color)
                for move_str in legal_moves:
                    if move_str.endswith(f"{king_pos[0]}{king_pos[1]}"):
                        check_count += 1
    return check_count > 0


def is_both_check(state_list):
    return is_in_check(state_list, '红') and is_in_check(state_list, '黑')


def is_suffocated(state_list, player_color):
    if is_in_check(state_list, player_color):
        return False
    for y in range(10):
        for x in range(9):
            piece = state_list[y][x]
            if piece.startswith(player_color):
                legal_moves = get_piece_legal_moves(state_list, y, x, player_color)
                if legal_moves:
                    return False
    return True


class IllegalMoveError(Exception):
    pass


class Board:
    def __init__(self, enable_complex_repetition=True):
        self.state_list = copy.deepcopy(state_list_init)
        self.game_start = False
        self.winner = None
        self.state_deque = copy.deepcopy(state_deque_init)
        self.kill_action = 0
        self.action_count = 0
        self.start_player = 1
        self.id2color = {1: '红', 2: '黑'}
        self.color2id = {'红': 1, '黑': 2}
        self.backhand_player = 2
        self.current_player_color = '红'
        self.current_player_id = 1
        self.last_move = -1
        self.move_history = []
        self.enable_complex_repetition = enable_complex_repetition

    def init_board(self, start_player=1):
        self.start_player = start_player
        if start_player == 1:
            self.id2color = {1: '红', 2: '黑'}
            self.color2id = {'红': 1, '黑': 2}
            self.backhand_player = 2
        else:
            self.id2color = {2: '红', 1: '黑'}
            self.color2id = {'红': 2, '黑': 1}
            self.backhand_player = 1

        self.current_player_color = self.id2color[start_player]
        self.current_player_id = self.color2id['红']
        self.state_list = copy.deepcopy(state_list_init)
        self.state_deque = copy.deepcopy(state_deque_init)
        self.kill_action = 0
        self.game_start = False
        self.action_count = 0
        self.winner = None
        self.move_history = []

    @property
    def availables(self):
        return get_legal_moves(self.state_deque, self.current_player_color)

    def current_state(self):
        _current_state = np.zeros([9, 10, 9])
        _current_state[:7] = state_list2state_array(self.state_deque[-1]).transpose([2, 0, 1])
        if self.game_start:
            move = move_id2move_action[self.last_move]
            sy, sx = int(move[0]), int(move[1])
            ey, ex = int(move[2]), int(move[3])
            _current_state[7][sy][sx] = -1
            _current_state[7][ey][ex] = 1
        if self.action_count % 2 == 0:
            _current_state[8][:, :] = 1.0
        return _current_state.copy()

    def do_move(self, move: int) -> bool:
        try:
            self.game_start = True
            self.action_count += 1
            move_action = move_id2move_action[move]
            start_y, start_x = int(move_action[0]), int(move_action[1])
            end_y, end_x = int(move_action[2]), int(move_action[3])

            state_list = copy.deepcopy(self.state_deque[-1])

            # 必须应将
            if is_in_check(state_list, self.current_player_color):
                return False

            # 处理吃子逻辑
            target = state_list[end_y][end_x]
            if target != '一一':
                self.kill_action = 0
                if target == '红帅' and self.current_player_color == '黑':
                    self.winner = self.color2id['黑']
                elif target == '黑帅' and self.current_player_color == '红':
                    self.winner = self.color2id['红']
            else:
                self.kill_action += 1

            next_state = change_state(state_list, move_action)

            if is_in_check(next_state, self.current_player_color):
                return False
            if is_king_face_to_face(next_state):
                return False
            if is_both_check(next_state):
                return False

            self.state_list = next_state
            self.current_player_color = '黑' if self.current_player_color == '红' else '红'
            self.current_player_id = 1 if self.current_player_id == 2 else 2
            self.last_move = move
            self.state_deque.append(copy.deepcopy(self.state_list))
            self.move_history.append(move)
            return True

        except Exception as e:
            print(f"[警告] 动作 {move} 非法: {e}")
            return False

    def has_a_winner(self):
        if self.winner is not None:
            return True, self.winner
        if self.kill_action >= CONFIG['kill_action']:
            return True, self.backhand_player
        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        repetition_result = "normal"
        if self.enable_complex_repetition:
            repetition_result = check_complex_repetition(self.state_deque, self.move_history)

        if repetition_result in ["long_check", "long_capture", "long_chase"]:
            return True, self.backhand_player
        if win:
            return True, winner
        if self.kill_action >= CONFIG['kill_action']:
            return True, -1
        if is_suffocated(self.state_deque[-1], self.current_player_color):
            return True, self.backhand_player
        return False, -1

    def get_current_player_color(self):
        return self.current_player_color

    def get_current_player_id(self):
        return self.current_player_id

    def get_piece_legal_moves(self, y, x):
        return get_piece_legal_moves(self.state_list, y, x, self.current_player_color)


def get_piece_legal_moves(state_list, y, x, player_color='红'):
    piece = state_list[y][x]
    if piece == '一一':
        return []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    legal_moves = []

    if piece == f'{player_color}车':
        for dy, dx in directions:
            ny, nx = y, x
            while True:
                ny += dy
                nx += dx
                if not check_bounds(ny, nx):
                    break
                target = state_list[ny][nx]
                if target == '一一':
                    legal_moves.append(f"{y}{x}{ny}{nx}")
                else:
                    if check_obstruct(target, player_color):
                        legal_moves.append(f"{y}{x}{ny}{nx}")
                    break

    elif piece == f'{player_color}马':
        knight_moves = [(-2, -1), (-1, -2), (-2, 1), (1, -2),
                        (2, -1), (-1, 2), (2, 1), (1, 2)]
        for dy, dx in knight_moves:
            ny, nx = y + dy, x + dx
            if not check_bounds(ny, nx):
                continue
            leg_y, leg_x = y + dy // 2, x + dx // 2
            if state_list[leg_y][leg_x] != '一一':
                continue
            if check_obstruct(state_list[ny][nx], player_color):
                legal_moves.append(f"{y}{x}{ny}{nx}")

    elif piece == f'{player_color}炮':
        for dy, dx in directions:
            ny, nx = y, x
            hits = False
            while True:
                ny += dy
                nx += dx
                if not check_bounds(ny, nx):
                    break
                target = state_list[ny][nx]
                if not hits:
                    if target == '一一':
                        legal_moves.append(f"{y}{x}{ny}{nx}")
                    else:
                        hits = True
                else:
                    if target != '一一' and check_obstruct(target, player_color):
                        legal_moves.append(f"{y}{x}{ny}{nx}")
                    break

    elif piece == f'{player_color}兵':
        forward = -1 if player_color == '黑' else 1
        moves = [(forward, 0)]
        if (player_color == '红' and y <= 4) or (player_color == '黑' and y >= 5):
            moves.extend([(0, 1), (0, -1)])
        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if check_bounds(ny, nx) and check_obstruct(state_list[ny][nx], player_color):
                legal_moves.append(f"{y}{x}{ny}{nx}")

    elif piece == f'{player_color}士':
        for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = y + dy, x + dx
            if ((player_color == '红' and 0 <= ny <= 2 and 3 <= nx <= 5) or
                (player_color == '黑' and 7 <= ny <= 9 and 3 <= nx <= 5)):
                if check_obstruct(state_list[ny][nx], player_color):
                    legal_moves.append(f"{y}{x}{ny}{nx}")

    elif piece == f'{player_color}象':
        elephant_moves = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
        for dy, dx in elephant_moves:
            ny, nx = y + dy, x + dx
            if not check_bounds(ny, nx):
                continue
            eye_y, eye_x = y + dy // 2, x + dx // 2
            if state_list[eye_y][eye_x] != '一一':
                continue
            if check_obstruct(state_list[ny][nx], player_color):
                if (player_color == '红' and ny <= 4) or (player_color == '黑' and ny >= 5):
                    legal_moves.append(f"{y}{x}{ny}{nx}")

    elif piece == f'{player_color}帅':
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if check_bounds(ny, nx) and check_obstruct(state_list[ny][nx], player_color):
                if ((player_color == '红' and 0 <= ny <= 2 and 3 <= nx <= 5) or
                    (player_color == '黑' and 7 <= ny <= 9 and 3 <= nx <= 5)):
                    legal_moves.append(f"{y}{x}{ny}{nx}")

    return legal_moves


def get_legal_moves(state_deque: deque, current_player_color: str) -> List[int]:
    state_list = state_deque[-1]
    old_state_list = state_deque[-4] if len(state_deque) >= 4 else state_list

    moves = []
    for y in range(10):
        for x in range(9):
            piece = state_list[y][x]
            if not piece.startswith(current_player_color):
                continue

            legal_moves = get_piece_legal_moves(state_list, y, x, current_player_color)
            for m in legal_moves:
                try:
                    next_state = change_state(state_list, m)
                    if not is_in_check(next_state, current_player_color) and next_state != old_state_list:
                        moves.append(m)
                except IllegalMoveError:
                    continue

    return [move_action2move_id[m] for m in moves if m in move_action2move_id]


def check_complex_repetition(state_deque, move_history):
    states = list(state_deque)
    if len(states) < 4:
        return "normal"

    if all(str(s) == str(states[-1]) for s in states[:-1]):
        return "repetition_draw"

    moves = []
    for m in move_history[-(len(states) - 1):]:
        move_action = move_id2move_action.get(m, None)
        if move_action is None:
            return "normal"
        moves.append(move_action)

    action_types = []
    for i in range(len(moves)):
        move_str = moves[i]
        sy, sx = int(move_str[0]), int(move_str[1])
        ey, ex = int(move_str[2]), int(move_str[3])
        target = states[i][ey][ex]
        if target != '一一':
            action_types.append("capture")
        elif is_in_check(states[i + 1], '红' if i % 2 == 0 else '黑'):
            action_types.append("check")
        else:
            action_types.append("idle")

    if len(action_types) >= 6:
        pattern = ''.join(['C' if t == 'check' else 'X' if t == 'capture' else 'I' for t in action_types[-6:]])
        if pattern == 'CCCCCC':
            return "long_check"
        elif pattern == 'XXXXXX':
            return "long_capture"
        elif pattern.startswith('CX') and pattern[::2] == 'CCC' and pattern[1::2] == 'XXX':
            return "long_chase"
        elif pattern.startswith('XC') and pattern[::2] == 'XXX' and pattern[1::2] == 'CCC':
            return "long_chase"

    return "normal"


class Game:
    def __init__(self, board):
        self.board = board

    def graphic(self, board, p1, p2):
        print('player1 take:', p1)
        print('player2 take:', p2)
        print_board(state_list2state_array(board.state_deque[-1]))

    def start_play(self, player1, player2, start_player=1, is_shown=1):
        self.board.init_board(start_player)
        players = {1: player1, 2: player2}
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current = self.board.get_current_player_id()
            move = players[current].get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    print("Game end. Winner is", players[winner])
                else:
                    print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=False, temp=1e-3, logger=None):
        self.board.init_board()
        player.reset_player()
        states, mcts_probs, current_players = [], [], []
        _count = 0

        while True:
            start_time = time.time()
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            success = self.board.do_move(move)
            if not success:
                continue
            result_msg = f'第{_count + 1}步，走一步要花: {time.time() - start_time}'
            if logger and _count % 20 == 0:
                logger.info(result_msg)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            _count += 1

            end, winner = self.board.game_end()
            if end:
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                else:
                    winner_z[:] = 0.0
                player.reset_player()

                if is_shown:
                    result = "平局" if winner == -1 else f"玩家 {winner} 获胜"
                    print(result)
                    if logger:
                        logger.info(result)

                return winner, zip(states, mcts_probs, winner_z)
