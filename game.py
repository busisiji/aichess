"""
棋盘游戏控制 - 完整封装版本
"""

import time
import numpy as np
import copy
from collections import deque
from typing import List, Tuple, Dict, Set, Optional, Union

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
    copy_list = copy.deepcopy(state_list)
    y, x, toy, tox = map(int, move)
    copy_list[toy][tox] = copy_list[y][x]
    copy_list[y][x] = '一一'
    if is_king_face_to_face(copy_list):
        raise IllegalMoveError("非法状态：将帅面对面且中间无子！")
    return copy_list


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
    _move_id2move_action = {}
    _move_action2move_id = {}
    row = [str(i) for i in range(9)]
    column = [str(i) for i in range(10)]

    # 士、象等走法
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
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and 0 <= l2 < 10 and 0 <= n2 < 9:
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


def flip_map(string: str) -> str:
    new_str = ''
    for index in range(4):
        if index == 0 or index == 2:
            new_str += string[index]
        else:
            new_str += str(8 - int(string[index]))
    return new_str


def check_bounds(toY: int, toX: int) -> bool:
    return 0 <= toY < 10 and 0 <= toX < 9


def check_obstruct(piece: str, current_player_color: str) -> bool:
    if piece == '一一':
        return True
    return (current_player_color == '红' and '黑' in piece) or (current_player_color == '黑' and '红' in piece)


def is_king_face_to_face(state_list: List[List[str]]) -> bool:
    k_pos = None
    K_pos = None
    for y in range(10):
        for x in range(9):
            if state_list[y][x] == '黑帅':
                k_pos = (y, x)
            elif state_list[y][x] == '红帅':
                K_pos = (y, x)

    if not k_pos or not K_pos:
        return False

    k_y, k_x = k_pos
    K_y, K_x = K_pos

    if k_x != K_x:
        return False

    min_y, max_y = sorted([k_y, K_y])
    if max_y - min_y <= 1:
        return True

    for y in range(min_y + 1, max_y):
        if state_list[y][k_x] != '一一':
            return False
    return True


def is_in_check(state_list: List[List[str]], player_color: str) -> bool:
    """
    判断指定玩家是否处于被将军状态（包含单将、双将、闷宫等情况）
    :param state_list: 当前棋盘状态
    :param player_color: 玩家颜色（'红' 或 '黑'）
    :return: True 表示被将军
    """
    king_pos = None
    for y in range(10):
        for x in range(9):
            if player_color == '红' and state_list[y][x] == '红帅':
                king_pos = (y, x)
            elif player_color == '黑' and state_list[y][x] == '黑帅':
                king_pos = (y, x)

    if not king_pos:
        return False

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


def is_stalemate(state_deque: deque, current_player_color: str) -> bool:
    moves = get_legal_moves(state_deque, current_player_color)
    return len(moves) == 0 or is_suffocated(state_deque[-1], current_player_color)

def is_suffocated(state_list: List[List[str]], player_color: str) -> bool:
    """
    判断是否处于"闷宫"状态：将帅被困无法移动但未被将军
    """
    king_pos = None
    for y in range(10):
        for x in range(9):
            if player_color == '红' and state_list[y][x] == '红帅':
                king_pos = (y, x)
            elif player_color == '黑' and state_list[y][x] == '黑帅':
                king_pos = (y, x)

    if not king_pos:
        return False

    # 获取将帅当前位置的所有合法移动
    legal_moves = get_piece_legal_moves(state_list, *king_pos, player_color)

    # 如果没有合法移动，且未被将军，则为闷宫
    return len(legal_moves) == 0 and not is_in_check(state_list, player_color)

def check_repetition_rules(state_deque: deque) -> str:
    from collections import Counter
    states = list(state_deque)
    count = Counter(str(states[i]) for i in range(len(states)))
    if len(count) == 1:
        last_move = move_id2move_action[state_deque.board.last_move]
        start_y, start_x = int(last_move[0]), int(last_move[1])
        piece = state_deque.board.state_deque[-2][start_y][start_x]
        target = state_deque.board.state_deque[-1][int(last_move[2])][int(last_move[3])]
        if target != '一一':
            return "long_capture"
        elif is_in_check(state_deque.board.state_deque[-1], state_deque.board.get_current_player_color()):
            return "long_check"
    elif len(count) == 2:
        return "repetition_draw"
    return "normal"
def check_complex_repetition(state_deque: deque) -> str:
    """
    检查复杂循环局面：
    - 一将一闲
    - 一将一捉
    - 长捉
    """
    states = list(state_deque)
    moves = [move_id2move_action[m] for m in state_deque.board.move_history[-len(states)+1:]]

    if len(states) < 4:
        return "normal"

    last_state = states[-1]
    prev_states = states[:-1]

    # 简单重复
    if all(str(s) == str(last_state) for s in prev_states):
        return "repetition_draw"

    # 复杂循环：比如一将一捉
    action_types = []
    for i in range(len(moves)):
        move_str = moves[i]
        sy, sx, ey, ex = map(int, move_str)
        piece = state_deque[i][sy][sx]
        target = state_deque[i][ey][ex]
        if target != '一一':
            action_types.append("capture")
        elif is_in_check(state_deque[i+1], '红' if i % 2 == 0 else '黑'):
            action_types.append("check")
        else:
            action_types.append("idle")

    # 示例：检查最近四步动作类型是否是 check-idle-check-idle
    if len(action_types) >= 4:
        pattern = ''.join([{'check': 'C', 'idle': 'I', 'capture': 'X'}[t] for t in action_types[-4:]])
        if pattern in ['CICD', 'CDCI', 'CCCC', 'XXXX']:
            return "complex_repetition"

    return "normal"


class IllegalMoveError(Exception):
    pass


class Board:
    def __init__(self):
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

    def init_board(self, start_player: int = 1):
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
    def availables(self) -> List[int]:
        return get_legal_moves(self.state_deque, self.current_player_color)

    def current_state(self) -> np.ndarray:
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

            if state_list[end_y][end_x] != '一一':
                self.kill_action = 0
                if self.current_player_color == '黑' and state_list[end_y][end_x] == '红帅':
                    self.winner = self.color2id['黑']
                elif self.current_player_color == '红' and state_list[end_y][end_x] == '黑帅':
                    self.winner = self.color2id['红']
            else:
                self.kill_action += 1

            next_state = change_state(state_list, move_action)
            if is_in_check(next_state, self.current_player_color):
                raise IllegalMoveError("不能让自己处于被将军状态！")

            state_list[end_y][end_x] = state_list[start_y][start_x]
            state_list[start_y][start_x] = '一一'
            self.current_player_color = '黑' if self.current_player_color == '红' else '红'
            self.current_player_id = 1 if self.current_player_id == 2 else 2
            self.last_move = move
            self.state_deque.append(state_list)
            self.move_history.append(move_action)
            return True
        except Exception as e:
            print(f"[错误] 动作 {move} 导致异常: {e}")
            return False

    def has_a_winner(self) -> Tuple[bool, int]:
        if self.winner is not None:
            return True, self.winner
        elif self.kill_action >= CONFIG['kill_action']:
            return True, self.backhand_player
        return False, -1

    def game_end(self) -> Tuple[bool, int]:
        win, winner = self.has_a_winner()
        if check_complex_repetition(self.state_deque) in ["long_check", "long_capture", "complex_repetition"]:
            return True, self.backhand_player
        if win:
            return True, winner
        elif self.kill_action >= CONFIG['kill_action']:
            return True, -1
        elif is_stalemate(self.state_deque, self.current_player_color):
            return True, -1
        return False, -1

    def get_current_player_color(self) -> str:
        return self.current_player_color

    def get_current_player_id(self) -> int:
        return self.current_player_id

    def get_piece_legal_moves(self, y: int, x: int) -> List[str]:
        return get_piece_legal_moves(self.state_list, y, x, self.current_player_color)


def get_piece_legal_moves(state_list: List[List[str]], y: int, x: int, player_color: str = '红') -> List[str]:
    piece = state_list[y][x]
    if piece == '一一':
        return []

    legal_moves = []

    if piece == f'{player_color}车':
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            while 0 <= ny < 10 and 0 <= nx < 9:
                target = state_list[ny][nx]
                if target == '一一':
                    legal_moves.append(f"{y}{x}{ny}{nx}")
                else:
                    if (player_color == '红' and '黑' in target) or (player_color == '黑' and '红' in target):
                        legal_moves.append(f"{y}{x}{ny}{nx}")
                    break
                ny += dy
                nx += dx

    elif piece == f'{player_color}马':
        knight_moves = [(-2, -1), (-1, -2), (-2, 1), (1, -2),
                        (2, -1), (-1, 2), (2, 1), (1, 2)]
        for dy, dx in knight_moves:
            ny, nx = y + dy, x + dx
            if not check_bounds(ny, nx):
                continue
            leg_y, leg_x = y + dy // 2, x + dx // 2
            if state_list[leg_y][leg_x] == '一一' and check_obstruct(state_list[ny][nx], player_color):
                legal_moves.append(f"{y}{x}{ny}{nx}")

    elif piece == f'{player_color}炮':
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dy, dx in directions:
            hits = False
            ny, nx = y + dy, x + dx
            while check_bounds(ny, nx):
                target = state_list[ny][nx]
                if not hits:
                    if target == '一一':
                        legal_moves.append(f"{y}{x}{ny}{nx}")
                    else:
                        hits = True
                else:
                    if target != '一一':
                        if (player_color == '红' and '黑' in target) or (player_color == '黑' and '红' in target):
                            legal_moves.append(f"{y}{x}{ny}{nx}")
                        break
                ny += dy
                nx += dx

    elif piece == f'{player_color}兵':
        forward = -1 if player_color == '黑' else 1
        moves = []
        if (player_color == '红' and y > 4) or (player_color == '黑' and y < 5):
            moves.extend([(forward, 0), (0, 1), (0, -1)])
        else:
            moves.append((forward, 0))
        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if check_bounds(ny, nx) and check_obstruct(state_list[ny][nx], player_color):
                legal_moves.append(f"{y}{x}{ny}{nx}")

    elif piece == f'{player_color}士':
        for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = y + dy, x + dx
            if player_color == '红' and 0 <= ny <= 2 and 3 <= nx <= 5:
                if check_obstruct(state_list[ny][nx], player_color):
                    legal_moves.append(f"{y}{x}{ny}{nx}")
            elif player_color == '黑' and 7 <= ny <= 9 and 3 <= nx <= 5:
                if check_obstruct(state_list[ny][nx], player_color):
                    legal_moves.append(f"{y}{x}{ny}{nx}")

    elif piece == f'{player_color}象':
        elephant_moves = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
        for dy, dx in elephant_moves:
            ny, nx = y + dy, x + dx
            if not check_bounds(ny, nx):
                continue
            leg_y, leg_x = y + dy // 2, x + dx // 2
            if state_list[leg_y][leg_x] == '一一' and check_obstruct(state_list[ny][nx], player_color):
                if (player_color == '红' and ny <= 4) or (player_color == '黑' and ny >= 5):
                    legal_moves.append(f"{y}{x}{ny}{nx}")

    elif piece == f'{player_color}帅':
        for dy, dx in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            ny, nx = y + dy, x + dx
            if player_color == '红' and 0 <= ny <= 2 and 3 <= nx <= 5:
                if check_obstruct(state_list[ny][nx], player_color):
                    legal_moves.append(f"{y}{x}{ny}{nx}")
            elif player_color == '黑' and 7 <= ny <= 9 and 3 <= nx <= 5:
                if check_obstruct(state_list[ny][nx], player_color):
                    legal_moves.append(f"{y}{x}{ny}{nx}")

    return legal_moves


def get_legal_moves(state_deque: deque, current_player_color: str) -> List[int]:
    state_list = state_deque[-1]
    old_state_list = state_deque[-4]
    moves = []

    for y in range(10):
        for x in range(9):
            piece = state_list[y][x]
            if piece == '一一':
                continue

            if not piece.startswith(current_player_color):
                continue

            legal_moves = get_piece_legal_moves(state_list, y, x, current_player_color)
            for m in legal_moves:
                if change_state(state_list, m) != old_state_list:
                    moves.append(m)

    for move in moves[:]:
        next_state = change_state(state_list, move)
        if is_king_face_to_face(next_state):
            moves.remove(move)

    moves_id = [move_action2move_id[m] for m in moves]
    return moves_id


class Game:
    def __init__(self, board: Board):
        self.board = board

    def graphic(self, board, player1_color, player2_color):
        print('player1 take: ', player1_color)
        print('player2 take: ', player2_color)
        print_board(state_list2state_array(board.state_deque[-1]))

    def start_play(self, player1, player2, start_player=1, is_shown=1):
        if start_player not in (1, 2):
            raise Exception('start_player should be either 1 or 2')
        self.board.init_board(start_player)
        p1, p2 = 1, 2
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current_player = self.board.get_current_player_id()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
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
        p1, p2 = 1, 2
        player.reset_player()
        states, mcts_probs, current_players = [], [], []
        _count = 0
        while True:
            if _count % 20 == 0:
                start_time = time.time()
                move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
                result_msg = f'第{_count + 1}步，走一步要花: {time.time() - start_time}'
                if logger:
                    logger.info(result_msg)
            else:
                move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            _count += 1
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            self.board.do_move(move)
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
                    result_msg = "🤝 平局！" if winner == -1 else f"🏆 玩家 {winner} 获胜！"
                    print(result_msg)
                    if logger:
                        logger.info(result_msg)
                return winner, zip(states, mcts_probs, winner_z)
