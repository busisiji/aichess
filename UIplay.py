import pygame
import sys
import copy
from game import move_action2move_id, Game, Board
from mcts import MCTSPlayer
import time
from config import CONFIG
import threading

# 动态导入框架和模型
if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')
    exit()

class Human:
    def __init__(self):
        self.agent = 'HUMAN'

    def get_action(self, move):
        return move_action2move_id.get(move, -1)

    def set_player_ind(self, p):
        self.player = p


# 判断设备是否支持 GPU
def get_device(use_gpu=True):
    if CONFIG['use_frame'] == 'paddle':
        import paddle
        return 'gpu' if use_gpu and paddle.is_compiled_with_cuda() else 'cpu'
    elif CONFIG['use_frame'] == 'pytorch':
        import torch
        return 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    return 'cpu'


# 加载神经网络模型（GPU）
device = get_device(use_gpu=True)
print(f"正在使用设备：{device}")

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
    policy_value_net = PolicyValueNet(model_file=CONFIG['paddle_model_path'], device=device)
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
    policy_value_net = PolicyValueNet(model_file=CONFIG['pytorch_model_path'], device=device)


# 初始化pygame
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load('bgm/yinzi.ogg')
pygame.mixer.music.set_volume(0.03)
pygame.mixer.music.play(loops=-1, start=0.0)

size = width, height = 700, 700

# 先设置窗口
screen = pygame.display.set_mode(size)
pygame.display.set_caption("中国象棋")

# ✅ 在这里定义 clock
clock = pygame.time.Clock()

# 再加载图片资源
bg_image = pygame.image.load('imgs/board.png').convert_alpha()
bg_image = pygame.transform.smoothscale(bg_image, size)

fire_image = pygame.transform.smoothscale(pygame.image.load("imgs/fire.png").convert_alpha(), (width // 10, height // 10))
fire_image.set_alpha(200)
fire_rect = fire_image.get_rect()



# 棋盘初始化
board_list_init = [
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

# 图像资源加载
str2image = {
    piece: pygame.transform.smoothscale(pygame.image.load(f"imgs/{img}.png").convert_alpha(), (width // 10 - 10, height // 10 - 10))
    for piece, img in {
        '红车': 'hongche',
        '红马': 'hongma',
        '红象': 'hongxiang',
        '红士': 'hongshi',
        '红帅': 'hongshuai',
        '红炮': 'hongpao',
        '红兵': 'hongbing',
        '黑车': 'heiche',
        '黑马': 'heima',
        '黑象': 'heixiang',
        '黑士': 'heishi',
        '黑帅': 'heishuai',
        '黑炮': 'heipao',
        '黑兵': 'heibing'
    }.items()
}

str2image_rect = {k: v.get_rect() for k, v in str2image.items()}


# 绘制棋盘图像
x_ratio, y_ratio = 80, 72
x_bais, y_bais = 30, 25

def board2image(board):
    images = []
    for i in range(10):
        for j in range(9):
            piece = board[i][j]
            if piece != '一一' and piece in str2image:
                rect = str2image_rect[piece].copy()
                rect.center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                images.append((str2image[piece], rect))
    return images


# 加载火焰选中效果
fire_image = pygame.transform.smoothscale(pygame.image.load("imgs/fire.png").convert_alpha(), (width // 10, height // 10))
fire_image.set_alpha(200)
fire_rect = fire_image.get_rect()


# 初始化游戏
board = Board()
start_player = 1

# 创建 AI 玩家，降低模拟次数提升响应速度
player1 = MCTSPlayer(policy_value_net.policy_value_fn,
                     c_puct=5,
                     n_playout=1000,
                     is_selfplay=0)
player2 = MCTSPlayer(policy_value_net.policy_value_fn,
                     c_puct=5,
                     n_playout=2000,
                     is_selfplay=0)

board.init_board(start_player)
p1, p2 = 1, 2
player1.set_player_ind(p1)
player2.set_player_ind(p2)
players = {p1: player1, p2: player2}


# 异步线程执行 MCTS 避免阻塞 UI
class AIThread(threading.Thread):
    def __init__(self, player, board):
        super().__init__()
        self.player = player
        self.board = board
        self.move = None

    def run(self):
        self.move = self.player.get_action(self.board)


# 游戏主循环变量
switch_player = True
draw_fire = False
move_action = ''
first_button = False
start_i_j = None

while True:
    screen.blit(bg_image, (0, 0))

    # 更新棋盘
    for image, rect in board2image(board.state_deque[-1]):
        screen.blit(image, rect)

    # 显示被选中的位置
    if draw_fire:
        screen.blit(fire_image, fire_rect)

    # 更新界面
    pygame.display.update()
    clock.tick(60)  # 不高于60帧

    # 处理鼠标事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if not first_button:
                for i in range(10):
                    for j in range(9):
                        if abs(j * x_ratio + x_bais - mouse_x) < 30 and abs(i * y_ratio + y_bais - mouse_y) < 30:
                            first_button = True
                            start_i_j = (j, i)
                            fire_rect.center = (start_i_j[0] * x_ratio + x_bais, start_i_j[1] * y_ratio + y_bais)
                            break
                    if first_button:
                        break
            else:
                for i in range(10):
                    for j in range(9):
                        if abs(j * x_ratio + x_bais - mouse_x) < 30 and abs(i * y_ratio + y_bais - mouse_y) < 30:
                            first_button = False
                            end_i_j = (j, i)
                            move_action = f"{start_i_j[1]}{start_i_j[0]}{end_i_j[1]}{end_i_j[0]}"
                            break
                    if not first_button:
                        break

    # 切换玩家
    if switch_player:
        current_player = board.get_current_player_id()
        player_in_turn = players[current_player]

    # AI 回合处理
    if player_in_turn.agent == 'AI':
        ai_thread = AIThread(player_in_turn, board)
        ai_thread.start()
        while ai_thread.is_alive():
            clock.tick(60)  # 主线程等待时保持渲染更新
        move = ai_thread.move
        board.do_move(move)
        switch_player = True
        draw_fire = False
        print('耗时：', time.time() - time.time())  # 可根据实际逻辑补充时间记录
    elif player_in_turn.agent == 'HUMAN':
        draw_fire = True
        switch_player = False
        if len(move_action) == 4:
            move = player_in_turn.get_action(move_action)
            if move != -1:
                board.do_move(move)
                switch_player = True
                move_action = ''
                draw_fire = False

    # 检查游戏结束
    end, winner = board.game_end()
    if end:
        if winner != -1:
            print("Game end. Winner is", players[winner])
        else:
            print("Game end. Tie")
        sys.exit()
