# -*- coding: utf-8 -*-

import multiprocessing
import time
import logging
import os
from collections import deque
import pickle
import threading

import numpy as np
import torch
from mpmath import mp

from pytorch_net import PolicyValueNet
from zip_array import compress_game_data, decompress_game_data

# 从已有模块导入
from game import Board, Game, move_id2move_action, move_action2move_id, flip_map
from mcts import MCTSPlayer
from config import CONFIG

# 使用统一数据服务
from data_service import DataManagementService

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 日志设置
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MergeData")

# 全局锁用于打印同步
print_lock = multiprocessing.Lock()

# 模型路径
MODEL_PATH = CONFIG['pytorch_model_path']


# 自定义 CollectPipeline 类（进程安全）
class ProcessSafeCollectPipeline:

    def __init__(self, model, process_id):
        self.process_id = process_id
        self.board = Board()
        self.game = Game(self.board)
        self.temp = 1e-3
        self.n_playout = CONFIG.get('play_out', 400)  # 默认值 400
        self.c_puct = CONFIG.get('c_puct', 5)
        self.buffer_size = CONFIG.get('buffer_size', 100000)
        self.iters = 0
        self.temp_play_data = []  # 临时缓存未完成的对局数据
        self.nums = []  # 记录每局步数 * 2（包括镜像）

        # 初始化统一数据服务
        self.data_service = DataManagementService(CONFIG)
        self.use_compression = self.data_service.use_compression

        # 加载传入的共享模型
        self.policy_value_net = model
        if CONFIG['use_frame'] == 'pytorch' and torch.cuda.is_available():
            self.policy_value_net.device = torch.device('cuda')

        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1
        )

        # 尝试恢复本地数据
        self.data_buffer = self.data_service.load_initial_data() or []

    def get_equi_data(self, play_data):
        """数据增强：水平翻转"""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            extend_data.append((state, mcts_prob, winner))

            # 翻转数据
            state_flip = np.zeros_like(state)
            state = state.transpose([1, 2, 0])  # CHW -> HWC
            for i in range(10):
                for j in range(9):
                    state_flip[i][j] = state[i][8 - j]
            state_flip = state_flip.transpose([2, 0, 1])  # HWC -> CHW

            mcts_prob_flip = [0] * len(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                action = move_id2move_action[i]
                flipped_action = flip_map(action)
                flipped_id = move_action2move_id.get(flipped_action, None)
                if flipped_id is not None:
                    mcts_prob_flip[i] = mcts_prob[flipped_id]

            extend_data.append((state_flip, mcts_prob_flip, winner))
        return extend_data

    def run(self, logger=None):
        try:
            while True:
                self.collect_selfplay_data(logger=logger)
                time.sleep(1)  # 避免 CPU 占用过高
        except KeyboardInterrupt:
            logger.info(f"[进程 {self.process_id}] 收到中断信号，正在保存最终数据...")

    def collect_selfplay_data(self, logger=None):
        try:
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_shown=True, logger=logger)
            play_data = list(play_data)
            episode_len = len(play_data)

            # 数据增强
            extended_data = self.get_equi_data(play_data)

            # 使用服务接口写入数据
            self.data_service.write_play_data(self.process_id, extended_data, self.iters + 1, [episode_len*2])

            self.iters += 1
            logger.info(f"[进程 {self.process_id}] 第 {self.iters} 局结束，总步数: {episode_len}, 胜者: {winner}")

        except Exception as e:
            logger.error(f"[进程 {self.process_id}] 对局中断或出错: {e}")
            print(f"[进程 {self.process_id}] ⚠️ 对局中断，放弃当前未完成的数据")

    def save_to_disk(self):
        """保留此方法作为备用，实际由 data_service 处理"""
        pass


# 主进程合并
def merge_and_cleanup_data_buffers(output_path, num_processes, buffer_size=100000):
    """
    使用统一数据服务进行数据合并与去重
    """
    data_service = DataManagementService(CONFIG)
    merged_data = data_service.merge_all_data(output_path, num_processes)
    return merged_data


# 配置日志系统
def setup_logger(process_id):
    logger = logging.getLogger(f"selfplay-{process_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除旧 handler
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, f"selfplay_process_{process_id}.log"), encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# 多进程入口函数
def run_pipeline(process_id, shared_model):
    logger = setup_logger(process_id)
    logger.info(f"__________________________________________")
    logger.info(f"进程 {process_id} 启动...")

    pipeline = ProcessSafeCollectPipeline(model=shared_model, process_id=process_id)
    logger.info(f"[进程 {process_id}] 初始化完成，开始持续采集数据")

    pipeline.run(logger=logger)

    logger.info(f"进程 {process_id} 结束.")
    with print_lock:
        print(f"✅ 进程 {process_id} 已结束")


# 动态调整进程管理器
def dynamic_process_manager(shared_model, target_num_processes):
    active_processes = []

    while True:
        current_num = len(active_processes)
        desired_num = CONFIG.get('num_processes', 4)

        if desired_num > current_num:
            # 启动新进程
            for pid in range(current_num, desired_num):
                p = mp.Process(target=run_pipeline, args=(pid, shared_model))
                p.start()
                active_processes.append(p)
                print(f"➕ 新增进程 PID={pid}")
        elif desired_num < current_num:
            # 终止多余的进程
            for p in active_processes[desired_num:]:
                p.terminate()
                p.join()
            active_processes = active_processes[:desired_num]
            print(f"➖ 减少进程至 {desired_num}")

        time.sleep(10)  # 每隔10秒检查一次配置变化


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows 支持
    NUM_PROCESSES = CONFIG['num_processes']
    interval = CONFIG['interval']

    print("🔄 加载模型中...")
    try:
        policy_value_net = PolicyValueNet(model_file=MODEL_PATH)
        print("✅ 模型加载成功：", MODEL_PATH)
    except Exception as e:
        print("❌ 模型加载失败，尝试初始化新模型")
        policy_value_net = PolicyValueNet()
        print("✅ 成功初始化空白模型")

    # 判断是否使用GPU
    device = 'cuda' if torch.cuda.is_available() and CONFIG['use_frame'] == 'pytorch' else 'cpu'
    print(f"🎮 使用设备: {device.upper()}")

    # 启动动态采集器
    dynamic_process_manager(policy_value_net, NUM_PROCESSES)
