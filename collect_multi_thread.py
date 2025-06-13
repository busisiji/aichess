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
from zip_array import zip_array_fast, recovery_array_fast,compress_game_data,decompress_game_data

# 从已有模块导入
from game import Board, Game, move_id2move_action, move_action2move_id, flip_map
from mcts import MCTSPlayer
from config import CONFIG

# Redis 支持
if CONFIG['use_redis']:
    import redis

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
        self.nums = [] #  记录每局步数 * 2（包括镜像）
        self.use_compression = CONFIG.get('use_data_compression', False)  # 默认不启用


        if CONFIG['use_redis']:
            self.redis_client = redis.Redis(
                host=CONFIG['redis_host'],
                port=CONFIG['redis_port'],
                db=CONFIG['redis_db']
            )
        else:
            self.data_buffer = deque(maxlen=self.buffer_size)
            self.temp_play_data = []

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

        # 尝试从磁盘/Redis 恢复数据
        self.load_data_buffer()

    def load_data_buffer(self):
        data_key = f"data/data_buffer_process_{self.process_id}"
        if CONFIG['use_redis']:
            meta = self.redis_client.get(data_key)
            if meta:
                try:
                    meta_dict = pickle.loads(meta)
                    self.data_buffer = deque(meta_dict['data_buffer'], maxlen=self.buffer_size)
                    self.iters = meta_dict.get('iters', 0)
                    self.nums = meta_dict.get('nums', [])
                    logger.info(f"[进程 {self.process_id}] 从 Redis 恢复数据，迭代次数: {self.iters}")
                except Exception as e:
                    logger.error(f"[进程 {self.process_id}] Redis 数据加载失败: {e}")
        else:
            filename = f"{data_key}.pkl"
            if os.path.exists(filename):
                try:
                    with open(filename, 'rb') as f:
                        data_dict = pickle.load(f)

                    # 如果启用了压缩，则对数据进行解压
                    loaded_data = data_dict.get('data_buffer', [])
                    if self.use_compression and loaded_data and isinstance(loaded_data[0], bytes):
                        decompressed_data = []
                        for item in loaded_data:
                            decompressed = decompress_game_data(item)
                            decompressed_data.extend(decompressed)
                        loaded_data = decompressed_data

                    self.data_buffer = deque(loaded_data, maxlen=self.buffer_size)
                    self.iters = data_dict.get('iters', 0)
                    self.nums = data_dict.get('nums', [])
                    logger.info(f"[进程 {self.process_id}] 从本地文件 {filename} 恢复数据，迭代次数: {self.iters}")
                except Exception as e:
                    logger.error(f"[进程 {self.process_id}] 文件 {filename} 加载失败: {e}")


    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            # 原始数据
            extend_data.append((state, mcts_prob, winner))

            # 水平翻转后的数据
            state = state.transpose([1, 2, 0])  # CHW -> HWC
            state_flip = np.zeros_like(state)
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

            # 先暂存在临时变量中
            self.temp_play_data = extended_data


            if CONFIG['use_redis']:
                data_key = f'train_data:{self.process_id}'
                data_dict = {
                    'data_buffer': self.temp_play_data,
                    'iters': self.iters + 1,
                    'nums': self.nums
                }
                try:
                    self.redis_client.setex(data_key, 3600, pickle.dumps(data_dict))
                    logger.info(f"[进程 {self.process_id}] 已写入 Redis: {data_key}")
                except Exception as e:
                    logger.error(f"[进程 {self.process_id}] Redis 写入失败: {e}")
            else:
                # 仅在完整局结束后才写入主 buffer
                self.data_buffer.extend(self.temp_play_data)
                self.iters += 1
                self.nums.append(episode_len*2)

            self.save_to_disk()

            logger.info(f"[进程 {self.process_id}] 第 {self.iters} 局结束，总步数: {episode_len}, 胜者: {winner}")

        except Exception as e:
            logger.error(f"[进程 {self.process_id}] 对局中断或出错: {e}")
            print(f"[进程 {self.process_id}] ⚠️ 对局中断，放弃当前未完成的数据")
            self.temp_play_data.clear()  # 清除临时数据


    def save_to_disk(self):
        if not CONFIG['use_redis']:
            data_key = f"data/data_buffer_process_{self.process_id}"
            filename = f"{data_key}.pkl"
            temp_filename = f"{filename}.tmp"

            # 如果启用了压缩，则对数据进行压缩
            if self.use_compression:
                compressed_data_list = []
                for item in self.data_buffer:
                    compressed_data = compress_game_data([item])  # 单条数据包装成列表
                    compressed_data_list.extend(compressed_data)
                data_to_save = {
                    'data_buffer': compressed_data_list,
                    'iters': self.iters,
                    'nums': self.nums,
                }
            else:
                data_to_save = {
                    'data_buffer': list(self.data_buffer),
                    'iters': self.iters,
                    'nums': self.nums,
                }

            try:
                with open(temp_filename, 'wb') as f:
                    pickle.dump(data_to_save, f)
                os.replace(temp_filename, filename)
            except Exception as e:
                logger.error(f"[进程 {self.process_id}] 保存文件失败: {e}")
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)



# 主进程合并
def merge_and_cleanup_data_buffers(output_path, num_processes, buffer_size=100000):
    merged_buffer = deque(maxlen=buffer_size)
    total_iters = 0
    skipped_episodes = 0
    step_nums_unique = []

    # Step 1: 加载已有主数据，并提取其 episode 哈希值用于后续对比
    existing_episodes = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'rb') as f:
                existing_data = pickle.load(f)
            for item in existing_data.get('data_buffer', []):
                merged_buffer.append(item)

            # 提取历史数据中的 episodes 并哈希存储
            play_index = 0
            play_data = existing_data.get('data_buffer', [])
            step_nums = existing_data.get('nums', [])

            for step_num in step_nums:
                episode_data = list(play_data)[play_index: play_index + step_num]
                if episode_data:
                    hashable = tuple((tuple(s.flatten()), tuple(mp), w) for s, mp, w in episode_data)
                    existing_episodes.add(hashable)
                play_index += step_num

            logger.info(f"✅ 已加载历史数据 {len(existing_data.get('data_buffer', []))} 条")
        except Exception as e:
            logger.error(f"❌ 加载历史数据失败: {e}")

    # Step 2: 收集所有子进程数据到一个全局缓冲区
    global_play_data = []
    global_step_nums = []
    total_iters = 0

    if CONFIG['use_redis']:
        redis_client = redis.Redis(
            host=CONFIG['redis_host'],
            port=CONFIG['redis_port'],
            db=CONFIG['redis_db']
        )

        for pid in range(num_processes):
            data_key = f'train_data:{pid}'
            item = redis_client.get(data_key)
            if item:
                try:
                    data_dict = pickle.loads(item)
                    play_data = data_dict.get('data_buffer', [])
                    step_nums = data_dict.get('nums', [])
                    pid_iters = data_dict.get('iters', 0)

                    global_play_data.extend(play_data)
                    global_step_nums.extend(step_nums)
                    total_iters += pid_iters

                    redis_client.delete(data_key)
                    logger.info(f"[进程 {pid}] 数据已从 Redis 取出")
                except Exception as e:
                    logger.error(f"❌ Redis 数据反序列化失败: {e}")
    else:
        for pid in range(num_processes):
            filename = f"data/data_buffer_process_{pid}.pkl"
            if not os.path.exists(filename):
                continue
            try:
                with open(filename, 'rb') as f:
                    data_dict = pickle.load(f)

                play_data = data_dict.get('data_buffer', [])
                step_nums = data_dict.get('nums', [])
                pid_iters = data_dict.get('iters', 0)

                # 如果是压缩数据，则解压
                if CONFIG.get('use_data_compression', False) :
                    decompressed = decompress_game_data(play_data)

                global_play_data.extend(decompressed)
                global_step_nums.extend(step_nums)
                total_iters += pid_iters

                logger.info(f"[进程 {pid}] 数据已从本地取出")
            except Exception as e:
                logger.error(f"❌ 合并 {filename} 时出错: {str(e)}")


    # Step 3: 根据 step_nums 划分 episode
    episodes = []
    play_index = 0
    for step_num in global_step_nums:
        episode_data = list(global_play_data)[play_index: play_index + step_num]
        if episode_data:
            episodes.append(episode_data)
        play_index += step_num

    # Step 4: 子数据集内部去重
    seen_episodes = set()
    unique_episodes_from_processes = []

    for ep_idx, episode in enumerate(episodes):
        hashable_episode = tuple(
            (tuple(state.flatten()), tuple(mcts_prob), winner)
            for state, mcts_prob, winner in episode
        )
        if hashable_episode not in seen_episodes:
            seen_episodes.add(hashable_episode)
            unique_episodes_from_processes.append(episode)
            step_nums_unique.append(global_step_nums[ep_idx])
        else:
            skipped_episodes += 1

    logger.info(f"📌 子数据集内部去重完成，共保留 {len(unique_episodes_from_processes)} 局")

    # Step 5: 与已有主数据进行比较，排除重复的局
    final_unique_episodes = []

    for episode in unique_episodes_from_processes:
        hashable = tuple(
            (tuple(state.flatten()), tuple(mcts_prob), winner)
            for state, mcts_prob, winner in episode
        )
        if hashable not in existing_episodes:
            final_unique_episodes.append(episode)
        else:
            skipped_episodes += 1

    logger.info(f"📌 与主数据对比后，共保留 {len(final_unique_episodes)} 局新数据")

    # Step 6: 写入最终数据
    for episode in final_unique_episodes:
        # 压缩数据
        compressed_data = compress_game_data(episode)
        merged_buffer.extend(compressed_data)

    # 构造输出结构
    merged_data = {
        'data_buffer': list(merged_buffer),
        'iters': total_iters,
        # 'nums': step_nums_unique,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(merged_data, f)

    logger.info(f"✅ 所有进程数据已追加合并至 {output_path}")
    logger.info(f"🚫 共跳过 {skipped_episodes} 局重复数据")



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
        print("✅ 模型加载成功：",  MODEL_PATH)
    except Exception as e:
        print("❌ 模型加载失败，尝试初始化新模型")
        policy_value_net = PolicyValueNet()
        print("✅ 成功初始化空白模型")

    # 判断是否使用GPU
    device = 'cuda' if torch.cuda.is_available() and CONFIG['use_frame'] == 'pytorch' else 'cpu'
    print(f"🎮 使用设备: {device.upper()}")

    # 先合并历史数据
    OUTPUT_PATH = CONFIG['train_data_buffer_path']

    # 使用 spawn 方式启动多进程（适用于 GPU）
    ctx = multiprocessing.get_context('spawn')
    Process = ctx.Process

    processes = []
    try:
        for pid in range(NUM_PROCESSES):
            p = Process(target=run_pipeline, args=(pid, policy_value_net))  # 只传两个参数
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("🛑 主进程收到中断信号，开始合并数据...")
        merge_and_cleanup_data_buffers(OUTPUT_PATH, NUM_PROCESSES)
        print("✅ 数据合并完成")
