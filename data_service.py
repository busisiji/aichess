# data_service.py
import os
import pickle
from collections import deque
from datetime import timedelta
from zip_array import compress_game_data, decompress_game_data


class DataManagementService:
    def __init__(self, config):
        self.config = config
        self.use_redis = config.get('use_redis', False)
        self.use_compression = config.get('use_data_compression', False)
        self.train_data_path = config['train_data_buffer_path']
        self.checkpoint_path = config.get('checkpoint_path', 'checkpoints/latest.pkl')
        self.buffer_size = config.get('buffer_size', 100000)
        self.data_buffer = deque(maxlen=self.buffer_size)

        # 初始化 Redis 客户端
        self.redis_client = None
        if self.use_redis:
            import redis
            self.redis_client = redis.Redis(
                host=config['redis_host'],
                port=config['redis_port'],
                db=config['redis_db']
            )

        # 加载检查点（用于断点续训）
        self.checkpoint = self.load_checkpoint()

    def load_initial_data(self):
        """一次性加载初始训练数据"""
        if self.use_redis:
            return self._load_from_redis()
        else:
            return self._load_from_local_file()

    def refresh_data(self):
        """增量刷新数据，适用于持续训练模式"""
        if self.use_redis:
            return self._refresh_from_redis()
        else:
            return self._refresh_from_local_file()

    def _load_from_local_file(self):
        """从本地文件加载全部数据"""
        try:
            if not os.path.exists(self.train_data_path):
                print(f"⚠️ 数据文件不存在：{self.train_data_path}")
                return []

            with open(self.train_data_path, 'rb') as f:
                data_dict = pickle.load(f)

            raw_data = data_dict.get('data_buffer', [])
            if self.use_compression and raw_data and isinstance(raw_data[0], bytes):
                raw_data = decompress_game_data(raw_data)

            self.data_buffer.extend(raw_data)
            print(f"✅ 已加载 {len(raw_data)} 条数据（来自本地）")
            return raw_data
        except Exception as e:
            print(f"❌ 加载本地数据失败: {e}")
            return []

    def _load_from_redis(self):
        """从 Redis 加载所有数据"""
        try:
            all_data = []
            for key in self.redis_client.keys("train_data:*"):
                item = self.redis_client.get(key)
                if item:
                    data_dict = pickle.loads(item)
                    play_data = data_dict.get('data_buffer', [])
                    all_data.extend(play_data)

            self.data_buffer.extend(all_data)
            print(f"✅ 已加载 {len(all_data)} 条数据（来自 Redis）")
            return all_data
        except Exception as e:
            print(f"❌ Redis 数据加载失败: {e}")
            return []

    def _refresh_from_local_file(self):
        """增量更新：重新加载本地文件中新增的数据"""
        try:
            current_len = len(self.data_buffer)
            with open(self.train_data_path, 'rb') as f:
                data_dict = pickle.load(f)

            raw_data = data_dict.get('data_buffer', [])
            if self.use_compression and raw_data and isinstance(raw_data[0], bytes):
                raw_data = decompress_game_data(raw_data)

            new_data = raw_data[current_len:]
            self.data_buffer.extend(new_data)
            print(f"🔄 新增加载 {len(new_data)} 条数据（本地增量更新）")
            return new_data
        except Exception as e:
            print(f"❌ 增量加载本地数据失败: {e}")
            return []

    def _refresh_from_redis(self):
        """增量更新：从 Redis 获取新写入的数据"""
        try:
            current_len = len(self.data_buffer)
            new_data = []

            for key in self.redis_client.keys("train_data:*"):
                item = self.redis_client.get(key)
                if item:
                    data_dict = pickle.loads(item)
                    play_data = data_dict.get('data_buffer', [])
                    new_data.extend(play_data)

            existing_set = set(tuple(d) for d in self.data_buffer)
            filtered_new = [d for d in new_data if tuple(d) not in existing_set]
            self.data_buffer.extend(filtered_new)
            print(f"🔄 新增加载 {len(filtered_new)} 条数据（Redis增量更新）")
            return filtered_new
        except Exception as e:
            print(f"❌ Redis 增量加载失败: {e}")
            return []

    def save_checkpoint(self, iters, model_path, extra_info=None):
        """保存当前训练状态到 checkpoint 文件"""
        checkpoint = {
            'iters': iters,
            'model_path': model_path,
            'timestamp': time.time(),
            'nums': list(self.checkpoint.get('nums', [])),
            'extra': extra_info or {}
        }

        try:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"💾 检查点已保存至: {self.checkpoint_path}")
        except Exception as e:
            print(f"❌ 保存检查点失败: {e}")

    def load_checkpoint(self):
        """从 checkpoint 文件恢复训练状态"""
        if not os.path.exists(self.checkpoint_path):
            print("🆕 未找到检查点文件，使用默认状态")
            return {'iters': 0, 'nums': [], 'model_path': None}

        try:
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"🔁 已恢复检查点，迭代次数: {checkpoint.get('iters', 0)}")
            return checkpoint
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            return {'iters': 0, 'nums': [], 'model_path': None}

    def get_train_data(self):
        """获取当前训练数据"""
        return list(self.data_buffer)

    def write_play_data(self, process_id, play_data, iters, nums):
        """写入对局数据（支持 Redis 或本地）"""
        data_key = f'train_data:{process_id}'
        data_to_save = {
            'data_buffer': play_data,
            'iters': iters,
            'nums': nums
        }

        if self.use_compression:
            compressed_data = [compress_game_data([item]) for item in play_data]
            data_to_save['data_buffer'] = compressed_data

        if self.use_redis:
            try:
                self.redis_client.setex(data_key, 3600, pickle.dumps(data_to_save))
                print(f"[进程 {process_id}] 已写入 Redis: {data_key}")
            except Exception as e:
                print(f"[进程 {process_id}] Redis 写入失败: {e}")
        else:
            filename = f"data/data_buffer_process_{process_id}.pkl"
            try:
                with open(filename + ".tmp", "wb") as f:
                    pickle.dump(data_to_save, f)
                os.replace(filename + ".tmp", filename)
                print(f"[进程 {process_id}] 已写入本地文件: {filename}")
            except Exception as e:
                print(f"[进程 {process_id}] 本地写入失败: {e}")

    def merge_all_data(self, output_path, num_processes):
        """合并所有采集进程数据到主缓冲区"""
        merged_buffer = deque(maxlen=self.buffer_size)
        total_iters = 0
        skipped_episodes = 0
        step_nums_unique = []

        existing_episodes = self._load_existing_episodes(output_path)

        global_play_data, global_step_nums, total_iters = self._collect_global_data(num_processes)

        episodes = self._split_into_episodes(global_play_data, global_step_nums)
        unique_episodes = self._deduplicate(episodes)
        final_unique = self._filter_against_existing(unique_episodes, existing_episodes)

        for episode in final_unique:
            compressed = compress_game_data(episode)
            merged_buffer.extend(compressed)
            step_nums_unique.append(len(episode))

        merged_data = {
            'data_buffer': list(merged_buffer),
            'iters': total_iters,
            'nums': step_nums_unique
        }

        try:
            with open(output_path, 'wb') as f:
                pickle.dump(merged_data, f)
            print(f"✅ 所有进程数据已合并至 {output_path}")
        except Exception as e:
            print(f"❌ 合并数据失败: {e}")

        return merged_data

    def _load_existing_episodes(self, path):
        if not os.path.exists(path):
            return set()

        try:
            with open(path, 'rb') as f:
                existing_data = pickle.load(f)
            existing_episodes = set()
            play_index = 0
            step_nums = existing_data.get('nums', [])
            play_data = existing_data.get('data_buffer', [])

            for step_num in step_nums:
                episode_data = play_data[play_index: play_index + step_num]
                if episode_data:
                    hashable = tuple((tuple(s.flatten()), tuple(mp), w) for s, mp, w in episode_data)
                    existing_episodes.add(hashable)
                play_index += step_num
            return existing_episodes
        except Exception as e:
            print(f"❌ 加载历史数据失败: {e}")
            return set()

    def _collect_global_data(self, num_processes):
        global_play_data = []
        global_step_nums = []
        total_iters = 0

        if self.use_redis:
            for pid in range(num_processes):
                data_key = f'train_data:{pid}'
                item = self.redis_client.get(data_key)
                if item:
                    try:
                        data_dict = pickle.loads(item)
                        play_data = data_dict.get('data_buffer', [])
                        step_nums = data_dict.get('nums', [])
                        pid_iters = data_dict.get('iters', 0)

                        global_play_data.extend(play_data)
                        global_step_nums.extend(step_nums)
                        total_iters += pid_iters
                        self.redis_client.delete(data_key)
                        print(f"[进程 {pid}] 数据已从 Redis 取出")
                    except Exception as e:
                        print(f"❌ Redis 数据反序列化失败: {e}")
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

                    if self.use_compression and play_data and isinstance(play_data[0], bytes):
                        play_data = decompress_game_data(play_data)

                    global_play_data.extend(play_data)
                    global_step_nums.extend(step_nums)
                    total_iters += pid_iters
                    print(f"[进程 {pid}] 数据已从本地取出")
                except Exception as e:
                    print(f"❌ 合并 {filename} 时出错: {str(e)}")

        return global_play_data, global_step_nums, total_iters

    def _split_into_episodes(self, play_data, step_nums):
        episodes = []
        play_index = 0
        for step_num in step_nums:
            episodes.append(play_data[play_index: play_index + step_num])
            play_index += step_num
        return episodes

    def _deduplicate(self, episodes):
        seen = set()
        unique = []
        for ep in episodes:
            hashable = tuple((tuple(s.flatten()), tuple(mp), w) for s, mp, w in ep)
            if hashable not in seen:
                seen.add(hashable)
                unique.append(ep)
        print(f"📌 子数据集内部去重完成，共保留 {len(unique)} 局")
        return unique

    def _filter_against_existing(self, episodes, existing_episodes):
        final_unique = []
        for ep in episodes:
            hashable = tuple((tuple(s.flatten()), tuple(mp), w) for s, mp, w in ep)
            if hashable not in existing_episodes:
                final_unique.append(ep)
        print(f"📌 与主数据对比后，共保留 {len(final_unique)} 局新数据")
        return final_unique
