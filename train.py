import os
import random
import time

import numpy as np
import pickle
from collections import deque, defaultdict

from config import CONFIG
from game import Game, Board
from mcts import MCTSPlayer
from mcts_pure import MCTS_Pure
from zip_array import decompress_game_data
if CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
elif CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
else:
    raise NotImplementedError("暂不支持所选框架")

class TrainPipeline:

    def __init__(self, init_model=None):
        # 训练参数
        self.board = Board()
        self.game = Game(self.board)
        self.n_playout = CONFIG['play_out']
        self.lr_multiplier = 1  # 学习率自适应调整
        self.learn_rate = CONFIG['learn_rate',1e-3]
        self.batch_size = CONFIG['batch_size']
        self.c_puct = CONFIG['c_puct']
        self.epochs = CONFIG['epochs']
        self.kl_targ = CONFIG['kl_targ',0.02]  # kl散度控制
        self.check_freq = CONFIG['check_freq',100] # 保存模型的频率
        self.game_batch_num = CONFIG['game_batch_num']  # 训练更新的次数
        self.use_compression = CONFIG.get('use_data_compression', False)
        self.temp = 1.0
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 500

        # 初始化数据缓冲区
        self.data_buffer = deque(maxlen=CONFIG['buffer_size'])

        # 加载模型
        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print('✅ 已加载上次模型:', init_model)
            except Exception as e:
                print('❌ 模型加载失败:', e)
                self.policy_value_net = PolicyValueNet()
        else:
            print('🆕 从零开始训练')
            self.policy_value_net = PolicyValueNet()
    def run_continuously(self):
        """持续训练模式，每隔一段时间检查是否有新数据加入"""
        try:
            while True:
                if not CONFIG['use_redis']:
                    # 从磁盘重新加载数据
                    try:
                        with open(CONFIG['train_data_buffer_path'], 'rb') as f:
                            data_dict = pickle.load(f)
                        raw_data = data_dict['data_buffer']
                        if self.use_compression:
                            new_data = decompress_game_data(raw_data)
                        else:
                            new_data = raw_data

                        self.data_buffer.extend(new_data)
                        print(f"📥 已加载 {len(new_data)} 条新数据")

                    except Exception as e:
                        print(f"❌ 加载新数据失败: {e}")
                else:
                    # 从 Redis 获取新数据
                    pass  # 略，根据你的 Redis 实现补充

                # 检查数据是否足够并训练
                if len(self.data_buffer) > self.batch_size:
                    print("🏋️ 开始本轮训练")
                    self.policy_update()
                else:
                    print("⏳ 数据不足，等待采集...")

                time.sleep(10)  # 每隔10秒检查一次
        except KeyboardInterrupt:
            print("\n\r🛑 训练已手动终止")
    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2 + 1,
                                          is_shown=1)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio


    def policy_update(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        state_batch = np.array(state_batch).astype('float32')
        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')
        winner_batch = np.array(winner_batch).astype('float32')

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(f"KL: {kl:.5f}, LR Multiplier: {self.lr_multiplier:.3f}, Loss: {loss}, Entropy: {entropy}")
        return loss, entropy

    def run(self):
        """开始训练"""
        train_data_path = CONFIG['train_data_buffer_path']

        # 一次性加载所有训练数据
        print(f"📂 正在加载训练数据: {train_data_path}")
        with open(train_data_path, 'rb') as f:
            data_dict = pickle.load(f)

        raw_data = data_dict['data_buffer']
        if self.use_compression:
            self.data_buffer.extend(decompress_game_data(raw_data))
        else:
            self.data_buffer.extend(raw_data)

        print(f"📥 成功加载 {len(self.data_buffer)} 条数据")

        # 开始训练
        for i in range(self.game_batch_num):
            if len(self.data_buffer) < self.batch_size:
                print("⚠️ 数据不足，无法继续训练")
                break

            print(f"🏋️ 第 {i+1} 轮训练开始")
            loss, entropy = self.policy_update()

            # 定期保存模型
            if (i + 1) % self.check_freq == 0:
                model_path = f'models/current_policy_batch_{i+1}.model'
                self.policy_value_net.save_model(model_path)
                print(f"💾 模型已保存至: {model_path}")

        # 最终保存模型
        final_model_path = CONFIG.get('final_model_path', 'models/final_policy.model')
        self.policy_value_net.save_model(final_model_path)
        print(f"🏁 训练完成，最终模型保存至: {final_model_path}")



if __name__ == '__main__':
    init_model = CONFIG.get('init_model_path', None)
    training_pipeline = TrainPipeline(init_model=init_model)

    if CONFIG['collect_and_train']:
        print("🔄 正在同时进行采集和训练...")
        from collect_multi_thread import run_pipeline, dynamic_process_manager
        import threading

        # 启动采集进程管理器作为后台线程
        def start_collector():
            shared_model = PolicyValueNet(model_file=init_model)
            dynamic_process_manager(shared_model, num_processes=CONFIG['num_processes'])

        collector_thread = threading.Thread(target=start_collector, daemon=True)
        collector_thread.start()

        # 开始训练主循环（可定期从磁盘/Redis 获取新数据）
        training_pipeline.run_continuously()
    else:
        print("🆕 只进行训练，不启动采集")
        training_pipeline.run()
