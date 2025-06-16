# -*- coding: utf-8 -*-

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

# 数据服务模块（统一 Redis / 文件）
from data_service import DataManagementService

# 根据配置选择网络框架
if CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
elif CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
else:
    raise NotImplementedError("暂不支持所选框架")


class TrainPipeline:

    def __init__(self, init_model=None):
        # 游戏环境初始化
        self.board = Board()
        self.game = Game(self.board)

        # 训练参数
        self.n_playout = CONFIG.get('play_out', 400)  # MCTS 搜索次数
        self.lr_multiplier = 1.0  # 学习率自适应调整因子
        self.learn_rate = CONFIG.get('learn_rate', 1e-3)
        self.batch_size = CONFIG.get('batch_size', 512)
        self.c_puct = CONFIG.get('c_puct', 5)
        self.epochs = CONFIG.get('epochs', 5)
        self.kl_targ = CONFIG.get('kl_targ', 0.02)  # KL散度目标
        self.check_freq = CONFIG.get('check_freq', 100)  # 模型保存频率
        self.game_batch_num = CONFIG.get('game_batch_num', 1000)  # 总训练轮数
        self.use_compression = CONFIG.get('use_data_compression', False)

        # 初始化数据服务
        self.data_service = DataManagementService()
        self.data_buffer = self.data_service.data_buffer  # 共享缓冲区

        # 加载检查点（断点续训）
        self.data_service.checkpoint = self.data_service.load_checkpoint()
        self.iters = self.data_service.checkpoint.get('iters', 0)
        self.model_path = self.data_service.checkpoint.get('model_path', None)
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 500

        # 加载模型
        if self.model_path:
            try:
                self.policy_value_net = PolicyValueNet(model_file=self.model_path)
                print(f'✅ 已从上次模型继续训练: {self.model_path}')
            except Exception as e:
                print(f'❌ 模型加载失败: {e}')
                self.policy_value_net = PolicyValueNet()
        else:
            print('🆕 从零开始训练')
            self.policy_value_net = PolicyValueNet()

    def run_continuously(self):
        """持续训练模式：每隔一段时间检查是否有新数据加入"""
        try:
            while True:
                # 刷新数据
                new_data = self.data_service.refresh_data()
                print(f"🔄 当前数据缓存大小: {len(self.data_buffer)}")

                # 如果数据足够则训练
                if len(self.data_buffer) >= self.batch_size:
                    print("🏋️ 开始本轮训练")
                    loss, entropy = self.policy_update()
                    self.iters += 1

                    # 定期保存模型和检查点
                    if self.iters % self.check_freq == 0:
                        model_path = f'models/current_policy_iter_{self.iters}.model'
                        self.policy_value_net.save_model(model_path)
                        self.data_service.save_checkpoint(self.iters, model_path)
                        print(f"💾 模型已保存至: {model_path}")

                        # 每隔一定迭代评估胜率
                        win_ratio = self.policy_evaluate(n_games=5)
                        if win_ratio > self.best_win_ratio:
                            best_model_path = f'models/best_policy_iter_{self.iters}.model'
                            self.policy_value_net.save_model(best_model_path)
                            self.best_win_ratio = win_ratio
                            print(f"🏆 最佳模型更新，胜率: {win_ratio:.2f}")
                else:
                    print("⏳ 数据不足，等待采集...")

                time.sleep(10)  # 每隔10秒检查一次
        except KeyboardInterrupt:
            print("\n\r🛑 训练已手动终止，正在保存最终模型...")
            final_model_path = CONFIG.get('pytorch_model_path', 'models/final_policy.model')
            self.policy_value_net.save_model(final_model_path)
            self.data_service.save_checkpoint(self.iters, final_model_path)
            print(f"✅ 最终模型已保存至: {final_model_path}")

    def policy_evaluate(self, n_games=10):
        """
        对抗纯 MCTS 玩家评估策略性能
        """
        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout
        )
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(
                current_mcts_player,
                pure_mcts_player,
                start_player=i % 2 + 1,
                is_shown=1
            )
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print(f"🎮 胜率: {win_ratio:.2f} (Win: {win_cnt[1]}, Lose: {win_cnt[2]}, Tie: {win_cnt[-1]})")
        return win_ratio

    def policy_update(self):
        """
        更新策略价值网络
        """
        if len(self.data_buffer) < self.batch_size:
            raise ValueError("⚠️ 数据不足，无法训练")

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

        print(f"📊 KL: {kl:.5f}, LR Multiplier: {self.lr_multiplier:.3f}, Loss: {loss:.4f}, Entropy: {entropy:.4f}")
        return loss, entropy

    def run(self):
        """
        单次训练流程（非持续训练）
        """
        print(f"📂 正在加载初始训练数据: {self.data_service.train_data_path}")
        self.data_service.load_initial_data()

        print(f"📥 成功加载 {len(self.data_buffer)} 条数据")

        for i in range(self.game_batch_num):
            if len(self.data_buffer) < self.batch_size:
                print("⚠️ 数据不足，无法继续训练")
                break

            print(f"🏋️ 第 {i+1} 轮训练开始")
            self.policy_update()

            # 定期保存模型
            if (i + 1) % self.check_freq == 0:
                model_path = f'models/current_policy_batch_{i+1}.model'
                self.policy_value_net.save_model(model_path)
                self.data_service.save_checkpoint(i + 1, model_path)
                print(f"💾 模型已保存至: {model_path}")

        # 最终保存
        if CONFIG['use_frame'] == 'paddle':
            self.policy_value_net.save_model(CONFIG['paddle_model_path'])
            print(f"🏁 训练完成，最终模型保存至: {CONFIG['paddle_model_path']}")
        elif CONFIG['use_frame'] == 'pytorch':
            self.policy_value_net.save_model(CONFIG['pytorch_model_path'])
            print(f"🏁 训练完成，最终模型保存至: {CONFIG['pytorch_model_path']}")
        else:
            print('不支持所选框架')



if __name__ == '__main__':
    init_model = CONFIG.get('init_model_path', None)
    training_pipeline = TrainPipeline(init_model=init_model)

    if CONFIG.get('collect_and_train', False):
        print("🔄 正在同时进行采集和训练...")
        from collect_multi_thread import run_pipeline, dynamic_process_manager
        import threading

        def start_collector():
            shared_model = PolicyValueNet(model_file=init_model)
            dynamic_process_manager(shared_model, num_processes=CONFIG['num_processes'])

        collector_thread = threading.Thread(target=start_collector, daemon=True)
        collector_thread.start()

        # 主线程运行训练
        training_pipeline.run_continuously()
    else:
        print("🆕 只进行训练，不启动采集")
        training_pipeline.run()
