from game import Board, Game
from mcts import MCTSPlayer
from pytorch_net import PolicyValueNet
import torch

def evaluate_models(new_model_path, old_model_path, num_games=10):
    new_model = PolicyValueNet(model_file=new_model_path)
    old_model = PolicyValueNet(model_file=old_model_path)

    new_player = MCTSPlayer(new_model.policy_value_fn, is_selfplay=0)
    old_player = MCTSPlayer(old_model.policy_value_fn, is_selfplay=0)

    board = Board()
    game = Game(board)
    results = {'new': 0, 'old': 0, 'draw': 0}

    for i in range(num_games):
        winner = game.start_play(new_player, old_player, start_player=i % 2 + 1, is_shown=1)
        if winner == 1:
            results['new'] += 1
        elif winner == 2:
            results['old'] += 1
        else:
            results['draw'] += 1

    print(f"\n✅ 新模型胜率: {results['new']}/{num_games}, 败率: {results['old']}/{num_games}, 平局: {results['draw']}/{num_games}")
    return results
