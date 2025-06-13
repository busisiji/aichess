# -*- coding: utf-8 -*-
import numpy as np
import copy
import time
from config import CONFIG
from collections import deque   # 这个队列用来判断长将或长捉
import random

num2array = dict({1 : np.array([1, 0, 0, 0, 0, 0, 0]), 2: np.array([0, 1, 0, 0, 0, 0, 0]),
                  3: np.array([0, 0, 1, 0, 0, 0, 0]), 4: np.array([0, 0, 0, 1, 0, 0, 0]),
                  5: np.array([0, 0, 0, 0, 1, 0, 0]), 6: np.array([0, 0, 0, 0, 0, 1, 0]),
                  7: np.array([0, 0, 0, 0, 0, 0, 1]), 8: np.array([-1, 0, 0, 0, 0, 0, 0]),
                  9: np.array([0, -1, 0, 0, 0, 0, 0]), 10: np.array([0, 0, -1, 0, 0, 0, 0]),
                  11: np.array([0, 0, 0, -1, 0, 0, 0]), 12: np.array([0, 0, 0, 0, -1, 0, 0]),
                  13: np.array([0, 0, 0, 0, 0, -1, 0]), 14: np.array([0, 0, 0, 0, 0, 0, -1]),
                  15: np.array([0, 0, 0, 0, 0, 0, 0])})
def array2num(array):
    return list(filter(lambda string: (num2array[string] == array).all(), num2array))[0]

# 压缩存储
def state_list2state_num_array(state_list):
    _state_array = np.zeros([10, 9, 7])
    for i in range(10):
        for j in range(9):
            _state_array[i][j] = num2array[state_list[i][j]]
    return _state_array

#(state, mcts_prob, winner) ((9,10,9),2086,1) => ((9,90),(2,1043),1)
def zip_state_mcts_prob(tuple):
    state, mcts_prob, winner = tuple
    state = state.reshape((9,-1))
    mcts_prob = mcts_prob.reshape((2,-1))
    state = zip_array(state)
    mcts_prob = zip_array(mcts_prob)
    return state,mcts_prob,winner

def recovery_state_mcts_prob(tuple):
    state, mcts_prob, winner = tuple
    state = recovery_array(state)
    mcts_prob = recovery_array(mcts_prob)
    state = state.reshape((9,10,9))
    mcts_prob = mcts_prob.reshape(2086)
    return state,mcts_prob,winner

def zip_array(array, data=0.):  # 压缩成稀疏数组
    zip_res = []
    zip_res.append([len(array), len(array[0])])
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i][j] != data:
                zip_res.append([i, j, array[i][j]])
    return np.array(zip_res, dtype=object)


def recovery_array(array, data=0.):  # 恢复数组
    recovery_res = []
    for i in range(array[0][0]):
        recovery_res.append([data for i in range(array[0][1])])
    for i in range(1, len(array)):
        # print(len(recovery_res[0]))
        recovery_res[array[i][0]][array[i][1]] = array[i][2]
    return np.array(recovery_res)

# 将二维数组转换为稀疏表示形式，仅记录非零值的位置及其数值。
def zip_array_fast(array, data=0.):
    if not isinstance(array, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(array)}")
    if array.ndim == 1:
        array = array.reshape(-1, 1)  # 自动转为二维
    elif array.ndim != 2:
        raise ValueError(f"Input array must be 1D or 2D, got shape {array.shape}")

    indices = np.where(array != data)
    values = array[indices]
    shape = array.shape
    return np.array([shape[0], shape[1]], dtype=np.int64), np.column_stack((*indices, values))


# 将压缩后的数据还原成原始格式
def recovery_array_fast(shape_info, data_array, data=0.):
    rows, cols = shape_info
    res = np.full((rows, cols), data, dtype=np.float32)
    for i in range(len(data_array)):
        x = int(data_array[i][0])
        y = int(data_array[i][1])
        val = data_array[i][2]
        res[x, y] = val
    return res

# 遍历所有需要压缩的状态数据，对每个状态进行压缩处理。
def batch_zip_states(states):
    compressed_list = []
    for state in states:
        shape_info, sparse_data = zip_array_fast(state)
        compressed_list.append((shape_info, sparse_data))
    return compressed_list

def compress_game_data(game_data):
    """
    压缩棋局数据。

    Args:
        game_data (list): 包含多个 (state, mcts_prob, winner) 的列表。

    Returns:
        list: 压缩后的数据列表，每个元素包含形状信息、稀疏数据以及胜者。
    """
    compressed_data = []
    for state, mcts_prob, winner in game_data:
        state = np.array(state)
        state_2d = state.reshape(-1, state.shape[-1])  # 转换为二维数组以便压缩
        mcts_prob = np.array(mcts_prob)
        mcts_prob_2d = mcts_prob.reshape(-1, 1)

        shape_info_state, sparse_data_state = zip_array_fast(state_2d)
        shape_info_mcts, sparse_data_mcts = zip_array_fast(mcts_prob_2d)

        compressed_data.append((
            shape_info_state, sparse_data_state,
            shape_info_mcts, sparse_data_mcts,
            winner
        ))
    return compressed_data

def decompress_game_data(compressed_data, original_shape=(9, 10, 9)):
    """
    解压棋局数据。

    Args:
        compressed_data (list): 压缩后的数据列表，每个元素包含形状信息、稀疏数据以及胜者。
        original_shape (tuple): 恢复后 state 的原始形状，默认为 (9, 10, 9)。

    Returns:
        list: 解压后的棋局数据列表，每个元素为 (state, mcts_prob, winner)。
    """
    decompressed_data = []
    for entry in compressed_data:
        if len(entry) != 5:
            return compressed_data

        shape_info_state, sparse_data_state, shape_info_mcts, sparse_data_mcts, winner = entry

        # 恢复 state 和 mcts_prob
        state_2d = recovery_array_fast(shape_info_state, sparse_data_state)
        mcts_prob_2d = recovery_array_fast(shape_info_mcts, sparse_data_mcts)

        # 恢复原始形状
        state = state_2d.reshape(original_shape)
        mcts_prob = mcts_prob_2d.flatten()

        decompressed_data.append((state, mcts_prob, winner))
    return decompressed_data
