import numpy as np  # 导入 numpy 库，用于科学计算和随机数生成
from torch import Tensor  # 导入 PyTorch 中的 Tensor 类

class ReplayBuffer:
    def __init__(self, config):
        # 初始化经验回放缓冲区，配置了缓冲区的大小、批量大小、观察维度等参数
        self.capacity = config.replay_buffer_size  # 缓冲区的容量，最大存储条目数
        self.batch_size = config.batch_size  # 每次采样的批次大小
        self.obs_dim = config.obs_dim  # 每个观察的维度（状态的大小）
        self.n_agents = config.n_agents  # 智能体的数量
        self.entries = 0  # 当前存储的条目数

        # 初始化存储观察、动作、奖励、下一个观察和完成标志的内存
        # 使用 PyTorch Tensor 存储数据，可以方便地与模型（基于 PyTorch）进行交互
        self.memory_obs = Tensor(self.capacity, self.obs_dim)  # 存储所有的观察（状态）
        self.memory_nobs = Tensor(self.capacity, self.obs_dim)  # 存储下一个观察（状态）
        self.memory_acts = Tensor(self.capacity, self.n_agents)  # 存储所有的动作
        self.memory_rwd = Tensor(self.capacity, 1)  # 存储所有的奖励（每个经验的奖励）
        self.memory_done = Tensor(self.capacity, 1)  # 存储所有的完成标志（是否结束）

    def store(self, obs, acts, rwd, nobs, done):
        # 将一个新经历（状态、动作、奖励、下一个状态、完成标志）存储到经验回放缓冲区
        store_index = self.entries % self.capacity  # 计算当前存储的位置，采用模运算保证缓冲区的循环利用

        # 存储当前的经验数据
        self.memory_obs[store_index, :] = Tensor(obs)  # 存储当前观察
        self.memory_nobs[store_index, :] = Tensor(nobs)  # 存储下一个观察
        self.memory_acts[store_index, :] = Tensor(acts)  # 存储动作
        self.memory_rwd[store_index, :] = Tensor(rwd)  # 存储奖励
        self.memory_done[store_index, :] = Tensor(done)  # 存储完成标志

        # 更新存储条目的数量
        self.entries += 1

    def sample(self):
        # 从回放缓冲区中随机抽取一个批次的样本
        if not self.ready():
            return None  # 如果缓冲区没有足够的数据，就返回 None

        # 从缓冲区中随机选择一批索引，确保每个样本只被选择一次
        idxs = np.random.choice(
            np.min((self.entries, self.capacity)),  # 选择最小值，确保不超过容量或当前存储的条目数
            size=(self.batch_size,),  # 选择批次大小的索引
            replace=False  # 不放回选择（即每个样本只能被选择一次）
        )

        # 返回一个字典，包含从缓冲区中抽取的数据
        return {
            "obs": self.memory_obs[idxs, :],  # 当前的观察（状态）
            "acts": self.memory_acts[idxs, :],  # 对应的动作
            "rwd": self.memory_rwd[idxs, :],  # 对应的奖励
            "nobs": self.memory_nobs[idxs, :],  # 下一个观察（状态）
            "done": self.memory_done[idxs, :],  # 完成标志（是否结束）
        }

    def ready(self):
        # 判断缓冲区是否准备好进行采样
        return self.batch_size <= self.entries  # 如果缓冲区中存储的条目数大于等于批次大小，返回 True，表示可以进行采样
