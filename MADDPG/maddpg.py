import torch  # 导入 PyTorch 主库
from MADDPG.agent import Agent  # 导入单个智能体的封装
from typing import List, Optional  # 导入类型提示所需的 List、Optional
import torch.nn.functional as F  # 导入常用的神经网络函数接口

from MADDPG.gradient_estimators import GradientEstimator  # 导入离散动作梯度估计器基类

class MADDPG:
    def __init__(
        self,
        use_cuda : bool,  # 指定是否使用 CUDA
        n_agents : int,  # 智能体数量
        obs_dim : int,  # 单个智能体观测维度
        act_dim : int,  # 单个智能体动作空间大小
        critic_lr : float,  # Critic 学习率
        actor_lr : float,  # Actor 学习率
        gradient_clip : float,  # 梯度裁剪阈值
        hidden_dim : int,  # 隐层维度
        gamma : float,  # 折扣因子
        soft_update_size : float,  # 软更新系数
        policy_regulariser : float,  # 策略正则化权重
        gradient_estimator : GradientEstimator,  # 离散动作梯度估计器
        models : Optional[ List[Agent] ] = None,  # 可选的已保存模型列表
    ):
        # 缓存核心超参数，并构造每个智能体的观测与动作规格
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.use_cuda = self.device.type == "cuda"
        self.n_agents = n_agents  # 保存智能体数量
        self.gamma = gamma  # 保存折扣因子
        obs_dims = [obs_dim for _ in range(self.n_agents)]  # 为每个智能体复制观测维度
        act_dims = [act_dim for _ in range(self.n_agents)]  # 为每个智能体复制动作维度
        # 针对每个波束创建一个 Agent（或复用已加载的模型）
        self.agents = [
            Agent(
                use_cuda=self.use_cuda,  # 传入 CUDA 设置
                agent_idx=ii,  # 当前智能体索引
                obs_dims=obs_dims,  # 所有智能体观测维度列表
                act_dims=act_dims,  # 所有智能体动作维度列表
                hidden_dim=hidden_dim,  # 隐层维度
                critic_lr=critic_lr,  # Critic 学习率
                actor_lr=actor_lr,  # Actor 学习率
                gradient_clip=gradient_clip,  # 梯度裁剪阈值
                soft_update_size=soft_update_size,  # 软更新幅度
                policy_regulariser=policy_regulariser,  # 策略正则化
                gradient_estimator=gradient_estimator,  # 离散梯度估计器
                device=self.device,
            )
            for ii in range(self.n_agents)  # 遍历所有智能体索引
        ] if models is None else models  # 若提供模型则直接复用

        if models is not None:
            for agent in self.agents:
                agent.set_device(self.device)

        # 共享的离散梯度估计器，为所有智能体提供采样策略
        self.gradient_estimator = gradient_estimator  # 保存梯度估计器引用

    def acts(self, obs):
        # 依次前向计算所有智能体的行为策略，得到联合动作
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = [
            self.agents[ii].act_behaviour(obs_tensor).detach().cpu()
            for ii in range(self.n_agents)
        ]  # 逐个智能体采样动作并转回 CPU
        return actions  # 返回动作列表

    def update(self, sample):
        # 解包批量数据，若可用则搬运到 GPU
        obs = sample['obs'].to(self.device)  # 当前观测批次
        nobs = sample['nobs'].to(self.device)  # 下一时刻观测批次
        acts = sample['acts'].to(self.device, dtype=torch.int64)  # 已执行动作
        rewards = sample['rwd'].to(self.device)  # 奖励
        dones = sample['done'].to(self.device)  # 终止标记
        all_obs = obs  # 当前观测别名，方便传参
        all_nobs = nobs  # 下一观测别名

        # 使用目标策略计算下一时刻动作，用于 TD 目标
        target_actions = [
            self.agents[ii].act_target(nobs)  # 计算每个智能体的目标动作
            for ii in range(self.n_agents)  # 遍历所有智能体
        ]

        # 将离散动作转换为 one-hot，以便输入到 Critic
        target_actions_one_hot = [
            F.one_hot(target_actions[ii], num_classes=self.agents[ii].n_acts).to(self.device, dtype=torch.float32)  # 目标动作 one-hot
            for ii in range(self.n_agents)  # 遍历所有智能体
        ]

        sampled_actions_one_hot = [
            F.one_hot(acts[:, ii], num_classes=self.agents[ii].n_acts).to(self.device, dtype=torch.float32)  # 经验中真实动作的 one-hot
            for ii in range(self.n_agents)  # 遍历所有智能体
        ]

        # 对每个智能体更新 Critic/Actor，其他智能体动作视作常量
        for ii, agent in enumerate(self.agents):  # 遍历智能体及其代理
            agent.update_critic(
                all_obs=all_obs,  # 当前观测
                all_nobs=all_nobs,  # 下一观测
                target_actions_per_agent=target_actions_one_hot,  # 下一动作 one-hot
                sampled_actions_per_agent=sampled_actions_one_hot,  # 当前动作 one-hot
                rewards=rewards,  # 奖励
                dones=dones,  # 终止标记
                gamma=self.gamma,  # 折扣因子
            )

            agent.update_actor(
                all_obs=all_obs,  # 当前全局观测
                agent_obs=obs,  # 当前局部观测
                sampled_actions=sampled_actions_one_hot,  # 当前动作 one-hot
            )

        # 执行软更新，稳定目标网络
        for agent in self.agents:  # 遍历每个智能体
            agent.soft_update()  # 执行软更新

        # 触发梯度估计器状态更新（如退火温度）
        if self.gradient_estimator is not None:
            self.gradient_estimator.update_state()  # 更新梯度估计器内部状态
        return None  # 返回空值结束
