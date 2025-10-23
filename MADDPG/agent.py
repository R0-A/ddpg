from copy import deepcopy  # 导入 deepcopy 用于深拷贝对象
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的功能性函数，如损失函数等
from torch.optim import Adam  # 导入 Adam 优化器
from MADDPG.networks import ActorNetwork, CriticNetwork  # 导入 MADDPG 的演员网络和评论员网络

class Agent:
    def __init__(self,
        use_cuda,  # 是否使用CUDA
        agent_idx,  # 当前智能体的索引（在多智能体环境中）
        obs_dims,  # 每个智能体的观察维度
        act_dims,  # 每个智能体的动作维度
        hidden_dim,  # 隐藏层的维度
        critic_lr,  # 评论员网络的学习率
        actor_lr,  # 演员网络的学习率
        gradient_clip,  # 梯度裁剪的大小
        soft_update_size,  # 软更新的大小
        policy_regulariser,  # 策略正则化
        gradient_estimator,  # 梯度估计器
        device=None,
    ):
        self.device = device if device is not None else torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.use_cuda = self.device.type == "cuda"  # 是否使用GPU加速
        self.agent_idx = agent_idx  # 当前智能体的索引
        self.soft_update_size = soft_update_size  # 软更新步长
        self.n_obs = obs_dims[self.agent_idx]  # 获取当前智能体的观察维度
        self.n_acts = act_dims[self.agent_idx]  # 获取当前智能体的动作维度
        self.n_agents = len(obs_dims)  # 智能体的总数
        self.gradient_clip = gradient_clip  # 梯度裁剪的大小
        self.policy_regulariser = policy_regulariser  # 策略正则化参数
        self.gradient_estimator = gradient_estimator  # 梯度估计器，用于稳定训练

        # ******************** POLICY (演员网络) ********************

        # 创建演员网络（Policy network），用于产生动作
        self.policy = ActorNetwork(self.n_obs, hidden_dim, self.n_acts)
        
        # 创建目标演员网络（Target policy network），用于稳定训练
        self.target_policy = ActorNetwork(self.n_obs, hidden_dim, self.n_acts)
        self.target_policy.hard_update(self.policy)  # 将目标演员网络参数设置为演员网络的当前参数

        # 将网络移动到目标设备
        self.policy = self.policy.to(self.device)
        self.target_policy = self.target_policy.to(self.device)

        # ******************** CRITIC (评论员网络) ********************

        # 创建评论员网络（Critic network），用于评估动作的价值
        self.critic = CriticNetwork(obs_dims, act_dims, hidden_dim)
        self.target_critic = CriticNetwork(obs_dims, act_dims, hidden_dim)
        self.target_critic.hard_update(self.critic)  # 将目标评论员网络参数设置为评论员网络的当前参数

        # 将评论员网络移动到目标设备
        self.critic = self.critic.to(self.device)
        self.target_critic = self.target_critic.to(self.device)

        # ******************** OPTIMISERS (优化器) ********************

        # 创建演员网络的优化器
        self.optim_actor = Adam(self.policy.parameters(), lr=actor_lr, eps=0.001)
        
        # 创建评论员网络的优化器
        self.optim_critic = Adam(self.critic.parameters(), lr=critic_lr, eps=0.001)

    # ******************** ACTION SELECTION (动作选择) ********************

    # 使用当前策略网络来选择动作（行为策略）
    def act_behaviour(self, obs):
        # 将观测输入到策略网络，获取动作的输出
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        policy_output = self.policy(obs_tensor)
        
        # 使用梯度估计器来计算不需要梯度的动作输出（返回最优动作）
        gs_output = self.gradient_estimator(policy_output, need_gradients=False)

        # 返回动作输出的最大值的索引作为最终动作
        return torch.argmax(gs_output, dim=-1)

    # 使用目标策略网络来选择目标动作（用于目标更新）
    def act_target(self, obs):
        # 将观测输入到目标策略网络，获取目标动作的输出
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        policy_output = self.target_policy(obs_tensor)
        
        # 使用梯度估计器计算不需要梯度的目标动作输出
        gs_output = self.gradient_estimator(policy_output, need_gradients=False)

        # 返回目标动作的最大值的索引作为目标动作
        return torch.argmax(gs_output, dim=-1)

    # ******************** CRITIC UPDATE (评论员网络更新) ********************

    # 更新评论员网络的参数
    def update_critic(self, all_obs, all_nobs, target_actions_per_agent, sampled_actions_per_agent, rewards, dones, gamma):
        # 将目标动作和采样动作连接成一个批次
        target_actions = torch.cat(target_actions_per_agent, dim=1).to(self.device)
        sampled_actions = torch.cat(sampled_actions_per_agent, dim=1).to(self.device)

        # 计算目标Q值（使用目标评论员网络）
        Q_next_target = self.target_critic(torch.cat((all_nobs, target_actions), dim=1))
        
        # 计算目标y值（奖励 + 折扣后的下一个Q值）
        target_ys = rewards + (1 - dones) * gamma * Q_next_target
        
        # 计算当前Q值（使用评论员网络）
        behaviour_ys = self.critic(torch.cat((all_obs, sampled_actions), dim=1))

        # 计算MSE损失（评论员网络的目标是最小化此损失）
        loss = F.mse_loss(behaviour_ys, target_ys.detach())

        # 更新评论员网络的参数
        self.optim_critic.zero_grad()  # 清空优化器的梯度
        loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)  # 梯度裁剪
        self.optim_critic.step()  # 更新评论员网络的参数

    # ******************** ACTOR UPDATE (演员网络更新) ********************

    # 更新演员网络的参数
    def update_actor(self, all_obs, agent_obs, sampled_actions):
        # 获取演员网络的输出（动作）
        policy_outputs = self.policy(agent_obs)
        
        # 使用梯度估计器计算梯度（目标是最大化评论员的Q值）
        gs_outputs = self.gradient_estimator(policy_outputs)

        # 深拷贝当前采样的动作，并使用目标梯度更新当前智能体的动作
        _sampled_actions = deepcopy(sampled_actions)
        _sampled_actions[self.agent_idx] = gs_outputs  # 更新当前智能体的动作

        # 计算损失（负的评论员Q值）
        loss = - self.critic(torch.cat((all_obs, *_sampled_actions), dim=1)).mean()
        
        # 加入策略正则化项（鼓励策略输出动作的范围广泛性）
        loss += (policy_outputs ** 2).mean() * self.policy_regulariser

        # 更新演员网络的参数
        self.optim_actor.zero_grad()  # 清空优化器的梯度
        loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)  # 梯度裁剪
        self.optim_actor.step()  # 更新演员网络的参数

    # ******************** SOFT UPDATE (软更新目标网络) ********************

    # 软更新目标评论员网络和目标策略网络的参数
    def soft_update(self):
        # 目标评论员网络的软更新
        self.target_critic.soft_update(self.critic, self.soft_update_size)
        
        # 目标策略网络的软更新
        self.target_policy.soft_update(self.policy, self.soft_update_size)

    def set_device(self, device):
        """将代理及其优化器的状态转移到指定设备"""
        current_device = getattr(self, "device", torch.device("cpu"))
        if current_device == device:
            return
        self.device = device
        self.use_cuda = device.type == "cuda"
        self.policy = self.policy.to(device)
        self.target_policy = self.target_policy.to(device)
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)

        for optim in (self.optim_actor, self.optim_critic):
            for state in optim.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
