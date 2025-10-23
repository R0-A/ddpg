from torch import nn  # 从 PyTorch 中导入 nn 模块，用于构建神经网络

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.obs_dim = obs_dim  # 存储状态维度（观察的维度）
        
        # 定义神经网络层，使用 nn.Sequential 依次堆叠多个全连接层和激活函数
        self.layers = nn.Sequential(*[
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),  # 输入层到第一个隐藏层，ReLU 激活
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),  # 第一个隐藏层到第二个隐藏层，ReLU 激活
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),  # 第二个隐藏层到第三个隐藏层，ReLU 激活
            nn.Linear(hidden_dim, n_actions)  # 最后一层输出动作维度
        ])

    def forward(self, obs):
        # 前向传播：将状态输入网络，输出动作
        return self.layers(obs)

    def hard_update(self, source):
        # 硬更新：将源网络的参数复制到目标网络中
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        # 软更新：根据 t 值对网络参数进行软更新（将目标网络参数进行平滑更新）
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


class CriticNetwork(nn.Module):
    def __init__(self, all_obs_dims, all_acts_dims, hidden_dim):
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        # 计算输入层的大小，输入包括所有智能体的状态和动作
        input_size = all_obs_dims[0] + sum(all_acts_dims)
        
        # 定义神经网络层，使用 nn.Sequential 依次堆叠多个全连接层和激活函数
        self.layers = nn.Sequential(*[
            nn.Linear(input_size, hidden_dim),  # 输入层到第一个隐藏层
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(hidden_dim, hidden_dim),  # 第一个隐藏层到第二个隐藏层
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(hidden_dim, hidden_dim),  # 第二个隐藏层到第三个隐藏层
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(hidden_dim, 1),  # 输出层，最终输出一个值（动作价值 Q）
        ])

    def forward(self, obs_and_acts):
        # 前向传播：将状态和动作拼接起来输入网络，输出 Q 值（动作价值）
        return self.layers(obs_and_acts)

    def hard_update(self, source):
        # 硬更新：将源网络的参数复制到目标网络中
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        # 软更新：根据 t 值对网络参数进行软更新
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)
