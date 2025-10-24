import random  # 导入随机数生成模块
from typing import Union  # 导入 Union 类型，用于类型注解
from typing import List  # 导入 List 类型，用于类型注解
import torch  # 导入 PyTorch 库

from MADDPG import gradient_estimators  # 导入 MADDPG 库中的梯度估计器
from MADDPG.agent import Agent  # 导入 MADDPG 中的智能体类
from MADDPG.buffer import ReplayBuffer  # 导入 MADDPG 中的经验回放缓冲区类
import numpy as np  # 导入 numpy 库，用于科学计算
from MADDPG.maddpg import MADDPG  # 导入 MADDPG 类
import os  # 导入 os 库，用于文件路径操作
import os.path as path  # 导入 path 模块，用于文件路径操作
from tqdm import tqdm  # 导入 tqdm，用于显示进度条
import datetime  # 导入 datetime 库，用于处理日期和时间
import time  # 导入 time 库，用于计时

# 进度条格式
BAR_FORMAT = "{l_bar}{bar:50}{r_bar}{bar:-10b}"

# 将动作列表转换为整数列表
def _to_action_list(actions):
    processed = []  # 创建一个空的列表，用于存储处理后的动作
    for act in actions:
        if isinstance(act, torch.Tensor):
            # 如果动作是 torch.Tensor 类型，将其转换为整数
            processed.append(int(act.detach().cpu().item()))
        elif isinstance(act, np.ndarray):
            # 如果动作是 numpy.ndarray 类型，将其转换为整数
            processed.append(int(np.asarray(act).reshape(-1)[0]))
        else:
            # 如果动作是其他类型，直接转换为整数
            processed.append(int(act))
    return processed  # 返回处理后的动作列表

# 执行一个训练回合
def play_episode(
        env,
        buffer: Union[ReplayBuffer, None],
        action_fn,
        smax=False,
        render=False
):
    steps = 0  # 初始化步数
    episode_Th = 0  # 初始化 Th
    episode_dump = 0  # 初始化 dump
    episode_Cn = 0  # 初始化 Cn
    episode_average_tao = 0  # 初始化 average_tao
    episode_F = 0  # 初始化 F
    episode_rwd = 0  # 初始化奖励
    done = [False]  # 初始化 done 状态，表示回合是否结束
    obs = env.reset()  # 重置环境，获取初始状态
    dn = env.CalDn()  # 计算初始的 Dn（可能是某个环境的状态信息）

    # 开始回合，直到所有的 done 标志为 True
    while not any(done):
        if smax:
            # 如果启用了 smax，则使用当前状态（dn）计算动作，并掩码动作
            acts = env.mask_actions(_to_action_list(action_fn(dn)))
        else:
            # 否则，使用当前观察到的状态（obs）计算动作并掩码动作，同时计算执行时间
            acts = env.mask_actions(_to_action_list(action_fn(obs)))

        # 环境执行一步，返回新状态、奖励、done标志等
        nobs, rwd, done, Th, dump, Cn, average_tao, F = env.step(np.array(acts), steps)
        if render:
            # 如果需要渲染环境，展示当前状态
            env.render(steps)

        steps += 1  # 步数加一
        # 累加各项统计数据
        episode_Th += Th
        episode_dump += dump
        episode_Cn += Cn
        episode_average_tao += average_tao
        episode_F += F
        episode_rwd += rwd[0]  # 累积奖励（rwd 是一个列表，取第一个元素）

        # 如果缓冲区不为空，将当前状态、动作、奖励等存储到回放缓冲区中
        if buffer is not None:
            buffer.store(
                obs=obs,
                acts=acts,
                rwd=rwd,
                nobs=nobs,
                done=done,
            )
        
        obs = nobs  # 更新当前观察到的状态
        dn = env.CalDn()  # 更新 Dn

    # 返回各项统计结果
    return episode_Th, episode_dump, episode_Cn, episode_average_tao, episode_F , episode_rwd

# 执行一个电力分配的回合
def play_episode_power(
        env,
        action_fn,
        render=False
):
    done = [False]  # 初始化 done 状态
    while not any(done):  # 直到回合结束
        steps = 0  # 初始化步数
        obs = env.reset()  # 重置环境，获取初始状态
        acts = env.mask_actions(_to_action_list(action_fn(obs)))  # 计算动作并掩码
        # 执行环境的步进
        nobs, rwd, done, Th, dump, Cn, average_tao, F = env.step(np.array(acts), steps)
        if render:
            # 如果需要渲染，展示环境状态
            env.render(steps)
        steps += 1  # 步数加一

# MADDPG训练过程
def maddpg_train(config, env):
    buffer = ReplayBuffer(config)  # 初始化经验回放缓冲区
    use_cuda = config.cuda and torch.cuda.is_available()
    if config.cuda and not use_cuda:
        print("CUDA 未检测到，回退到 CPU 设备运行。")
    device = torch.device("cuda" if use_cuda else "cpu")
    # 根据配置选择梯度估计器
    if config.gradient_estimator == "stgs":
        gradient_estimator = gradient_estimators.STGS(config.gumbel_temp)
    elif config.gradient_estimator == "grmck":
        gradient_estimator = gradient_estimators.GRMCK(config.gumbel_temp, config.rao_k)
    elif config.gradient_estimator == "gst":
        gradient_estimator = gradient_estimators.GST(config.gumbel_temp, config.gst_gap)
    elif config.gradient_estimator == "tags":
        gradient_estimator = gradient_estimators.TAGS(config.tags_start, config.tags_end, config.tags_period)
    else:
        print("Unknown gradient estimator type")  # 如果没有找到正确的梯度估计器类型，打印错误
        return None
    
    # 加载已保存的模型（如果需要）
    models = None
    if config.load:
        models = torch.load(
            path.join('save2/', config.model_episode + '.pt'),
            map_location=device,
            weights_only=False
        )
    
    # 创建 MADDPG 模型实例
    maddpg = MADDPG(
        use_cuda=use_cuda,  # 是否使用 CUDA
        n_agents=config.n_agents,  # 智能体的数量（波束数量）
        obs_dim=config.obs_dim,  # 状态的维度
        act_dim=config.act_dim,  # 动作的维度
        critic_lr=config.critic_lr,  # critic 网络的学习率
        actor_lr=config.actor_lr,  # actor 网络的学习率
        gradient_clip=config.gradient_clip,  # 梯度裁剪幅度
        hidden_dim=config.hidden_dim,  # 隐藏层维度
        gamma=config.gamma,  # 折扣因子
        soft_update_size=config.soft_update_size,  # 软更新幅度
        policy_regulariser=config.policy_regulariser,  # 策略正则化
        gradient_estimator=gradient_estimator,  # 选择的梯度估计器
        models=models  # 加载的模型（如果有）
    )

    # 进行预热
    availabel_action = [i for i in range(config.act_dim)]  # 每个智能体可选的动作
    availabel_actions = [availabel_action for _ in range(config.n_agents)]  # 所有智能体可选的动作

    # 进行预热回合
    for _ in tqdm(range(config.warmup_episode), bar_format=BAR_FORMAT, postfix="Warming up..."):
        _, _, _, _, _, _ = play_episode(
            env,
            buffer,
            action_fn=maddpg.acts,  # 使用 MADDPG 的动作策略
        )

    # 主训练过程
    with tqdm(total=config.total_episode, bar_format=BAR_FORMAT) as pbar:
        episode = 0  # 初始化回合数
        while episode < config.total_episode:
            _, _, _, _, _ ,_= play_episode(
                env,
                buffer,
                action_fn=maddpg.acts,  # 使用 MADDPG 动作策略
                render=False)  # 不渲染

            # 每训练一定轮次进行更新
            if episode % config.train_episode == 0:
                for _ in range(config.train_repeat):
                    sample = buffer.sample()  # 从回放缓冲区采样
                    if sample is not None:
                        maddpg.update(sample)  # 更新 MADDPG 模型

            # 每一定轮次进行评估
            if episode % config.eval_episode == 0:
                episode_Ths = 0
                episode_dumps = 0
                episode_Cns = 0
                episode_average_taos = 0
                episode_Fs = 0
                episode_rwds = 0
                for _ in range(config.eval_repeat):
                    episode_Th, episode_dump, episode_Cn, episode_average_tao, episode_F , episode_rwd = play_episode(
                        env,
                        buffer,
                        action_fn=maddpg.acts,
                        render=False
                    )
                    episode_Ths += episode_Th
                    episode_dumps += episode_dump
                    episode_Cns += episode_Cn
                    episode_average_taos += episode_average_tao
                    episode_Fs += episode_F
                    episode_rwds += episode_rwd
                # 输出评估结果
                now = datetime.datetime.now()
                formatted_date = now.strftime("%Y-%m-%d")
                formatted_time = now.strftime("%H:%M:%S")
                output = f"Time {formatted_time}  |  Episode {episode}  |  Beta {env.beta}  |  Th {episode_Ths / config.eval_repeat}  |  dump {episode_dumps / config.eval_repeat}  |  Cn {episode_Cns / config.eval_repeat}  |  average_tao {episode_average_taos / config.eval_repeat}  |  F {episode_Fs / config.eval_repeat}  |  Rwd {episode_rwds / config.eval_repeat}"

                # 保存日志
                log_dir = path.join(config.save_dir, "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_path = path.join(log_dir, "output5.txt")
                with open(log_path, "a") as file:
                    file.write(output)
                    file.write("\n")
                print(output)

            pbar.update(config.train_episode)

            # 每一定轮次保存模型
            if episode % config.save_episode == 0:
                save_path = path.join("save3/", f"{episode}.pt")
                torch.save(maddpg.agents, save_path)
            episode += 1
            if 100000 < episode < 150000:
                env.DecayBeta()  # 衰减 Beta 参数

# 评估训练后的模型
def maddpg_test(config, env):
    use_cuda = config.cuda and torch.cuda.is_available()
    if config.cuda and not use_cuda:
        print("CUDA 未检测到，回退到 CPU 设备运行。")
    device = torch.device("cuda" if use_cuda else "cpu")
    agents: List[Agent] = torch.load(
        path.join('save/', config.model_episode + '.pt'),
        map_location=device
    )  # 加载模型
    maddpg = MADDPG(
        use_cuda=use_cuda,
        n_agents=config.n_agents,
        obs_dim=config.obs_dim,
        act_dim=config.act_dim,
        models=agents,
        critic_lr=0,
        actor_lr=0,
        gradient_clip=0,
        hidden_dim=0,
        gamma=0,
        soft_update_size=0,
        policy_regulariser=0,
        gradient_estimator=None,
    )
    episode_Ths = 0
    episode_dumps = 0
    episode_Cns = 0
    episode_average_taos = 0
    episode_Fs = 0
    for eval_num in range(config.eval_repeat):
        episode_Th, episode_dump, episode_Cn, episode_average_tao, episode_F = play_episode(
            env=env,
            buffer=None,
            action_fn=maddpg.acts,
            render=True if (eval_num == config.eval_repeat - 1) and config.render else False
        )
        episode_Ths += episode_Th
        episode_dumps += episode_dump
        episode_Cns += episode_Cn
        episode_average_taos += episode_average_tao
        episode_Fs += episode_F
    print(f"Max {env.uniform_max}  Th {episode_Ths / config.eval_repeat / 6}  dump {episode_dumps / config.eval_repeat / 6}  Cn {(episode_Ths / config.eval_repeat + episode_Cns / config.eval_repeat) / 6}  average_tao {(episode_average_taos / config.eval_repeat / 50 + 1) * 2}  F {episode_Fs / config.eval_repeat / 50 * 2}")

# 电力分配评估
def power_allocate(config, env):
    use_cuda = config.cuda and torch.cuda.is_available()
    if config.cuda and not use_cuda:
        print("CUDA 未检测到，回退到 CPU 设备运行。")
    device = torch.device("cuda" if use_cuda else "cpu")
    agents: List[Agent] = torch.load(
        path.join('save/', config.model_episode + '.pt'),
        map_location=device
    )  # 加载模型
    maddpg = MADDPG(
        use_cuda=use_cuda,
        n_agents=config.n_agents,
        obs_dim=config.obs_dim,
        act_dim=config.act_dim,
        models=agents,
        critic_lr=0,
        actor_lr=0,
        gradient_clip=0,
        hidden_dim=0,
        gamma=0,
        soft_update_size=0,
        policy_regulariser=0,
        gradient_estimator=None,
    )
    for eval_num in range(config.eval_repeat):
         _ = play_episode_power(
            env=env,
            buffer=None,
            action_fn=maddpg.acts,
            render=True if (eval_num == config.eval_repeat - 1) and config.render else False
        )
