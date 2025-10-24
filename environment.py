import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib import rcParams



class Satellite:
    def __init__(self, config):
        self.num_satellites = 5
        self.total_beams = 10
        self.beams_per_satellite = 2
        beams_per_sat = self.beams_per_satellite
        self.total_beams = 10
        self.K = 10
        self.N = config.N   # N个小区
        self.T_ttl = config.T_ttl  # 流量需求最大时延
        self.Episode_Length = config.Episode_Length     # 每回合长度
        self.beta = config.beta
        # self.Cell = [
        #             [-2, 4], [0, 4], [2, 4],
        #         [-3, 2], [-1, 2], [1, 2], [3, 2],
        #     [-4, 0], [-2, 0], [0, 0], [2, 0], [4, 0],
        #         [-3, -2], [-1, -2], [1, -2], [3, -2],
        #             [-2, -4], [0, -4], [2, -4]
        # ]   # 19 | 4
        # self.Cell = [
        #             [-4, 4], [-2, 4], [0, 4], [2, 4], [4, 4],
        #         [-5, 2], [-3, 2], [-1, 2], [1, 2], [3, 2], [5, 2],
        #             [-4, 0], [-2, 0], [0, 0], [2, 0], [4, 0],
        #         [-5, -2], [-3, -2], [-1, -2], [1, -2], [3, -2], [5, -2],
        #             [-4, -4], [-2, 4], [0, -4], [2, -4], [4, -4],
        # ]   # 27 | 6
        self.Cell = [
                        [-3, 6], [-1, 6], [1, 6], [3, 6],
                    [-4, 4], [-2, 4], [0, 4], [2, 4], [4, 4],
                [-5, 2], [-3, 2], [-1, 2], [1, 2], [3, 2], [5, 2],
            [-6, 0], [-4, 0], [-2, 0], [0, 0], [2, 0], [4, 0], [6, 0],
                [-5, -2], [-3, -2], [-1, -2], [1, -2], [3, -2], [5, -2],
                    [-4, -4], [-2, 4], [0, -4], [2, -4], [4, -4],
                        [-3, -6], [-1, -6], [1, -6], [3, -6]
        ]   # 37 | 8
        self.num_high_orbit = 1
        self.low_orbit_altitude = max(1e-3, getattr(config, "low_orbit_altitude", getattr(config, "h0", 550)))
        self.high_orbit_altitude = max(1e-3, getattr(config, "high_orbit_altitude", self.low_orbit_altitude))
        self.satellite_altitudes = self._init_satellite_altitudes()
        self.satellite_coverages = self._init_satellite_coverages()
        self.agent_allowed_actions = self._init_agent_allowed_actions()
        self.D = np.zeros((self.T_ttl, self.N))  # 流量需求矩阵
        self.Actions = [0 for _ in range(self.K)]   # 记录当前时刻智能体的动作选择
        base_palette = ['gold', 'royalblue', 'darkorange', 'seagreen', 'crimson', 'mediumpurple', 'sienna', 'deeppink']
        self.AgentColors = []
        for sat_idx in range(self.num_satellites):
            if sat_idx < self.num_high_orbit:
                color = 'gold'
            else:
                color = base_palette[(sat_idx - self.num_high_orbit + 1) % len(base_palette)]
            self.AgentColors.extend([color] * self.beams_per_satellite)
        if len(self.AgentColors) < self.K:
            self.AgentColors.extend(['royalblue'] * (self.K - len(self.AgentColors)))
        self.AgentRGBColors = [mcolors.CSS4_COLORS[color] for color in self.AgentColors]        # 获取颜色名称对应的RGB颜色值
        self.distribution = config.distribution     # 到达流量服从的分布
        self.an_lambda = np.random.rand(self.N)  # 使用随机数控制各小区在某一时刻有到达流量的概率
        self.intra_sat_cochannel_factor = getattr(config, "intra_sat_cochannel_factor", 1.0)
        self.intra_sat_adjacent_factor = getattr(config, "intra_sat_adjacent_factor", 0.5)
        self.inter_sat_cochannel_factor = getattr(config, "inter_sat_cochannel_factor", 0.3)
        self.inter_sat_adjacent_factor = getattr(config, "inter_sat_adjacent_factor", 0.15)
        if self.distribution == "uniform":  # 均匀分布
            self.uniform_min = config.uniform_min
            self.uniform_max = config.uniform_max
        self.GenAn()
        self.AddAn()
        # 仿真参数
        self.Sate_B = config.Sate_B                                             # 卫星总频宽(GHz)
        self.Sate_P = 10                                                        # 卫星波束总功率(dBw)/10W
        self.Beam_P_max = 5                                                     # 波束最大功率(dBW)/50W
        self.N0 = config.N0                                                     # 高斯白噪声功率谱密度
        self.noise_W = 6.918309709189322e-21
        self.Noise_P = -179.2                                                   # 单个波束噪声功率(dBW)
        self.fc = config.fc                                                     # 载波频率fc：11.7GHz
        self.RE = config.RE                                                     # 地球半径RE：6371km
        self.Gm = config.Gm                                                     # 卫星天线发射增益
        self.Gr = config.Gr                                                     # 地面接收端天线增益
        self.h0 = self.low_orbit_altitude                                       # 低轨卫星轨道高度(km)
        self.FSL = self._compute_fsl(self.h0)                                # 自由空间损耗
        self.PL = self.dBi_to_dB(self.Gm) - self.FSL + self.dBi_to_dB(self.Gr)  # 等效信道增益
        self.FPL = 7.9432823e-11
        self.satellite_path_gains = [self._compute_path_gain(h) for h in self.satellite_altitudes]
        self.satellite_noise = [self.noise_W / gain for gain in self.satellite_path_gains]
        
    def reset(self):
        self.D = np.zeros((self.T_ttl, self.N))     # 清空流量表
        self.an_lambda = np.random.rand(self.N)     # 重新初始化到达概率
        self.GenAn()
        self.AddAn()
        self.Actions = [0 for _ in range(self.K)]
        return self.get_obs()

    def GenAn(self):    # 产生到达流量需求 TODO:每次产生不同参数的分布
        if self.distribution == "uniform":
            self.An = np.random.uniform(low=self.uniform_min, high=self.uniform_max, size=self.N)  # 各小区流量需求到达率
            # dis = np.random.rand(self.N)    # 产生随机数与lambda对比，如果随机数大于lambda，则该小区本时刻的流量到达设为0
            # for index in range(self.N):
            #     if dis[index] > self.an_lambda[index]:
            #         self.An[index] = 0

    def AddAn(self):    # 加入到达流量需求
        for i in range(self.N):
            self.D[0][i] = self.An[i]

    def SetBeta(self, b):
        self.beta = b

    def DecayBeta(self):
        if self.beta > 0.1:
            self.beta -= 0.000018

    def _init_satellite_altitudes(self):
        if self.num_satellites <= 0:
            return []
        altitudes = []
        for sat_idx in range(self.num_satellites):
            if sat_idx < self.num_high_orbit:
                altitudes.append(self.high_orbit_altitude)
            else:
                altitudes.append(self.low_orbit_altitude)
        return altitudes

    def _build_low_orbit_coverages(self, low_count):
        if low_count <= 0:
            return []
        if low_count == 4:
            quadrant_filters = [
                lambda x, y: x <= 0 and y >= 0,   # 左上象限
                lambda x, y: x >= 0 and y >= 0,   # 右上象限
                lambda x, y: x <= 0 and y <= 0,   # 左下象限
                lambda x, y: x >= 0 and y <= 0,   # 右下象限
            ]
            coverages = []
            for idx in range(4):
                filt = quadrant_filters[idx]
                coverage = {cell_idx for cell_idx, (x, y) in enumerate(self.Cell) if filt(x, y)}
                coverages.append(coverage)
            return coverages
        else:
            quadrant_filters = [
                lambda x, y: x <= 0 and y >= 0,   # 左上象限
                lambda x, y: x >= 0 and y >= 0,   # 右上象限
                lambda x, y: x <= 0 and y <= 0,   # 左下象限
                lambda x, y: x >= 0 and y <= 0,   # 右下象限
            ]
            coverages = []
            for idx in range(low_count):
                filt = quadrant_filters[idx % len(quadrant_filters)]
                coverage = {cell_idx for cell_idx, (x, y) in enumerate(self.Cell) if filt(x, y)}
                coverages.append(coverage)
            return coverages

    def _init_satellite_coverages(self):
        coverages = []
        if self.num_satellites <= 0:
            return coverages
        low_coverages = self._build_low_orbit_coverages(self.num_satellites - self.num_high_orbit)
        low_idx = 0
        for sat_idx in range(self.num_satellites):
            if sat_idx < self.num_high_orbit:
                coverages.append(set(range(self.N)))
            else:
                if low_idx < len(low_coverages):
                    coverages.append(low_coverages[low_idx])
                else:
                    coverages.append(set(range(self.N)))
                low_idx += 1
        return coverages

    def _init_agent_allowed_actions(self):
        allowed = []
        if self.K <= 0:
            return allowed
        full_coverage = tuple(range(self.N))
        for agent_idx in range(self.K):
            sat_idx = self.get_satellite_index(agent_idx)
            if 0 <= sat_idx < len(self.satellite_coverages):
                coverage = self.satellite_coverages[sat_idx]
                allowed.append(tuple(sorted(coverage)) if coverage else tuple())
            else:
                allowed.append(full_coverage)
        return allowed

    def _compute_fsl(self, altitude):
        altitude = max(1e-3, altitude)
        return 32.44 + 20 * np.log10(self.fc * 1000) + 20 * np.log10(altitude)

    def _compute_path_gain(self, altitude):
        fsl = self._compute_fsl(altitude)
        pl = self.dBi_to_dB(self.Gm) - fsl + self.dBi_to_dB(self.Gr)
        return 10 ** (pl / 10)

    def get_satellite_index(self, beam_idx):
        if self.beams_per_satellite <= 0:
            return 0
        return beam_idx // self.beams_per_satellite

    def dBi_to_dB(self, dBi):
        return dBi - 2.15

    def is_cell_in_coverage(self, sat_idx, cell_idx):
        if cell_idx < 0 or cell_idx >= self.N:
            return False
        if sat_idx < 0 or sat_idx >= len(self.satellite_coverages):
            return True
        coverage = self.satellite_coverages[sat_idx]
        if not coverage:
            return False
        return cell_idx in coverage

    def get_allowed_actions_for_agent(self, agent_idx):
        if agent_idx < 0 or agent_idx >= len(self.agent_allowed_actions):
            return tuple(range(self.N))
        return self.agent_allowed_actions[agent_idx]

    def mask_actions(self, actions):
        masked_actions = []
        for agent_idx, act in enumerate(actions):
            allowed = self.get_allowed_actions_for_agent(agent_idx)
            if not allowed:
                masked_actions.append(int(act))
                continue
            action_idx = int(act)
            if action_idx not in allowed:
                action_idx = int(np.random.choice(allowed))
            masked_actions.append(action_idx)
        return masked_actions

    def Inf(self, i, j):    # 判断两个小区是否相邻：X Y差值均<=2
        a = self.Cell[i]
        b = self.Cell[j]
        if abs(a[0] - b[0]) <= 2 and abs(a[1] - b[1]) <= 2:
            return True
        return False

    def GenCn_Simple(self, action):
        Cn = np.zeros(self.N)
        for i in range(self.K):
            act_i = action[i]
            sat_i = self.get_satellite_index(i)
            if not self.is_cell_in_coverage(sat_i, act_i):
                continue
            has_interference = False
            for j in range(self.K):
                if i == j:
                    continue
                act_j = action[j]
                sat_j = self.get_satellite_index(j)
                if not self.is_cell_in_coverage(sat_j, act_j):
                    continue
                if act_i == act_j:
                    if sat_i == sat_j and self.intra_sat_cochannel_factor > 0:
                        has_interference = True
                    elif sat_i != sat_j and self.inter_sat_cochannel_factor > 0:
                        has_interference = True
                elif self.Inf(act_i, act_j):
                    if sat_i == sat_j and self.intra_sat_adjacent_factor > 0:
                        has_interference = True
                    elif sat_i != sat_j and self.inter_sat_adjacent_factor > 0:
                        has_interference = True
                if has_interference:
                    break
            if not has_interference:
                Cn[act_i] = 300
        return Cn

    def GenCn(self, action, power):    # 计算有波束照射的小区的最大传输量
        Cn = np.zeros(self.N)
        for i in range(self.K):
            act_i = action[i]
            sat_i = self.get_satellite_index(i)
            if not self.is_cell_in_coverage(sat_i, act_i):
                continue
            path_gain_i = self.satellite_path_gains[sat_i] if sat_i < len(self.satellite_path_gains) else self.FPL
            noise_i = self.noise_W / self.FPL
            power_rec_i = power[i]
            power_inf = 0
            for j in range(self.K):
                if i == j:
                    continue
                act_j = action[j]
                sat_j = self.get_satellite_index(j)
                if not self.is_cell_in_coverage(sat_j, act_j):
                    continue
                eff_power_j = power[j]
                if act_j == act_i:
                    if sat_i == sat_j:
                        power_inf += eff_power_j * self.intra_sat_cochannel_factor
                    else:
                        power_inf += eff_power_j * self.inter_sat_cochannel_factor
                elif self.Inf(act_i, act_j):
                    if sat_i == sat_j:
                        power_inf += eff_power_j * self.intra_sat_adjacent_factor
                    else:
                        power_inf += eff_power_j * self.inter_sat_adjacent_factor
            SINR = power_rec_i / (noise_i + power_inf)
            Cn[act_i] = self.Sate_B * np.log2(1 + SINR) * 18    
            # 计算得到的数值单位是Gbps，需要乘以1000（因为GHz->MHz），再乘以0.02（每一个时隙的时间）
        return Cn

    def CalDn(self):  # 各小区总流量需求
        Dn = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            for j in range(self.T_ttl):
                Dn[i] += self.D[j, i]
        return Dn

    def GetTao(self):
        tao = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            tao_l = 0
            tao_total = 0
            for l in range(self.T_ttl):
                tao_l += l * self.D[l, i]
                tao_total += self.D[l, i]
            if tao_total == 0:
                tao[i] = 0
            else:
                tao[i] = tao_l / tao_total
        return tao

    def GetF(self):     # 时延公平性 | 平均时延
        tao = self.GetTao()
        return np.max(tao) - np.min(tao), np.average(tao)

    def get_max_lenth(self):
        return max(np.count_nonzero(self.D, axis=0))

    def SetAction(self, action):
        self.Actions = action

    def get_obs(self):
        obs = self.D[0:10, :].ravel(order='f') / self.uniform_max
        return obs

    def get_state(self):
        state = self.D[0:10, :].ravel(order='f') / self.uniform_max
        for act in self.Actions:
            act_dis = np.zeros(self.N)
            act_dis[act] = 1
            state = np.concatenate((state, act_dis))
        return state

    def step(self, action, step):
        self.SetAction(action)
        power = [2.5 for _ in range(self.K)]
        Cn = self.GenCn(action, power)
        Dn = self.CalDn()
        a = sum(Cn)
        b = sum(self.An)

        # 流量传输
        Th = sum(np.minimum(Dn, Cn))  # 总传输量
        for i in range(self.N):
            if Dn[i] == 0 or Cn[i] == 0:    # 小区没有流量需求或波束没有覆盖，不操作
                continue
            elif Cn[i] >= Dn[i]:            # 等待的流量全部传输完毕
                for j in range(self.T_ttl):
                    self.D[j, i] = 0
                Cn[i] -= Dn[i]
            else:                           # 不能全部传输，从等待时间最长的流量开始计算
                for j in range(self.T_ttl):
                    if self.D[self.T_ttl - j - 1, i] == 0:
                        continue
                    if Cn[i] >= self.D[self.T_ttl - j - 1, i]:
                        Cn[i] -= self.D[self.T_ttl - j - 1, i]
                        self.D[self.T_ttl - j - 1, i] = 0
                    else:
                        self.D[self.T_ttl - j - 1, i] -= Cn[i]
                        Cn[i] = 0

        # 统计丢包，并加入新的流量需求
        dump = 0
        for o in self.D[self.T_ttl - 1]:  # 统计丢弃流量，当前时刻等待时间==T_ttl的流量将被丢弃
            dump += o
        for i in range(self.T_ttl - 1):  # 流量需求表中的元素向下移动一个单位，时刻+1
            for j in range(self.N):
                self.D[self.T_ttl - i - 1][j] = self.D[self.T_ttl - i - 2][j]
        self.GenAn()
        self.AddAn()    # 加入新到来的流量需求

        # 计算公平性指标，供奖励函数和日志使用
        F, average_tao = self.GetF()

        # 获取观测
        next_obs = self.get_obs()
        # if Th>=600 and Th<620:
        #     reward = 0.1
        # elif Th>=620 and Th<640:
        #     reward = 0.15
        # elif Th>=640 and Th<660:
        #     reward = 0.2
        # elif Th>=660 and Th<680:
        #     reward = 0.25
        # elif Th >= 680 and Th<700:
        #     reward = 0.3
        # elif Th>=700 and Th<720:
        #     reward = 0.35
        # elif Th>=720 and Th<740:
        #     reward = 0.4
        # elif Th>=740 and Th<760:
        #     reward = 0.45
        # elif Th>=760 and Th<780:
        #     reward = 0.5
        # elif Th>=780 and Th<800:
        #     reward = 0.55
        # elif Th>=800 and Th<810:
        #     reward = 0.6
        # elif Th>=810 and Th<820:
        #     reward = 0.7
        # elif Th>=820 and Th<830:
        #     reward = 0.9
        # elif Th>=830 and Th<840:
        #     reward = 1.0
        # elif Th>=840 and Th<850:
        #     reward = 1.1
        # elif Th>=840 and Th<850:
        #     reward = 1.2
        # elif Th>=850 and Th<860:
        #     reward = 1.3
        # elif Th>=850 and Th<860:
        #     reward = 1.4
        # elif Th>=860 and Th<870:
        #     reward = 1.5
        # elif Th>=860 and Th<870:
        #     reward = 1.6
        # elif Th>=870 and Th<880:
        #     reward = 1.7
        # elif Th>=870 and Th<880:
        #     reward = 1.8
        # elif Th>=880 and Th<890:
        #     reward = 1.9
        # elif Th>=890 and Th<900:
        #     reward = 2.0
        # elif Th>=900:
        #     reward = 3.0    
        # elif Th<600 and Th>=590:
        #     reward = -0.1
        # elif Th<590 and Th>=580:
        #     reward = -0.3
        # elif Th<580 and Th>=560:
        #     reward = -0.5
        # elif Th<560 and Th>=540:
        #     reward = -1.0
        # elif Th<540 and Th>=520:
        #     reward = -1.5
        # elif Th<520 and Th>=500:
        #     reward = -1.9
        # elif Th<500 and Th>=480:
        #     reward = -2.3
        # elif Th<480 :
        #     reward = -3.0
        reward = Th/50 - 10*dump
        

        terminate = True if step == self.Episode_Length else False
        # 奖励 | 传输流量 | 丢包 | 富余传输量 | 时延公平性 | 平均时延
        return next_obs, [reward], [terminate], Th, dump, sum(Cn), average_tao, F

    def render(self, step):
        # 设置小区边长
        side_length = 1

        # 定义正六边形顶点坐标
        hexagon_vertices = np.array([
            [0, side_length],
            [np.sqrt(3) / 2 * side_length, side_length / 2],
            [np.sqrt(3) / 2 * side_length, -side_length / 2],
            [0, -side_length],
            [-np.sqrt(3) / 2 * side_length, -side_length / 2],
            [-np.sqrt(3) / 2 * side_length, side_length / 2]
        ])

        # 定义每一行的小区数量
        row_counts = [4, 5, 6, 7, 6, 5, 4]
        # row_counts = [5, 6, 5, 6, 5]
        # row_counts = [3, 4, 5, 4, 3]

        # 创建画布，并指定tight_layout=True
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

        # 绘制小区
        Dn = self.CalDn()
        cell_id = 0
        y_offset = -4.5 * side_length
        for row_count in row_counts:
            x_offset = -((row_count - 1) / 2) * np.sqrt(3) * side_length
            for _ in range(row_count):
                hexagon = plt.Polygon(hexagon_vertices + [x_offset, y_offset], closed=True, edgecolor='black', fill=True)

                # 设置小区的颜色深度和智能体选择的颜色
                if cell_id in self.Actions:
                    agent_index = np.where(self.Actions == cell_id)[0][0]   # 获取智能体在self.Actions中的索引
                    agent_color = self.AgentRGBColors[agent_index % len(self.AgentColors)]  # 获取对应智能体的颜色
                    # hexagon.set_facecolor(agent_color)
                    hexagon.set_facecolor(plt.cm.Blues(0.5))
                else:
                    color_depth = Dn[cell_id] / (self.uniform_max * self.T_ttl)  # 根据流量需求值计算灰度深度
                    hexagon.set_facecolor(plt.cm.Reds(color_depth))  # 设置小区的填充颜色为RGB颜色值
                cell_id += 1
                ax.add_patch(hexagon)
                x_offset += np.sqrt(3) * side_length
            y_offset += 1.5 * side_length

        # 设置坐标轴范围
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)

        # 隐藏坐标轴
        ax.axis('off')

        rcParams['pdf.fonttype'] = 42  # 设置PDF输出的字体类型
        # 生成PDF矢量图
        filename = f"render/step_{step}.pdf"
        fig.savefig(filename, format='pdf', dpi=300)

        # 保存图像为文件
        filename = f"render/step_{step}.png"
        plt.savefig(filename)  # 将图像保存为名为'visualization.png'的文件
        plt.close()
