import argparse
import math
import torch
from environment import Satellite as Env
from MADDPG.maddpg_main import maddpg_train, maddpg_test, power_allocate


def train(config):
    env = Env(config)
    maddpg_train(config, env)


def test(config):
    env = Env(config)
    for i in range(140):
        env.uniform_max = float(i) + 10
        maddpg_test(config, env)


def power(config):
    env = Env(config)
    power_allocate(config, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Environment Settings
    parser.add_argument("--N", default=37, type=int, help="Number of cells")
    parser.add_argument("--K", default=10, type=int, help="Total number of cooperative beams")
    parser.add_argument("--num_satellites", default=5, type=int, help="Number of cooperating satellites")
    parser.add_argument("--num_high_orbit", default=1, type=int, help="Number of high-orbit satellites")
    parser.add_argument('--T_ttl', type=int, default=10, help='Packet delay threshold')
    parser.add_argument("--Episode_Length", default=50, type=int, help="Time steps per episode")
    parser.add_argument("--distribution", default="uniform", choices=["uniform"], type=str, help="Distribution of arrival traffic")
    parser.add_argument("--uniform_min", default=0.0, type=float)
    parser.add_argument("--uniform_max", default=50.0, type=float)
    parser.add_argument('--Sate_B', type=float, default=2, help='卫星频宽2GHz')
    parser.add_argument('--Sate_P', type=float, default=33, help='卫星总功率33dBW(20W)')
    parser.add_argument('--Beam_P_max', type=float, default=30, help='波束最大功率限制30dBW(10W)')
    parser.add_argument('--T_slot', type=float, default=0.1, help='每个时隙0.02s')
    parser.add_argument('--Gm', type=float, default=40.3, help='Maximum transmit antenna gain Gm(dBi)')
    parser.add_argument('--Gr', type=float, default=31.6, help='Terminal antenna gain Gr(dBi)')
    parser.add_argument('--fc', type=float, default=11.7, help='band frequency(GHz)')
    parser.add_argument('--h0', type=float, default=550, help='低轨卫星轨道高度(km)')
    parser.add_argument('--high_orbit_altitude', type=float, default=750, help='高轨卫星轨道高度(km)')
    parser.add_argument('--RE', type=float, default=6371, help='地球半径(km)')
    parser.add_argument('--N0', type=float, default=-171.6, help='Noise power spectral density N0(dBm/Hz)')
    parser.add_argument('--inter_sat_cochannel_factor', type=float, default=0.3,
                        help='Scaling factor for co-channel interference between satellites')
    parser.add_argument('--inter_sat_adjacent_factor', type=float, default=0.15,
                        help='Scaling factor for adjacent-cell interference between satellites')

    # Train Settings
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument("--warmup", default=True, type=bool, help="Warmup before training")
    parser.add_argument("--warmup_episode", default=500, type=int, help="Warmup episode")
    parser.add_argument("--total_episode", default=10000, type=int, help="Total episode for training")
    parser.add_argument("--replay_buffer_size", default=2_000_000, type=int, help="Replay buffer size (steps)")
    parser.add_argument("--train_episode", default=1, type=int, help="Train every % episodes")
    parser.add_argument("--train_repeat", default=5, type=int, help="Repeat times per training")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epsilon_begin", default=1, type=float)
    parser.add_argument("--epsilon_end", default=0.05, type=float)
    parser.add_argument("--epsilon_steps", default=500000, type=int)
    parser.add_argument("--beta", default=1.0, type=float)
    # Network Settings
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--qmix_hidden_dim", default=64, type=int)
    parser.add_argument("--hyper_hidden_dim", default=128, type=int)
    parser.add_argument("--two_hyper_layers", default=False, type=bool)

    parser.add_argument("--critic_lr", default=5e-4, type=float, help="Learning rate for AC")
    parser.add_argument("--actor_lr", default=5e-4, type=float, help="Learning rate for AC")
    parser.add_argument('--optimizer', default="Adam", choices=["Adam", "RMS"], type=str)
    parser.add_argument("--gradient_clip", default=1.0, type=float)
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounted factor")
    parser.add_argument("--hard_update_episode", default=40, type=int)
    parser.add_argument("--soft_update_size", default=0.01, type=float)
    parser.add_argument("--policy_regulariser", default=0.001, type=float)

    # Gradient Estimation hyperparams
    parser.add_argument("--gradient_estimator", default="tags", choices=["stgs", "grmck", "gst", "tags"], type=str)
    parser.add_argument("--gumbel_temp", default=0.01, type=float)
    parser.add_argument("--rao_k", default=1, type=int)  # For GRMCK
    parser.add_argument("--gst_gap", default=1.0, type=float)  # For GST
    parser.add_argument("--tags_start", default=5.0, type=float)  # For TAGS
    parser.add_argument("--tags_end", default=0.05, type=float)  # For TAGS
    parser.add_argument("--tags_period", default=500000, type=int)  # For TAGS

    # Evaluate Settings
    parser.add_argument("--eval_episode", default=50, type=int, help="Evaluate every % episodes")
    parser.add_argument("--eval_repeat", default=10, type=int, help="Repeat times per evaluating")



    # Test Settings
    parser.add_argument("--test", default=True, type=bool, help="Test or Train")
    parser.add_argument("--render", default=False, type=bool, help="Render during last eval")

    # Save & Load Settings
    parser.add_argument("--save_episode", default=200, type=int, help="Save model every % episodes")
    parser.add_argument("--load", default=False, type=bool, help="Load model")
    parser.add_argument("--save_dir", default="save5", type=str, help="Save path")
    parser.add_argument("--model_episode", default="9800", type=str, help="Model path")

    # Power Allocate
    parser.add_argument("--power_allocate", default=False, type=bool)

    config = parser.parse_args()
    config.num_satellites = max(1, config.num_satellites)
    config.num_high_orbit = max(0, min(config.num_high_orbit, config.num_satellites))
    config.low_orbit_altitude = config.h0
    
    config.beams_per_satellite = 2
    config.total_beams = int(config.beams_per_satellite * config.num_satellites)
    config.K = config.total_beams
    config.n_agents = config.total_beams
    config.act_dim = config.N
    config.obs_dim = config.T_ttl * config.N
    config.state_dim = config.obs_dim + config.act_dim * config.total_beams
    config.epsilon_decrease = (config.epsilon_begin - config.epsilon_end) / config.epsilon_steps
    config.test = False
    config.load = True

    if config.cuda and not torch.cuda.is_available():
        print("CUDA 未检测到，自动切换到 CPU 运行。")
        config.cuda = False

    if config.power_allocate:

        power(config)
    else:
        if not config.test:
            train(config)
        else:
            test(config)
