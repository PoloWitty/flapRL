# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import pdb
import random
import copy
import math
import time
from distutils.util import strtobool
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import flappy_bird_gymnasium
from gymnasium.wrappers import TransformReward

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--rounding",type=int, default=1,
        help='discretize the oberservation space according to the rounding num')

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="FlappyBird-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args

class ProcessObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Process observation space for the flappy bird env.
    """

    def __init__(self, env: gym.Env, rounding=0):
        """A wrapper that process observations for the flappy bird env.

        Args:
            env: The environment to apply the wrapper

        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
                    -np.inf, np.inf, shape=(4,), dtype=np.float32
        )
        self.rounding = rounding

    def discretize(self, num):
        '''
        Discretizes the input num base on the value rounding.
        
        Args:
            num (int): An input value.
            
        Returns:
            int: A discretized output value.
        '''
        if self.rounding==0:
            return num
        else:
            return self.rounding * math.floor(num / self.rounding)

    def observation(self, observation):
        """Process the observations.

        Args:
            observation: The observation to process

        Returns:
            The processed observations
        """
        hor_dist_to_next_pipe = observation[3] # ['next_pipe_dist_to_player']
        ver_dist_to_next_pipe_down = observation[5]-observation[9] # observation['next_pipe_bottom_y'] - observation['player_y']
        ver_dist_to_next_pipe_top = observation[4]-observation[9] # observation['next_pipe_bottom_y'] - observation['player_y']
        player_vel = observation[10]

        # state = []
        # state.append('player_vel' + ' ' + str(observation[10]))
        # state.append('hor_dist_to_next_pipe' + ' ' + str(hor_dist_to_next_pipe))
        # state.append('ver_dist_to_next_pipe' + ' ' + str(ver_dist_to_next_pipe))

        # processed_observation = {
        #     'hor_dist_to_next_pipe': hor_dist_to_next_pipe,
        #     'ver_dist_to_next_pipe_down': ver_dist_to_next_pipe_down,
        #     'ver_dist_to_next_pipe_top': ver_dist_to_next_pipe_top,
        #     'player_vel': player_vel
        #     }

        return np.array([
            self.discretize(hor_dist_to_next_pipe),
            self.discretize(ver_dist_to_next_pipe_down),
            self.discretize(ver_dist_to_next_pipe_top),
            player_vel
        ],dtype=np.float32)
        # return np.array([self.discretize(o) for o in observation], dtype=np.float32)


def make_env(env_id, seed, idx, capture_video, run_name, rounding=0):
    def thunk():
        if capture_video and idx == 0:
            env = ProcessObservation(gym.make(env_id, render_mode="rgb_array"),rounding=rounding) # (512, 288, 3) when render()
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = ProcessObservation(gym.make(env_id),rounding=rounding)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

# copy from https://github.com/vwxyzjn/cleanrl/blob/7e24ae238eab6a8e7efbbf452cb4a8922bcda73f/cleanrl_utils/evals/dqn_eval.py
def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
    rounding: int = 1
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name, rounding=rounding)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.hor_dist_net = nn.Sequential(
            GaussianRBF(32,cutoff=512,start=0),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self._ver_dist_transform = GaussianRBF(32,cutoff=288,start=-288)# 2 ver dist
        self.ver_dist_net = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.vel_net = nn.Sequential(
            GaussianRBF(32,cutoff=100,start=-100), # 1 vertical vel 
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.att = nn.Sequential(
            nn.Linear(16*3, 24),
            nn.ReLU(),
            nn.Linear(24,12),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(12, env.single_action_space.n),
        )

    def forward(self, x):
        # hor
        hor_x = self.hor_dist_net(x[:,0]) 
        # ver
        ver_x_down = self._ver_dist_transform(x[:,1])
        ver_x_top = self._ver_dist_transform(x[:,2])
        ver_x = torch.concat([ver_x_down,ver_x_top],dim=-1)
        ver_x = self.ver_dist_net(ver_x)
        # vel
        vel_x = self.vel_net(x[:,3])

        x = torch.concat((hor_x,ver_x,vel_x),dim=-1)
        x = self.att(x)
        x = self.out(x)
        return x

# class QNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(np.array(env.single_observation_space.shape).prod(), 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, env.single_action_space.n),
#         )

#     def forward(self, x):
#         return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.rounding) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
            rounding=args.rounding
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from huggingface_sb3 import load_from_hub, package_to_hub
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

            # Define the name of the environment
            env_id = args.env_id

            # Create the evaluation env and set the render_mode="rgb_array"
            eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

            # Define the model architecture we used
            model_architecture = "DQN"

            ## Define the commit message
            commit_message = f"{model_architecture} on {env_id} env, first commit"

            # method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
            package_to_hub(model=q_network, # Our trained model
               model_name=repo_name, # The name of our trained model
               model_architecture=model_architecture, # The model architecture we used: in our case PPO
               env_id=env_id, # Name of the environment
               eval_env=eval_env, # Evaluation Environment
               repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
               commit_message=commit_message)

    envs.close()
    writer.close()