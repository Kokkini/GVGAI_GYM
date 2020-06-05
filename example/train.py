import gym_gvgai
import os
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.common.atari_wrappers import WarpFrame
from stable_baselines import DQN  # test arbitrary agent
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--env', help='environment ID', type=str, default='gvgai-golddigger-lvl0-v0')
parser.add_argument('--seed', help='RNG seed', type=int, default=None)
parser.add_argument('--num_timesteps', help='', type=float, default=1e7)
parser.add_argument('--save_video_interval', help='Save video every x episodes (0 = disabled)', default=10, type=int)
parser.add_argument('--log_path', help='Path to save log to', default='data/logs', type=str)
parser.add_argument('--batch_size', help='batch size for both pretraining and training', type=int, default=32)
parser.add_argument('--buffer_size', help='experience replay buffer size', type=float, default=1e6)
parser.add_argument('--exploration_fraction',
                    help='anneal exploration epsilon for this fraction of total training steps', type=float,
                    default=0.1)
parser.add_argument('--exploration_final_eps', help='exploration epsilon after annealing', type=float, default=0.1)
parser.add_argument('--train_freq', help='train every x frames', type=int, default=1)
parser.add_argument('--prioritized', help='1: prioritized replay, 0: none', type=int, default=0)
parser.add_argument('--double_q', help='1: use double q, 0: none', type=int, default=0)
args = parser.parse_args()

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN and others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy training performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True


# Create log dir
log_dir = args.log_path
os.makedirs(log_dir, exist_ok=True)

env = gym_gvgai.make(args.env)
env = WarpFrame(env)
if args.save_video_interval != 0:
    env = Monitor(env, log_dir, allow_early_resets=True, video_callable=(lambda ep: ep % args.save_video_interval == 0), force=True)
else:
    env = Monitor(env, log_dir, allow_early_resets=True)

model = DQN(CnnPolicy, env,
            verbose=1,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_eps,
            tensorboard_log="tensorboard_log",
            prioritized_replay=bool(args.double_q),
            double_q=bool(args.double_q),
            buffer_size= int(args.buffer_size),
            train_freq=args.train_freq,
            batch_size=args.batch_size,
            seed=args.seed)

model.learn(total_timesteps=int(args.num_timesteps), callback=callback)
