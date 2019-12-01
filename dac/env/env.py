import gym
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
#from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.vec_env.vec_normalize import VecNormalize

# np.random.seed(1)
# torch.manual_seed(np.random.randint(int(1e6)))

# class NormalizedEnv(gym.Wrapper):
#     def __init__(self, env_fn, count=1e-8, clip=10.0):
#         super().__init__(env_fn())
#         self.env = env_fn()
#         self.action_space = self.env.action_space
#         self.observation_space = self.env.observation_space
#         self.running_mean_std = RunningMeanStd()
#         self.count = count
#         self.clip = clip

#     def reset(self):
#         return self.env.reset()

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)

#         obs = np.asarray(obs)
#         self.running_mean_std.update(obs)
#         obs = np.clip((obs - self.running_mean_std.mean) / np.sqrt(self.running_mean_std.var + self.count),
#                        -self.clip, self.clip)
#         return obs, reward, done, info


# def env_fn(env_name):
#     def _make():
#         return gym.make(env_name)

#     return _make


def make_env(env_name: str, rank=0):
    def _thunk():
        env = gym.make(env_name)
#        env.seed(1)
        env.seed(np.random.randint(1e5)+rank)
        return env

    return _thunk

