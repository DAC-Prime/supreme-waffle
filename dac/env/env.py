import gym
import numpy as np
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.vec_env.vec_normalize import VecNormalize

class NormalizedEnv(gym.Wrapper):
    def __init__(self, env_fn, count=1e-8, clip=10.0):
        super().__init__(env_fn())
        self.env = env_fn()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.running_mean_std = RunningMeanStd()
        self.count = count
        self.clip = clip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs = np.asarray(obs)
        self.running_mean_std.update(obs)
        obs = np.clip((obs - self.running_mean_std.mean) / np.sqrt(self.running_mean_std.var + self.count),
                       -self.clip, self.clip)
        return obs, reward, done, info


def env_fn(env_name):
    def _make():
        return gym.make(env_name)

    return _make


def make_env(env_name: str):
    return NormalizedEnv(env_fn(env_name))
