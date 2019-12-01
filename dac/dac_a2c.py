import os
import datetime
import logging
from tqdm import tqdm
from enum import Enum

from network import *
from env.env import *
import torch
import numpy as np

def cat(x, size):
    x = x[:size]
    return torch.cat(x).detach()


class MdpType(Enum):
    high = "high"
    low = "low"


class DACA2CAgent:
    def __init__(self, env_name, device, steps=2e6):
        # for a2c only
        self.entropy_weight = 0.01

        # config
        self.num_workers = 1
        self.discount = 0.99
        self.use_gae = True
        self.gae_tau = 0.95
        self.optimization_epochs = 5
        # self.optimization_epochs = 50
        self.ppo_clip_param = 0.2

        self.cur_steps = 0
        self.max_steps = steps
        self.num_steps = 2048
        # self.num_steps = 512
        self.env_name = env_name
        self.num_envs = 1
        self.num_options = 4
        self.mini_batch_size = 64
        # self.mini_batch_size = 32
        
        self.is_init_states = tensor(np.ones((self.num_workers))).byte()
        self.worker_index = tensor(np.arange(self.num_workers)).long()
        self.prev_options = tensor(np.zeros(self.num_workers)).long()

        train_envs = [make_env(self.env_name) for i in range(self.num_envs)]
        train_envs = SubprocVecEnv(train_envs)
        self.envs = VecNormalize(train_envs, ret=False)
        eval_env = [make_env(self.env_name)]
        eval_env = DummyVecEnv(eval_env)
        self.eval_env = VecNormalize(eval_env, ret=False)

        obs_dim = self.eval_env.observation_space.shape[0]
        action_dim = self.eval_env.action_space.shape[0]
        self.dac_net = DACNetwork(obs_dim, action_dim, self.num_options, device=device)
        self.opt = optim.Adam(self.dac_net.parameters(),
                                    lr=3e-4,
                                    eps=1e-5)

        path='./data/{}'.format(self.env_name)
        if not os.path.exists(path):
            os.makedirs(path)
        # curtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        curtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_{:04d}".format(random.randint(1,9999))
        self.train_logger = logging.getLogger("train")
        self.eval_logger = logging.getLogger("eval")
        self.train_logger.setLevel(logging.INFO)
        self.eval_logger.setLevel(logging.INFO)
        self.train_logger.addHandler(logging.FileHandler("{path}/dac_ppo_train_{curtime}.log".format(path=path, curtime=curtime), "w"))
        self.eval_logger.addHandler(logging.FileHandler("{path}/dac_ppo_eval_{curtime}.log".format(path=path, curtime=curtime), "w"))


    def compute_pi_h(self, prediction, pre_option, is_init_states):
        master_policy = prediction["master_policy"]
        beta = prediction["beta"]

        mask = torch.zeros_like(master_policy)
        mask[self.worker_index, pre_option] = 1

        # pi_h = beta * master_policy + (1 - beta) * mask
        is_init_states = is_init_states.view(-1, 1).expand(-1, master_policy.size(1))
        pi_h = torch.where(is_init_states, master_policy, beta * master_policy + (1 - beta) * mask)
        # print("pi_h %s" % pi_h)

        return pi_h


    def compute_pi_l(self, options, action, mean, std):
        options = options.unsqueeze(-1).expand(-1, -1, mean.size(-1))
        mean = mean.gather(1, options).squeeze(1)
        std = std.gather(1, options).squeeze(1)
        normal_dis = torch.distributions.Normal(mean, std)
        
        pi_l = normal_dis.log_prob(action).sum(-1).exp().unsqueeze(-1)

        return pi_l


    def compute_log_pi(self, mdp_type, pi_h, options, action, mean, std):
        if mdp_type == MdpType.high:
            return pi_h.add(1e-5).log().gather(1, options)
        elif mdp_type == MdpType.low:
            pi_l = self.compute_pi_l(options, action, mean, std)
            return pi_l.add(1e-5).log()
        else:
            raise NotImplementedError


    def compute_adv(self, values, rewards, dones):
        advantanges = [None] * self.num_steps
        returns = [None] * self.num_steps

        ret = values[-1].detach() # TODO: or this
        adv = tensor(np.zeros((self.num_workers, 1)))
        for i in reversed(range(self.num_steps)):
            ret = rewards[i] + self.discount * (1 - dones[i]) * ret # TODO: or this
            td_error = rewards[i] + self.discount * (1 - dones[i]) * values[i + 1] - values[i] # td_error ?
            adv = td_error + self.discount * self.gae_tau * (1 - dones[i]) * adv
            # ret = adv + values[i] # TODO: maybe not this

            advantanges[i] = adv.detach()
            returns[i] = ret.detach()

        return advantanges, returns


    def random_iter(self, *argv):
        batch_size = argv[0].size(0)
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield (cache[rand_ids, :] for cache in argv)


    def permut_iter(self, *argv):
        batch_size = argv[0].size(0)
        indices = np.asarray(np.random.permutation(np.arange(batch_size)))
        batches = indices[:batch_size // self.mini_batch_size * self.mini_batch_size].reshape(-1, self.mini_batch_size)
        for batch in batches:
            # yield batch
            # print("batch: %s" % batch)
            yield (cache[batch, :] for cache in argv)
        r = len(indices) % self.mini_batch_size
        if r:
            # yield indices[-r:]
            batch = indices[-r:]
            # print("batch: %s" % batch)
            yield (cache[batch, :] for cache in argv)

    def learn(self,
              mdp_type, 
              log_pi,#
              value, # v
              returns, #
              advs, #
              pi_h #
              ):

        # print("learning")
        advs = (advs - advs.mean()) / advs.std() # not sure

        if mdp_type == MdpType.high:
            entropy = -(pi_h * pi_h.add(1e-5).log()).sum(-1).mean()
        elif mdp_type == MdpType.low:
            entropy = 0
        else:
            raise NotImplementedError

        # policy_loss = -torch.min(objective, objective_clipped).mean() # + 0.01 * entropy
        # value_loss = 0.5 * (sample_returns - value).pow(2).mean()
        policy_loss = -(log_pi * advs.detach()).mean() - self.entropy_weight * entropy # detach() ?
        value_loss = (value - returns.detach()).pow(2).mul(0.5).mean()

        loss = policy_loss + value_loss
        return loss


    # use code from ppo.py here
    def actual_run(self, progress_bar):
        # print("running")
        states = self.envs.reset()
        # print("train state shape {}".format(states.shape))
        # storage = Storage(config.rollout_length, ['adv_bar', 'adv_hat', 'ret_bar', 'ret_hat'])
        cumu_rewd = np.zeros(self.num_envs)

        while self.cur_steps <= self.max_steps:
            rewards_cache = []
            states_cache = []
            dones_cache = []
            actions_cache = []
            options_cache = []
            prev_options_cache = []
            init_states_cache = []
            pi_h_cache = []
            log_pi_h_cache = []
            log_pi_l_cache = []
            value_h_cache = []
            value_l_cache = []
            means_cache = []
            stds_cache = []

            for _ in range(self.num_steps):
                # print(self.cur_steps)
                prediction = self.dac_net(states)
                pi_h = self.compute_pi_h(prediction, self.prev_options, self.is_init_states)
                options = torch.distributions.Categorical(probs = pi_h).sample()

                # maybe need add log here

                mean = prediction['mean'][self.worker_index, options]
                std = prediction['std'][self.worker_index, options]
                actions = torch.distributions.Normal(mean, std).sample()

                pi_l = self.compute_pi_l(options.unsqueeze(-1), actions, prediction["mean"], prediction["std"])

                value_h = (prediction["q_option"] * pi_h).sum(-1).unsqueeze(-1)
                value_l = prediction["q_option"].gather(1, options.unsqueeze(-1))

                # print("train action shape {}".format(actions.shape))
                #next_state, reward, done, info = self.envs.step(actions) # done: terminated

                # seems info is not needed
                next_state, reward, done, info = self.envs.step(actions) # done: terminated

                cumu_rewd += reward
                for i in range(self.num_envs):
                    if done[i]:
                        print("Cumulative reward at step {cur_steps} is {reward}".format(cur_steps=self.cur_steps, reward=cumu_rewd[i]))
                        self.train_logger.info("{cur_steps} {reward}".format(cur_steps=self.cur_steps, reward=cumu_rewd[i]))
                        cumu_rewd[i] = 0

                rewards_cache.append(tensor(reward).unsqueeze(-1))
                states_cache.append(tensor(states))
                dones_cache.append(tensor(done).unsqueeze(-1))
                actions_cache.append(actions)
                options_cache.append(options.unsqueeze(-1))
                prev_options_cache.append(self.prev_options.unsqueeze(-1))
                init_states_cache.append(self.is_init_states.unsqueeze(-1))
                pi_h_cache.append(pi_h)
                log_pi_h_cache.append(pi_h[self.worker_index, options].add(1e-5).log().unsqueeze(-1))
                log_pi_l_cache.append(pi_l.add(1e-5).log())
                value_h_cache.append(value_h)
                value_l_cache.append(value_l)
                means_cache.append(prediction["mean"])
                stds_cache.append(prediction["std"])

                self.is_init_states = tensor(done).byte()
                self.prev_options = options
                states = next_state
                self.cur_steps += self.num_workers
                progress_bar.update(self.num_workers)
            
            prediction = self.dac_net(states)
            pi_h = self.compute_pi_h(prediction, self.prev_options, self.is_init_states)
            options = torch.distributions.Categorical(probs = pi_h).sample()

            value_h = (prediction["q_option"] * pi_h).sum(-1).unsqueeze(-1)
            value_l = prediction["q_option"].gather(1, options.unsqueeze(-1))

            value_h_cache.append(value_h)
            value_l_cache.append(value_l)
            means_cache.append(prediction["mean"])
            stds_cache.append(prediction["std"])

            # computer advantange
            adv_h_cache, returns_h_cache = self.compute_adv(value_h_cache, rewards_cache, dones_cache)
            # print("adv_h_cache len %s" % len(adv_h_cache))
            adv_l_cache, returns_l_cache = self.compute_adv(value_l_cache, rewards_cache, dones_cache)

            rewards_cache = cat(rewards_cache, self.num_steps)
            states_cache = cat(states_cache, self.num_steps)
            dones_cache = cat(dones_cache, self.num_steps)
            actions_cache = cat(actions_cache, self.num_steps).detach()
            options_cache = cat(options_cache, self.num_steps)
            prev_options_cache = cat(prev_options_cache, self.num_steps)
            init_states_cache = cat(init_states_cache, self.num_steps)
            pi_h_cache = cat(pi_h_cache, self.num_steps).detach()
            log_pi_h_cache = cat(log_pi_h_cache, self.num_steps).detach()
            log_pi_l_cache = cat(log_pi_l_cache, self.num_steps).detach()
            value_h_cache = cat(value_h_cache, self.num_steps)
            value_l_cache = cat(value_l_cache, self.num_steps)
            means_cache = cat(means_cache, self.num_steps).detach()
            stds_cache = cat(stds_cache, self.num_steps).detach()
            adv_h_cache = cat(adv_h_cache, self.num_steps)
            returns_h_cache = cat(returns_h_cache, self.num_steps)
            adv_l_cache = cat(adv_l_cache, self.num_steps)
            returns_l_cache = cat(returns_l_cache, self.num_steps)

            mdp_types = [MdpType.high, MdpType.low]
            np.random.shuffle(mdp_types)

            # modified to adapt the parameter of dac_a2c learn function
            def helper(mdp_type):
                if mdp_type == MdpType.high:
                    return log_pi_h_cache, value_h_cache, returns_h_cache, adv_h_cache
                else:
                    return log_pi_l_cache, value_l_cache, returns_l_cache, adv_l_cache

            log_pi_cache, value_cache, returns_cache, adv_cache = helper(mdp_types[0])
            self.learn(
                mdp_types[0],
                log_pi_cache,
                value_cache,
                returns_cache,
                adv_cache,
                pi_h_cache
                )

            log_pi_cache, value_cache, returns_cache, adv_cache = helper(mdp_types[1])
            self.learn(
                mdp_types[1],
                log_pi_cache,
                value_cache,
                returns_cache,
                adv_cache,
                pi_h_cache
                )

            eval_reward = np.mean([self.test_env() for _ in range(10)])
            print("Evaluation reward at step {cur_steps} is {reward}".format(cur_steps=self.cur_steps, reward=eval_reward))
            self.eval_logger.info("{cur_steps} {reward}".format(cur_steps=self.cur_steps, reward=eval_reward))


    def test_env(self, vis=False):
        state = self.eval_env.reset()
        if vis:
            self.eval_env.render()

        is_init_states_test = tensor(np.ones((self.num_workers))).byte()
        prev_options_test = tensor(np.zeros(self.num_workers)).long()
        done = False
        total_reward = 0
        while not done:
            # print("eval state shape {}".format(state.shape))

            prediction = self.dac_net(state)
            pi_h = self.compute_pi_h(prediction, prev_options_test, is_init_states_test)
            options = torch.distributions.Categorical(probs = pi_h).sample()
            mean = prediction['mean'][self.worker_index, options]
            std = prediction['std'][self.worker_index, options]
            action = torch.distributions.Normal(mean, std).sample()

            next_state, reward, done, _ = self.eval_env.step(action)
            state = next_state
            is_init_states_test = tensor(done).byte()
            prev_options_test = options

            if vis:
                env.render()
            total_reward += reward[0]
        return total_reward


    def run(self):
        progress_bar = tqdm(total=self.max_steps)
        try:
            self.actual_run(progress_bar)
        finally:
            progress_bar.close()
            self.envs.close()
            self.eval_env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run PPO on a specific game.')
    parser.add_argument('-e', '--env_name', type=str, help='Name of the game', default='HalfCheetah-v2')
    parser.add_argument("-d", "--device", type=str, help="device to run the network on", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("-s", "--steps", type=int, help="number steps to train agent", default=2e6)
    # parser.add_argument('-n', '--num_envs', type=int, help='Number of workers', default=1)
    # parser.add_argument('-a', '--activationF', type=str, help='Types of activation function', default='relu')
    args = parser.parse_args()
    # activation_dict = {'tanh':nn.Tanh, 'relu':nn.ReLU}
    if args.device == "cuda":
        args.device = "cuda:0"
    agent = DACA2CAgent(args.env_name, args.device, args.steps)
    agent.run()