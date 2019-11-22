from enum import Enum

from ..network import *
from ..parameters import *
import torch
import numpy as np

# def tensor(x):
#     if isinstance(x, torch.tensor):
#         return x
#     x = torch.tensor(np.asanyarray(x, dtype = np.float), device = Config.DEVICE, dtype = torch.float32)
#     return x

class MdpType(Enum):
    high = "high"
    low = "low"

class DACPPOAgent:
    def __init__(self, env_name):
        # following fields are from config
        self.num_workers = 1
        self.discount = None
        self.use_gae = False
        self.optimization_epochs = 1
        self.gae_tau = 1.0

        self.cur_steps = 0
        self.max_steps = 2e6
        self.num_steps = 2048
        self.env_name = env_name
        self.num_envs = 1
        self.num_options = 4
        
        self.is_init_states = tensor(np.ones((self.num_workers))).byte()
        self.worker_index = tensor(np.arange(self.num_workers)).long()
        self.prev_options = tensor(np.zeros(self.num_workers)).long()

        self.envs = [make_env(self.env_name) for i in range(self.num_envs)]
        self.envs = SubprocVecEnv(self.envs)
        self.envs = VecNormalize(self.envs, ret=False)
        self.test_env = [make_env(self.env_name)]
        self.test_env = DummyVecEnv(self.env)
        self.test_env = VecNormalize(self.env, ret=False)

        self.states = self.envs.reset()
        self.test_env.reset()

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        # self.options = [LowerNetwork(obs_dim, action_dim) for _ in range(self.num_options)]
        # self.higher_policy = MasterNetwork(obs_dim, self.num_options)
        self.dac_net = DACNetwork(obs_dim, action_dim, self.num_options)
        self.opt = optim.Adam(self.dac_net.parameters(),
                                    lr=3e-4,
                                    eps=1e-5)
        # self.higher_optimizer =
        # self.lower_optimizer =

    def compute_pi_h(self, prediction, pre_option, is_init_states):
        intra_pi = prediction["master_policy"]
        beta = prediction["beta"]

        mask = torch.zeros_like(intra_pi)
        # mask[pre_option, self.worker_index]


        pi_h = beta * intra_pi + (1 - beta) * mask
        is_init_states = is_init_states.view(-1, 1).expand(-1, intra_pi.size(1))
        pi_h = torch.where(is_init_states, intra_pi, pi_h)

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
            # return pi_hat.add(1e-5).log().gather(1, options)
            return pi_h.log().gather(1, options)
        elif mdp_type == MdpType.low:
            pi_l = self.compute_pi_l(options, action, mean, std)
            # return pi_l.add(1e-5).log()
            return pi_l.log()
        else:
            raise NotImplementedError


    def compute_adv(self, parameters, mdp_type):
        # config = self.config

        # how to use mdp_type ???
        v = parameters.__getattribute__("v_%s" % (mdp_type.value))
        adv = parameters.__getattribute__("adv_%s" % (mdp_type.value))
        returns = parameters.__getattribute__("ret_%s" % (mdp_type.value))

        ret = v[-1].detach()
        advs = tensor(np.zeros((self.num_workers, 1)))

        for i in reversed(range(self.num_workers, 1)):
            ret = parameters.r[i] * + parameters.m[i] * ret
            if not self.use_gae:
                advs = ret - v[i].detach()
            else:
                td_error = parameters.r[i] + self.discount * parameters.m[i] * v[i + 1] - v[i] # td_error ?
                advs = advs * self.gae_tau * self.discount * parameters.m[i] + td_error
            adv[i] = advs.detach()
            returns[i] = ret.detach


    def learn(self):
        # config = self.config
        pass

    # use code from ppo.py here
    def run(self):
        states = self.states
        # storage = Storage(config.rollout_length, ['adv_bar', 'adv_hat', 'ret_bar', 'ret_hat'])
        cumu_rewd = np.zeros(self.num_envs)

        while self.cur_steps <= self.max_steps:
            for _ in range(self.num_steps):
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

                next_state, reward, done, info = self.envs.step(actions) # done: terminated
                cumu_rewd += reward
                
                #  storage.add(prediction)
                #  storage.add({'r': tensor(rewards).unsqueeze(-1),
                #              'm': tensor(1 - terminals).unsqueeze(-1),
                #              'a': actions,
                #              'o': options.unsqueeze(-1),
                #              'prev_o': self.prev_options.unsqueeze(-1),
                #              's': tensor(states),
                #              'init': self.is_initial_states.unsqueeze(-1),
                #              'pi_hat': pi_hat,
                #              'log_pi_hat': pi_hat[self.worker_index, options].add(1e-5).log().unsqueeze(-1),
                #              'log_pi_bar': pi_bar.add(1e-5).log(),
                #              'v_bar': v_bar,
                #              'v_hat': v_hat})

                self.is_initial_states = tensor(done).byte()
                self.prev_options = options
                states = next_state
                self.cur_steps += self.num_workers
            
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

            # storage.add(prediction)
            # storage.add({
            #     'v_bar': v_bar,
            #     'v_hat': v_hat,
            # })
            # storage.placeholder()

            # [o] = storage.cat(['o'])

            # log here

            # computer advantange 

            mdp_types = [MdpType.high, MdpType.low]
            np.random.shuffle(mdps)
            self.learn(storage, mdps[0])
            self.learn(storage, mdps[1])
