from ..network import *
from ..parameters import *
import torch
import numpy as np

def tensor(x):
    if isinstance(x, torch.tensor):
        return x
    x = torch.tensor(np.asanyarray(x, dtype= np.float), device = Config.DEVICE, dtype = torch.float32)
    return x

class DACPPOAgent:
    # def __init__(self, config):
    def __init__(self):
        # following fields are from config
        self.num_workers = 1
        self.discount = None
        self.use_gae = False
        self.optimization_epochs = 1
        self.gae_tau = 1.0

        # self.config = config

        self.network = config.network_fn()
        self.option_set = []
        self.is_init_states = tensor(np.ones((config.num_workers))).byte()
        # self.worker_index = 
        # self.pre_options = 
        self.worker_index = tensor(np.arange(self.num_workers)).long()


    def compute_pi_h(self, prediction, pre_option, is_init_states):
        intra_pi = prediction["policy_option"]
        beta = prediction["termination_prob"]

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
        if mdp_type == "high":
            # return pi_hat.add(1e-5).log().gather(1, options)
            return pi_h.log().gather(1, options)
        elif mdp_type == "low":
            pi_l = self.compute_pi_l(options, action, mean, std)
            # return pi_l.add(1e-5).log()
            return pi_l.log()
        else:
            raise NotImplementedError


    def compute_adv(self, parameters, mdp_type):
        # config = self.config

        # how to use mdp_type ???
        v = parameters.__getattribute__("v_%s" % (mdp_type))
        adv = parameters.__getattribute__("adv_%s" % (mdp_type))
        returns = parameters.__getattribute__("ret_%s" % (mdp_type))

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


    # use code from ppo.py here
    def step(self):
        pass