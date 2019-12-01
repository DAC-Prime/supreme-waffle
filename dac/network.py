import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEFAULT_HIDDEN_UNITS = 64
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def tensor(x):
    if torch.is_tensor(x):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
    return x

class DACNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, num_options):
        super(DACNetwork, self).__init__()

        self.higher_net = MasterNetwork(obs_dim, num_options)
        self.lower_nets = [LowerNetwork(obs_dim, action_dim) for _ in range(num_options)]
        self.to(DEVICE)

    def forward(self, x):
        mean = []
        std = []
        beta = []
        for lower_net in self.lower_nets:
            option_pred = lower_net(x)
            mean.append(option_pred["mean_action"].unsqueeze(1))
            std.append(option_pred["std_action"].unsqueeze(1))
            beta.append(option_pred["termination_prob"])
        mean = torch.cat(mean, dim=1)
        std = torch.cat(std, dim=1)
        beta = torch.cat(beta, dim=1)

        master_pred = self.higher_net(x)

        return {
            "mean": mean,
            "std": std,
            "beta": beta,
            "q_option": master_pred["q_option"],
            "master_policy": master_pred["master_policy"],
        }


class MasterNetwork(nn.Module):
    def __init__(self, obs_dim, num_options):
        super(MasterNetwork, self).__init__()

        self.master_policy_net = FCNetwork(obs_dim, num_options, lambda: nn.Softmax(dim=-1))
        self.value_net = FCNetwork(obs_dim, num_options)

    def forward(self, x):
        master_policy = self.master_policy_net(x)
        q_option = self.value_net(x)

        return {
            "master_policy": master_policy,
            "q_option": q_option,
        }


class LowerNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(LowerNetwork, self).__init__()

        self.policy_net = FCNetwork(obs_dim, action_dim, nn.Tanh)
        self.termination_net = FCNetwork(obs_dim, 1, nn.Sigmoid)
        self.std = nn.Parameter(torch.zeros((1, action_dim)))

    def forward(self, x):
        mean_action = self.policy_net(x)
        std_action = F.softplus(self.std).expand(mean_action.size(0), -1) # ?
        termination_prob = self.termination_net(x)

        return {
            "mean_action": mean_action,
            "std_action": std_action,
            "termination_prob": termination_prob,
        }


class FCNetwork(nn.Module):
    def __init__(self,
        input_dim, output_dim, output_activation=None,
        hidden_dims=(DEFAULT_HIDDEN_UNITS, DEFAULT_HIDDEN_UNITS), hidden_activation=nn.Tanh
    ):
        super(FCNetwork, self).__init__()

        layers = list()
        dims = (input_dim,) + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            layers.append(hidden_activation())
        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=output_dim))
        if output_activation is not None:
            layers.append(output_activation())

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        data = tensor(x)
        for layer in self.layers:
            data = layer(data)

        return data