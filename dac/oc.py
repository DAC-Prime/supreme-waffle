## Based on the implementation of OC in https://github.com/ShangtongZhang/DeepRL/blob/32bd11e9f01ef5e473d58d0bf64b89ad196e5727/deep_rl/agent/OC_agent.py
from network import *
from env.env import *
import random
import datetime
import os
import sys
import ast
class oc_agent():
    def __init__(self,
                 num_envs=16,
                 env_name="HalfCheetah-v2",
                 lr=3e-4,
                 num_steps=5,
                 num_options=4):
        self.num_envs = num_envs
        self.env_name = env_name
        self.lr = lr
        self.num_steps = num_steps
        self.envs = [make_env(self.env_name, i) for i in range(self.num_envs)]
        if self.num_envs == 1:
            self.envs = DummyVecEnv(self.envs)
        else:
            self.envs = SubprocVecEnv(self.envs)
        self.envs = VecNormalize(self.envs, ret=False)
        self.env = [make_env(self.env_name)]
        self.env = DummyVecEnv(self.env)
        self.env = VecNormalize(self.env, ret=False)
        self.num_inputs = self.envs.observation_space.shape[0]
        self.num_outputs = self.envs.action_space.shape[0]
        self.num_options = num_options
        self.model = DACNetwork(self.num_inputs, self.num_outputs, self.num_options)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr,
                                    eps=1e-5)
        self.max_frames = 2000000
        self.is_init_state = tensor(np.ones((self.num_envs))).byte()
        self.worker_index = tensor(np.arange(self.num_envs)).long()
        self.prev_option = tensor(np.zeros(self.num_envs)).long()
        self.eps = 0.1

    def sample_option(self, prediction, epsilon, prev_option, is_init_state):
        with torch.no_grad():
            q_option = prediction['q_option']
            pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
            greedy_option = q_option.argmax(dim=-1, keepdim=True)
            prob = 1 - epsilon + epsilon / q_option.size(1)
            prob = torch.zeros_like(pi_option).add(prob)
            pi_option.scatter_(1, greedy_option, prob)

            mask = torch.zeros_like(q_option)
            mask[:, prev_option] = 1
            beta = prediction['beta']
            pi_hat_option = (1 - beta) * mask + beta * pi_option

            dist = torch.distributions.Categorical(probs=pi_option)
            options = dist.sample()
            dist = torch.distributions.Categorical(probs=pi_hat_option)
            options_hat = dist.sample()

            options = torch.where(is_init_state, options, options_hat)
        return options


    def run(self):
        frame_idx = 0
        test_rewards = []
        state = self.envs.reset()
        cumu_rewd = np.zeros(self.num_envs)
        path='./data/{}'.format(self.env_name)
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)
        curtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S") \
                    + "_{:04d}".format(random.randint(1,9999))
        fd_train = open(path + '/oc_train_{}.log'.format(curtime), 'w')
        while frame_idx < self.max_frames:
            log_probs = []

            states = []
            actions = []
            rewards = []
            masks = []
            np_masks = []
            entropies = []
            #            entropy = 0
            prev_options = []
            rets = []
            advs = []
            predictions = []
            beta_advs = []
            qos = []
            options = []
            betas = []
            is_init_states = []
            for _ in range(self.num_steps):
                state = torch.FloatTensor(state)
                prediction = self.model(state)

                option = self.sample_option(prediction, self.eps, self.prev_option, self.is_init_state)
                mean = prediction['mean'][self.worker_index, option]
                std = prediction['std'][self.worker_index, option]
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

                log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
                entropy = dist.entropy().sum(-1).unsqueeze(-1)



                next_state, reward, done, info = self.envs.step(action)
                cumu_rewd += reward
                for i in range(self.num_envs):
                    if done[i]:
                        print("Cumulative reward at step " + str(frame_idx) +
                              " is " + str(cumu_rewd[i]))
                        fd_train.write("%d %f\n" % (frame_idx, cumu_rewd[i]))
                        cumu_rewd[i] = 0

                    fd_train.flush()

                log_probs.append(log_prob)
                prev_options.append(self.prev_option.unsqueeze(-1))
                options.append(option.unsqueeze(-1))
                rewards.append(torch.FloatTensor(reward).unsqueeze(-1))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(-1))
                entropies.append(entropy)
                states.append(state)
                actions.append(action)
                betas.append(prediction['beta'])
                qos.append(prediction['q_option'])
                is_init_states.append(self.is_init_state.unsqueeze(-1).float())
                self.is_init_state = tensor(done).byte()
                self.prev_option = option
                state = next_state
                frame_idx += self.num_envs

            next_state = torch.FloatTensor(next_state)
            with torch.no_grad():
                prediction = self.model(next_state)
                betass = prediction['beta'][self.worker_index, self.prev_option]
                ret = (1 - betass) * prediction['q_option'][self.worker_index, self.prev_option] + \
                  betass * torch.max(prediction['q_option'], dim=-1)[0]
                ret = ret.unsqueeze(-1)

            for i in reversed(range(self.num_steps)):
                ret = rewards[i] + 0.99 * masks[i] * ret
                adv = ret - qos[i].gather(1, options[i])
                rets.insert(0, ret)
                advs.insert(0, adv)
                v = qos[i].max(dim=-1, keepdim=True)[0] * (1 - self.eps) + qos[i].mean(-1).unsqueeze(-1) * self.eps
                q = qos[i].gather(1, prev_options[i])
                beta_advs.append(q - v + 0.01)


            log_probs = torch.cat(log_probs)
            betas = torch.cat(betas)
            qos = torch.cat(qos)
            states = torch.cat(states)
            actions = torch.cat(actions)
            rets = torch.cat(rets).detach()
            advs = torch.cat(advs).detach()
            beta_advs = torch.cat(beta_advs).detach()
            entropies = torch.cat(entropies)
            options = torch.cat(options)
            prev_options = torch.cat(prev_options)
            is_init_states = torch.cat(is_init_states)

            q_loss = (qos.gather(1, options) - rets).pow(2).mul(0.5).mean()
            pi_loss = -(log_probs * advs) - 0.01 * entropies
            pi_loss = pi_loss.mean()
            beta_loss = (betas.gather(1, prev_options) * beta_advs * (1 - is_init_states)).mean()

            self.optimizer.zero_grad()
            (pi_loss + q_loss + beta_loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        fd_train.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run OC on a specific game.')
    parser.add_argument('-e', '--env_name', type=str, help='Name of the game', default='HalfCheetah-v2')
    parser.add_argument('-n', '--num_envs', type=int, help='Number of workers', default=16)
    args = parser.parse_args()
    ocagent = oc_agent(num_envs=args.num_envs, env_name=args.env_name)
    ocagent.run()
