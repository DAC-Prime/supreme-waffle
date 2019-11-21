## Based on the implementation of PPO in https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
from network import *
from env.env import *


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=1e-3)
        nn.init.constant_(m.bias, 0)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dims, hidden_activation, std=0.0):
        super(ActorCritic, self).__init__()
        self.critic = FCNetwork(
            num_inputs, 1, hidden_dims=hidden_dims, hidden_activation=hidden_activation
        )
        # using relu instead of tanh(used in DAC paper) seems to make PPO learn faster and stabler at beginning
        self.actor = FCNetwork(num_inputs,
                               num_outputs,
                               hidden_dims=hidden_dims,
                               hidden_activation=hidden_activation)
        #        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        # log_std was used in the OpenAI's implementation of PPO, while the DAC author's has some modification
        self.std = nn.Parameter(torch.zeros(num_outputs))
        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu = F.tanh(self.actor(x))
        #        std   = self.log_std.exp()#.expand_as(mu)
        #        dist  = Normal(mu, std)
        dist = Normal(mu, F.softplus(self.std))
        return dist, value


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step +
                                               1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[
            rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


def ppo_update(model,
               optimizer,
               ppo_epochs,
               mini_batch_size,
               states,
               actions,
               log_probs,
               returns,
               advantages,
               clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(
                mini_batch_size, states, actions, log_probs, returns,
                advantages):
            dist, value = model(state)
            #            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                1.0 + clip_param) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss  # - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


class ppo_agent():
    def __init__(self,
                 num_envs=1,
                 env_name="HalfCheetah-v2",
                 hidden_dims=(64, 64),
                 hidden_activation=nn.Tanh,
                 lr=3e-4,
                 num_steps=2048,
                 mini_batch_size=64,
                 ppo_epochs=10):
        self.num_envs = num_envs
        self.env_name = env_name
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.lr = lr
        self.num_steps = num_steps
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.envs = [make_env(self.env_name) for i in range(self.num_envs)]
        self.envs = SubprocVecEnv(self.envs)
        self.envs = VecNormalize(self.envs, ret=False)
        self.num_inputs = self.envs.observation_space.shape[0]
        self.num_outputs = self.envs.action_space.shape[0]
        self.model = ActorCritic(self.num_inputs, self.num_outputs,
                                 self.hidden_dims, self.hidden_activation)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr,
                                    eps=1e-5)
        self.max_frames = 2000000

    def run(self):
        frame_idx = 0
        test_rewards = []
        state = self.envs.reset()
        cumu_rewd = np.zeros(self.num_envs)
        try:
            os.mkdir('data')
        except OSError as error:
            print(error)
        fd = open('./data/{}_ppo.log'.format(self.env_name), 'w')
        while frame_idx < self.max_frames:
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            np_masks = []
            #            entropy = 0

            for _ in range(self.num_steps):
                state = torch.FloatTensor(state)
                dist, value = self.model(state)

                action = dist.sample()
                next_state, reward, done, info = self.envs.step(action)
                cumu_rewd += reward
                for i in range(self.num_envs):
                    if done[i]:
                        print("Cumulative reward at step " + str(frame_idx) +
                              " is " + str(cumu_rewd[i]))
                        fd.write("%d %f\n" % (frame_idx, cumu_rewd[i]))
                        cumu_rewd[i] = 0

#                    fd.flush()

                log_prob = dist.log_prob(action)
                #                entropy += dist.entropy().mean()
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1))
                states.append(state)
                actions.append(action)
                state = next_state
                frame_idx += 1

            next_state = torch.FloatTensor(next_state)
            _, next_value = self.model(next_state)
            returns = compute_gae(next_value, rewards, masks, values)
            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states).detach()
            actions = torch.cat(actions).detach()
            advantage = returns - values

            ppo_update(self.model, self.optimizer, self.ppo_epochs,
                       self.mini_batch_size, states, actions, log_probs,
                       returns, advantage)

        fd.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run PPO on a specific game.')
    parser.add_argument('-e', '--env_name', type=str, help='Name of the game', default='HalfCheetah-v2')
    args = parser.parse_args()
    ppoagent = ppo_agent(env_name=args.env_name, hidden_activation=nn.Tanh)
    ppoagent.run()
