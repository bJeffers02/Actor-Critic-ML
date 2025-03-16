from collections import namedtuple
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.serialization
import matplotlib.pyplot as plt
import time

class ActorCriticNN(nn.Module):
    def __init__(self, network_dims):
        super(ActorCriticNN, self).__init__()
        num_layers = len(network_dims)
        self.layers = nn.ModuleList()
        for i in range(num_layers-2):
            self.layers.append(nn.Linear(network_dims[i], network_dims[i+1]))
        self.actor_layer = nn.Linear(network_dims[num_layers-2], network_dims[num_layers-1])
        self.critic_layer = nn.Linear(network_dims[num_layers-2], 1)
        self.episode_actions = []
        self.episode_rewards = []
    
    def forward(self, x):
        x = torch.from_numpy(x).float()
        for layer in self.layers:
            x = F.sigmoid(layer(x))
        action = F.softmax(self.actor_layer(x), dim=-1)
        state_value = self.critic_layer(x)
        return action, state_value

def train():
    eps = np.finfo(np.float32).eps.item()
    discounted_sum = 0
    returns = []
    episode_actions = actor_critic.episode_actions
    actor_losses = []
    critic_losses = []

    for reward in actor_critic.episode_rewards[::-1]:
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), discounted_sum in zip(episode_actions, returns):
        advantage = discounted_sum - value.item()
        actor_losses.append(-log_prob * advantage)
        critic_losses.append(F.smooth_l1_loss(value, torch.tensor([discounted_sum])))

    optimizer.zero_grad()
    loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()
    loss.backward()
    optimizer.step()
    del actor_critic.episode_actions[:]
    del actor_critic.episode_rewards[:]

time_steps = 10000
end_condition = 8000
gamma = 0.99
learning_rate = 3e-2
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    #actor_critic = ActorCriticNN([8, 64, 128, 64, 4])
    actor_critic = torch.load("LunarLander(12000).pth")
    optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    rewards = []
    episode_num = 0
    while True:
        episode_num += 1
        state, _ = env.reset()
        ep_reward = 0
        resets = 0
        t1 = time.time()
        for i in range(time_steps):
            state = np.ndarray.flatten(state)
            action_probabilities, state_value = actor_critic(state)
            m = Categorical(action_probabilities)
            action = m.sample()
            actor_critic.episode_actions.append(SavedAction(m.log_prob(action), state_value))
            action = action.item()
            state, reward, done, _, _ = env.step(action)
            actor_critic.episode_rewards.append(reward)
            ep_reward += reward
            if done:
                resets += 1
                state, _ = env.reset()
            rewards.append(ep_reward)
        t2 = time.time()
        train()
        t3 = time.time()
        print('Episode {}\tLast reward: {:.2f}'.format(episode_num, ep_reward)) 
        print("sim time: " + str(t2-t1))
        print("train time: " + str(t3-t2))
        if ep_reward > end_condition:
            print("Finished")
            break

    file_path = "LunarLander(" + str(end_condition) + ").pth" 
    torch.save(actor_critic, file_path)
    plt.plot(np.squeeze(rewards))
    plt.ylabel('Average Reward')
    plt.xlabel('Episodes')
    plt.show()

