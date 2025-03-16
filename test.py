from Lunar_Lander import ActorCriticNN
import gym
import torch
import torch.serialization
import numpy as np
from torch.distributions import Categorical
from matplotlib import pyplot as PLT

env = gym.make("LunarLander-v2", render_mode="human")
state, _ = env.reset()
model = torch.load("LunarLander(8000).pth")
#model = ActorCriticNN([8, 4])

model.eval()

for i in range(10000):
    state = np.ndarray.flatten(state)
    actions, _ = model(state)
    m = Categorical(actions)
    action = m.sample()
    action = action.item()
    state, reward, done, _, _ = env.step(action)
    if done:
        state, _ = env.reset()