from collections import Counter, defaultdict

import gym

from torch.utils.tensorboard import SummaryWriter


ENV_NAME = "FrozenLake-v0"
GAMMA = .9
TEST_EPISODES = 20


class Agent(object):
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.rewards = defaultdict(float)
        self.transitions = defaultdict(Counter)
        self.values = defaultdict(float)

        self.state = self.env.reset()

    def play_random_steps(self, n_steps):
        action = self.env.action_space.sample()
        new_state, reward, is_done, _ = self.env.step(action)

        self.rewards[new_state] = reward
        self.transitions[(self.state, action)][new_state] += 1
        self.state = self.env.reset() if is_done else new_state
