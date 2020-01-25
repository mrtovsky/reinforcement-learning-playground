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
        for _ in range(n_steps):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)

            if new_state not in self.rewards.keys():
                self.rewards[new_state] = reward
            self.transitions[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calculate_action_value(self, state, action):
        transition_counts = self.transitions[(state, action)]
        n_transitions = sum(transition_counts.values())

        action_value = 0
        for new_state, count in transition_counts.items():
            reward = self.rewards[new_state]
            proba_new_state = count / n_transitions
            action_value += (
                proba_new_state
                * (reward + GAMMA * self.values[new_state])
            )

        return action_value

    def pick_best_action(self, state):
        best_action, best_value = None, 0

        for action in range(self.env.action.n):
            action_value = self.calculate_action_value(state, action)
            if best_value < action_value:
                best_value = action_value
                best_action = action

        return best_action
