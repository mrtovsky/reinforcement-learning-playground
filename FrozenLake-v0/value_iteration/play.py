from collections import Counter, defaultdict

import gym

from torch.utils.tensorboard import SummaryWriter


ENV_NAME = "FrozenLake-v0"
GAMMA = .9
TEST_EPISODES = 20
REWARD_THRESHOLD = .8


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
        for idx, action in enumerate(range(self.env.action_space.n)):
            action_value = self.calculate_action_value(state, action)
            if idx == 0:
                best_action = action
                best_value = action_value
            elif best_value < action_value:
                best_value = action_value
                best_action = action

        return best_action

    def play_episode(self, env):
        state = env.reset()
        total_reward = 0

        is_done = False

        while not is_done:
            action = self.pick_best_action(state)
            new_state, reward, is_done, _ = env.step(action)

            if new_state not in self.rewards.keys():
                self.rewards[new_state] = reward
            self.transitions[(state, action)][new_state] += 1
            total_reward += reward
            state = new_state

        return total_reward

    def iterate_values(self):
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calculate_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    env_test = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-value-iteration")

    best_reward = 0
    well_trained = False
    idx = 0

    while not well_trained:
        idx += 1

        agent.play_random_steps(100)
        agent.iterate_values()

        reward = 0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(env_test)
        reward /= TEST_EPISODES

        writer.add_scalar("reward", reward, idx)

        if reward > best_reward:
            best_reward = reward

        if reward > REWARD_THRESHOLD:
            print("Solved in {} nr of steps!".format(idx))
            well_trained = True
    writer.close()
