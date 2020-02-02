from collections import Counter, defaultdict

import gym

from torch.utils.tensorboard import SummaryWriter


ENV_NAME = "FrozenLake-v0"
GAMMA = .9
ALPHA = .2
TEST_EPISODES = 20
REWARD_THRESHOLD = .8


class Agent(object):
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.values = defaultdict(float)

        self.state = self.env.reset()

    def update_value(self):
        action = self.env.action_space.sample()
        state = self.state

        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state

        _, new_state_value = self.pick_action(new_state)

        action_value = reward + GAMMA * new_state_value
        old_action_value = self.values[(state, action)]

        self.values[(state, action)] = (
            (1 - ALPHA) * old_action_value
            + ALPHA * action_value
        )

    def pick_action(self, state):
        best_value, best_action = 0, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value < action_value:
                best_action = action
                best_value = action_value

        if best_action is None:
            best_action = self.env.action_space.sample()

        return best_action, best_value

    def play_episode(self, env):
        state = env_test.reset()
        total_reward = 0

        is_done = False

        while not is_done:
            action, _ = self.pick_action(state)
            new_state, reward, is_done, _ = env_test.step(action)

            total_reward += reward
            state = new_state

        return total_reward


if __name__ == "__main__":
    env_test = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    best_reward = 0
    well_trained = False
    idx = 0

    while not well_trained:
        idx += 1

        agent.update_value()

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
