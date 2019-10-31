import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from batch import iterate, pick_elites
from net import Net


ENV_NAME = "CartPole-v0"
BATCH_SIZE = 16
HIDDEN_SIZE = 128
PERCENTILE = 70
MAX_TRAIN_STEPS = 1000
MAX_GAME_STEPS = 2000


def main():
    env = gym.make(ENV_NAME)
    env._max_episode_steps = MAX_TRAIN_STEPS
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=.01)

    for idx, batch in enumerate(iterate(env, net, BATCH_SIZE)):
        train, reward_thresh, reward_mean = pick_elites(batch, PERCENTILE)
        obs, actions = zip(*train)
        optimizer.zero_grad()
        action_scores = net(torch.FloatTensor(obs))
        loss = objective(action_scores, torch.LongTensor(actions))
        loss.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            idx, loss.item(), reward_mean, reward_thresh
        ))
        if reward_mean > MAX_TRAIN_STEPS - 1:
            print("Solved!")
            break
    env._max_episode_steps = MAX_GAME_STEPS
    obs = env.reset()
    is_done = False
    sm = nn.Softmax(dim=0)
    while not is_done:
        env.render()
        actions_probas = sm(net(torch.FloatTensor(obs))).data.numpy()
        action = np.random.choice(len(actions_probas), p=actions_probas)
        next_obs, _, is_done, _ = env.step(action)
        obs = next_obs
    env.close()


if __name__ == "__main__":
    main()
