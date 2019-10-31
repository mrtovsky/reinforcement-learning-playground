from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn


Episode = namedtuple("Episode", field_names=["reward", "steps"])
EpisodeStep = namedtuple("EpisodeStep", field_names=["observation", "action"])


def iterate(env, net, batch_size):
    sm = nn.Softmax(dim=0)
    while True:
        batch = []
        for _ in range(batch_size):
            episode_steps, episode_reward = [], 0
            obs = env.reset()
            is_done = False
            while not is_done:
                actions_probas = sm(net(torch.FloatTensor(obs))).data.numpy()
                action = np.random.choice(len(actions_probas), p=actions_probas)
                next_obs, reward, is_done, _ = env.step(action)
                episode_reward += reward
                episode_steps.append(EpisodeStep(
                    observation=obs,
                    action=action
                ))
                obs = next_obs
            batch.append(Episode(
                reward=episode_reward,
                steps=episode_steps
            ))
        yield batch


def pick_elites(batch, percentile):
    rewards = [episode.reward for episode in batch]
    reward_thresh = np.percentile(rewards, percentile)
    reward_mean = np.mean(rewards)

    train = [
        (step.observation, step.action)
        for episode in batch
        for step in episode.steps
        if episode.reward >= reward_thresh
    ]

    return train, reward_thresh, reward_mean
