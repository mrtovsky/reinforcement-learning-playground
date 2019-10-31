import torch.nn as nn


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                in_features=obs_size,
                out_features=hidden_size,
                bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_size,
                out_features=n_actions,
                bias=True
            )
        )

    def forward(self, x):
        return self.net(x)
