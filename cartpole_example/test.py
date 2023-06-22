import torch
from torch import multiprocessing as mp
import gym
class net(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(4, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1),

        )
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.layer(x)
        return x

n = net()
s = torch.tensor([[1., 1., 1., 1.], [2., 2., 2., 2.]])
a = n(s)

print(torch.tril(
            torch.outer(
                torch.tensor(1 / 2) ** torch.arange(0, 3, step=1),
                torch.tensor(2) ** torch.arange(0, 3, step=1)
            )
        ))



