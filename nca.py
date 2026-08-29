from dataclasses import dataclass

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


class LLNCAFilter(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 3
        self.conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=2,
            groups=in_channels,
        )
        self.reset()

    def reset(self):
        identity = torch.tensor([0.0, 1.0])
        previous = torch.tensor([1.0, 0.0])
        gradient = torch.tensor([-1.0, 1.0])
        kernel = torch.stack([identity, previous, gradient])[:, None, :]
        with torch.no_grad():
            self.conv.weight.copy_(kernel.repeat(self.in_channels, 1, 1))

    def forward(self, x):
        x = F.pad(x, (1, 0), mode="constant", value=0.0)
        return self.conv(x)


@dataclass
class LLNCANCAConfig:
    channels: int
    mlp_width: int
    mlp_depth: int
    activation_fn: str
    update_rate: float
    alive_threshold: float


class LLNCANCA(nn.Module):
    def __init__(self, config: LLNCANCAConfig):
        super().__init__()
        self.config = config

        self.filter = LLNCAFilter(self.config.channels)

        mlp_width = self.config.mlp_width
        filter_channels = self.filter.out_channels
        activation_fn: type[nn.Module] = getattr(nn, self.config.activation_fn)
        layers = []
        layers.append(nn.Conv1d(filter_channels, mlp_width, kernel_size=1))
        for _ in range(self.config.mlp_depth - 2):
            layers.append(nn.Conv1d(mlp_width, mlp_width, kernel_size=1))
            layers.append(activation_fn())
        layers.append(nn.Conv1d(mlp_width, self.config.channels, kernel_size=1))

        self.seq = nn.Sequential(*layers)

    def add_channels(self, x: torch.Tensor):
        b, c, w = x.shape
        n_h = self.config.channels - c
        h = torch.zeros((b, n_h, w), device=x.device)
        y = torch.cat([x, h], dim=1)
        return y

    def get_alive_mask(self, x: torch.Tensor):
        y = F.pad(x.abs(), (1, 0), mode="constant", value=0.0)
        y = F.max_pool1d(y, kernel_size=2, stride=1)
        y = y.amax(dim=1, keepdim=True)
        y = y >= self.config.alive_threshold
        return y.float()

    def get_update_mask(self, x: torch.Tensor):
        b, _, w = x.shape
        y = torch.rand(b, 1, w, device=x.device) < self.config.update_rate
        return y.float()

    def step(self, x: torch.Tensor, freeze_mask: torch.Tensor):
        y = self.filter(x)
        y = self.seq(y)

        alive_mask = self.get_alive_mask(x)
        update_mask = self.get_update_mask(x)
        y = y * freeze_mask * alive_mask * update_mask

        y = x + y
        return y

    def forward(
        self, x: torch.Tensor, steps: int = 1, freeze_mask: torch.Tensor | None = None
    ):
        if freeze_mask is None:
            freeze_mask = torch.ones_like(x)

        for i in range(steps):
            x = self.step(x, freeze_mask)

        return x


# nca img
# x.shape =
#   (B    , C      , W    )
#   (batch, channel, width)


# llnca adv nca gan
# use same nca setup
# frozen channels holding gen nca output
# output channel grades realism
# minimize average or wtvr
