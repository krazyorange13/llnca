import sys
import time
import math
import random

from dataclasses import dataclass
from tqdm import tqdm

# from main import LLNCA, LLNCAConfig


@dataclass
class GridConfig:
    folder: str
    iter_params: list[tuple[str, list]]
    curr_params: list[int]


class Grid:
    def __init__(self, config: GridConfig):
        self.config = config

        n_combs = math.prod([len(options) for name, options in config.iter_params])
        print("combinations:", n_combs)

        self.ljust_n = max([len(name) for name, options in config.iter_params])

    def _grid(self, n):
        if n >= len(self.config.iter_params):
            return

        param = self.config.iter_params[n]
        name, options = param
        for i in tqdm(
            range(len(options)),
            desc=name.ljust(self.ljust_n),
            leave=False,
            dynamic_ncols=True,
        ):
            time.sleep(0.1)
            self._grid(n + 1)

    def grid(self):
        self._grid(0)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        ...
    else:
        iter_params = [
            ("nca_widths", [128, 256, 512, 1024]),
            ("nca_depths", [8, 12, 16]),
            ("channels", [128, 256, 512]),
            ("batches", [8, 16, 32]),
            ("lrs", [1e-4, 3e-4, 1e-3, 3e-3]),
            ("lr_gammas", [0.999, 0.9999, 0.99999]),
            ("gammas", [(0.9, 0.9999), (0.99, 0.9999), (0.5, 0.5)]),
            ("backprop_chunks", [16, 32, 64, 128]),
        ]
        curr_params = [0] * len(iter_params)
        config = GridConfig(
            folder="grid/alpha",
            iter_params=iter_params,
            curr_params=curr_params,
        )
        grid = Grid(config)
        grid.grid()
