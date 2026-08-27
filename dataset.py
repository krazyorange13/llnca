import math
import random
from collections import defaultdict
from dataclasses import dataclass

from torch.utils.data import Dataset, Sampler


@dataclass
class LLNCADatasetConfig:
    file: str


class LLNCADataset(Dataset):
    def __init__(self, config: LLNCADatasetConfig):
        self.config = config
        self.rows = []
        self.load()

    def load(self):
        print(f"  \033[2min: {corptok_path}\033[0m")
        with open(self.config.file) as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        for i in range(0, len(lines), 2):
            x = lines[i][:]
            y = lines[i + 1][:]
            self.rows.append((x, y))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


@dataclass
class LLNCADataSamplerConfig:
    batch_l: int
    drop_last: bool
    shuffle: bool = True
    bin_interval: int = 8


class LLNCADataSampler(Sampler):
    def __init__(self, dataset: LLNCADataset, config: LLNCADataSamplerConfig):
        self.dataset = dataset
        self.config = config

        self.bins = defaultdict(list)
        b = self.config.bin_interval
        for i in range(len(dataset)):
            l = len(dataset[i][0] + dataset[i][1])
            l = math.ceil(l / b) * b
            self.bins[l].append(i)

        print("\033[2mbins:\033[0m")
        bin_sz_maxlen = max(len(str(bin_sz)) for bin_sz in self.bins)
        for bin_sz, bin_l in sorted(self.bins.items()):
            bin_sz_str = str(bin_sz).rjust(bin_sz_maxlen)
            print(f"  \033[2m{bin_sz_str}: {'.' * len(bin_l)}\033[0m")

    def __iter__(self):
        batches = []

        for idxs in self.bins.values():
            if self.config.shuffle:
                random.shuffle(idxs)

            for i in range(0, len(idxs), self.config.batch_l):
                batch = idxs[i : i + self.config.batch_l]
                if self.config.drop_last and len(batch) < self.config.batch_l:
                    continue
                batches.append(batch)

        if self.config.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        tn_batches = 0
        for idxs in self.bins.values():
            n_batches = len(idxs) // self.config.batch_l
            if not self.config.drop_last and len(idxs) % self.config.batch_l != 0:
                n_batches += 1
            tn_batches += n_batches
        return tn_batches


if __name__ == "__main__":
    corptok_path = "data/ezpz/ezpz.corptok.txt"
    print("loading dataset...")
    dataset_config = LLNCADatasetConfig(file=corptok_path)
    dataset = LLNCADataset(dataset_config)
    print("loading sampler...")
    sampler_config = LLNCADataSamplerConfig(batch_l=8, drop_last=False, shuffle=True)
    sampler = LLNCADataSampler(dataset, sampler_config)
