import sys
import random
from collections import defaultdict
from dataclasses import dataclass


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm


class PerceptionFilter(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 4
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
            # padding_mode="circular",
        )
        self.reset_params()

    def reset_params(self):
        identity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.T
        laplacian = torch.tensor([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]])
        kernel = torch.stack([identity, sobel_x, sobel_y, laplacian])[:, None, :, :]
        with torch.no_grad():
            self.conv.weight.copy_(kernel.repeat(self.in_channels, 1, 1, 1))

    def forward(self, x):
        return self.conv(x)


class NCA(nn.Module):
    def __init__(self, channels=16, update_rate=0.25):
        super(NCA, self).__init__()
        self.channels = channels
        self.update_rate = update_rate

        self.perception = PerceptionFilter(self.channels)
        self.perception.conv.weight.requires_grad = False

        self.seq = nn.Sequential(
            nn.Conv2d(self.channels * 4, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, self.channels, kernel_size=1, bias=False),
        )

        with torch.no_grad():
            self.seq[-1].weight.zero_()  # type: ignore

    def step(self, x, update_rate=None):
        pre_alive_mask = self.get_alive_mask(x)

        y = self.perception(x)
        y = self.seq(y)

        update_mask = self.get_update_mask(y.shape, update_rate)
        # x.add_(y * update_mask)
        x = x + y * update_mask

        post_alive_mask = self.get_alive_mask(x)
        combined_mask = pre_alive_mask * post_alive_mask
        # x.mul_(combined_mask)
        x = x * combined_mask

        return x

    def get_update_mask(self, shape, update_rate=None):
        b, _, h, w = shape
        update_rate = update_rate or self.update_rate
        update_mask = (torch.rand(b, 1, h, w) < update_rate).float()
        return update_mask

    def get_alive_mask(self, x, threshold=0.1):
        # x = F.pad(x, (1, 1, 1, 1), mode="circular")
        # NOTE: for single L channel
        alpha = x[:, :1, :, :]
        mask = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > threshold
        return mask.float()

    def forward(self, x, steps=1, update_rate=None):
        for i in range(steps):
            x = self.step(x, update_rate=update_rate)
        return x


class Renderer:
    def __init__(self, font_name, font_size):
        self.font_name = font_name
        self.font_size = font_size
        self.font = ImageFont.truetype(self.font_name, self.font_size)

    def text(self, text, match=None):
        if match is None:
            match = text

        # farm = "p'" * len(match)
        # farm = farm[: len(match)]
        # farm = "p'" + ("w" * (len(match) - 1))
        # farm = "w" * len(match)

        # bbox (left, top, right, bottom)
        bbox = self.font.getbbox(match)
        width = bbox[2] - bbox[0]
        # height = 9  # max height of our baby.otf font
        height = bbox[3] - bbox[1]
        width, height = int(width), int(height)

        img = Image.new("L", (width, height), color=(0))
        draw = ImageDraw.Draw(img)

        # -bbox[0], -bbox[1] offsets any font internal padding
        draw.text((-bbox[0], -bbox[1]), text, fill=(255), font=self.font)

        return np.asarray(img)


class SentenceDataset:
    def __init__(self, file, bin_size=16, trunc_ratio=3):
        self.file = file
        self.bin_size = bin_size
        self.trunc_ratio = trunc_ratio

        # list[sentence]
        self.sentences = self.load_sentences(self.file)
        # dict[bin number, list[(sentence, seed)]]
        self.bins = self.load_bins(self.sentences)

    def load_sentences(self, file):
        with open(file) as f:
            sentences = f.readlines()
        return sentences

    def load_bins(self, sentences):
        bins = defaultdict(list)
        for i, sentence in enumerate(sentences):
            seed = self.get_seed(sentence)
            bins[self.get_bin(sentence)].append((sentence, seed))
            # bins[self.get_bin(sentence)].append(i)
        return bins

    def get_seed(self, sentence):
        words = sentence.split()
        truncated = len(words) // self.trunc_ratio
        incomplete = " ".join(words[:-truncated])
        return incomplete

    def get_bin(self, sentence):
        return len(sentence) // self.bin_size


class Pool:
    def __init__(
        self,
        dataset: SentenceDataset,
        bin: int,
        renderer: Renderer,
        channels: int,
        pool_size=64,
    ):
        self.dataset = dataset
        self.bin = bin
        self.renderer = renderer
        self.channels = channels
        self.pool_size = pool_size
        self.reset()

    def reset(self):
        pairs = [self.get_pair() for i in range(self.pool_size)]
        sentences, seeds = zip(*pairs)
        self.ys = torch.stack(sentences)
        self.xs = torch.stack(seeds)

    def sample(self, batch_size=8, damaged=3):
        self.idxs = torch.randperm(self.pool_size)[:batch_size]
        self.y_batch = self.ys[self.idxs]
        self.x_batch = self.xs[self.idxs]

        losses = F.mse_loss(
            self.x_batch[:, :1, :, :],
            self.y_batch,
            reduction="none",
        ).sum(dim=(1, 2, 3))
        sorted_idxs = torch.argsort(losses)

        replace_idx = sorted_idxs[-1]
        sentence, seed = self.get_pair()
        self.y_batch[replace_idx] = sentence
        self.x_batch[replace_idx] = seed

        damaged_idxs = sorted_idxs[:damaged]
        self.x_batch[damaged_idxs] = self.damage(self.x_batch[damaged_idxs])

        return self.x_batch

    def get_pair(self):
        farm = "p'" + "w" * (self.dataset.bin_size * (self.bin + 1) - 1)
        sentence, seed = random.choice(self.dataset.bins[self.bin])
        sentence_img = self.renderer.text(sentence, match=farm) / 255.0
        seed_img = self.renderer.text(seed, match=farm) / 255.0
        sentence_t = (
            torch.tensor(sentence_img, dtype=torch.float32)
            .unsqueeze(2)
            .permute(2, 0, 1)
        )
        seed_t = (
            torch.tensor(seed_img, dtype=torch.float32).unsqueeze(2).permute(2, 0, 1)
        )
        c, h, w = seed_t.shape
        seed_hid = torch.zeros((self.channels - c, h, w))
        seed_t = torch.cat([seed_t, seed_hid])
        return sentence_t, seed_t

    def damage(self, batch):
        b, c, h, w = batch.shape
        grid = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        grid = torch.stack(grid, dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        center = torch.rand(b, 2, 1, 1) - 0.5
        radius = 0.3 * torch.rand(b, 1, 1, 1) + 0.1
        mask = ((grid - center) * (grid - center)).sum(1, keepdim=True).sqrt() > radius
        return batch * mask.float()

    def update(self, samples):
        loss = F.mse_loss(samples[:, :1, :, :], self.ys[self.idxs])

        samples = samples.detach()
        # replaces samples in batch returned by sample()
        self.xs[self.idxs] = samples

        return loss


class PoolPool:
    def __init__(self, pools: list[Pool]):
        self.pools = pools

    def sample(self, batch_size=8, damaged=3):
        self.idx = random.randint(0, len(self.pools) - 1)
        pool = self.pools[self.idx]
        return pool.sample(batch_size, damaged)

    def update(self, samples):
        return self.pools[self.idx].update(samples)


@dataclass(frozen=True)
class LLNCAConfig:
    name: str
    folder: str
    sentences_file: str
    font_name: str
    font_size: int
    bin_size: int
    trunc_ratio: int
    epochs: int
    batch_size: int
    channels: int
    lr: float = 2e-3
    lr_gamma: float = 0.9999
    betas: tuple[float, float] = (0.9, 0.9999)


class LLNCA:
    """Large Language Neural Cellular Automata"""

    def __init__(self, config: LLNCAConfig, state: None):
        self.config = config
        self.dataset = SentenceDataset(
            self.config.sentences_file,
            bin_size=self.config.bin_size,
            trunc_ratio=self.config.trunc_ratio,
        )
        self.renderer = Renderer(self.config.font_name, self.config.font_size)

        self.nca = NCA()
        self.optimizer = optim.Adam(
            self.nca.parameters(), lr=self.config.lr, betas=self.config.betas
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, self.config.lr_gamma
        )

        self.loaded_epoch = 0

        if state is not None:
            self.nca = NCA()
            self.nca.load_state_dict(state["nca"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.loaded_epoch = state["curr_epoch"]

    def train(self):
        print(f"model name: {self.config.name}")
        print(f"sentences file: {self.config.sentences_file}")

        pool = Pool(
            self.dataset, bin=1, renderer=self.renderer, channels=self.config.channels
        )
        self.poolpool = PoolPool([pool])
        curr_epoch = 0

        try:
            for i in tqdm(
                range(self.loaded_epoch, self.config.epochs),
                initial=self.loaded_epoch,
                total=self.config.epochs,
                leave=False,
            ):
                curr_epoch = i
                self.optimizer.zero_grad()
                x = self.poolpool.sample(self.config.batch_size)
                steps = random.randint(x.shape[3], int(x.shape[3] * 1.25))
                y = self.nca(x, steps)
                loss = self.poolpool.update(y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nca.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                if i % 100 == 0:
                    tqdm.write(f"epoch {curr_epoch} loss: {loss.item()}")
        except KeyboardInterrupt:
            print("training cancelled")

        self.save(curr_epoch)

    def save(self, curr_epoch):
        state = {
            "config": self.config,
            "curr_epoch": curr_epoch,
            "nca": self.nca.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        save_path = f"{self.config.folder}/{self.config.name}-{curr_epoch + 1}.tar"
        torch.save(state, save_path)
        print(f"model saved: {save_path}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        state = torch.load(sys.argv[1], weights_only=False)
        config = state["config"]
        llnca = LLNCA(config, state)
    else:
        config = LLNCAConfig(
            name="ezpz",
            folder="models",
            sentences_file="data/norm/ezpz.txt",
            font_name="/use/share/fonts/opentype/baby.otf",
            font_size=8,
            bin_size=16,
            trunc_ratio=3,
            epochs=1000,
            batch_size=8,
            channels=16,
        )
        llnca = LLNCA(config, None)
    llnca.train()
