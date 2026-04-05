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
            nn.Conv2d(self.channels * 4, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, self.channels, kernel_size=1, bias=False),
        )

        with torch.no_grad():
            self.seq[-1].weight.zero_()  # type: ignore

    def step(self, x, freeze_mask=None, update_rate=None):
        pre_alive_mask = self.get_alive_mask(x)

        y = self.perception(x)
        y = self.seq(y)

        update_mask = self.get_update_mask(y.shape, update_rate)
        if freeze_mask is not None:
            x = x + y * update_mask * freeze_mask
        else:
            x = x + y * update_mask

        post_alive_mask = self.get_alive_mask(x)
        combined_mask = pre_alive_mask * post_alive_mask
        x = x * combined_mask

        return x

    def get_update_mask(self, shape, update_rate=None):
        b, _, h, w = shape
        update_rate = update_rate or self.update_rate
        update_mask = (torch.rand(b, 1, h, w) < update_rate).float()
        return update_mask

    def get_alive_mask(self, x, threshold=0.01):
        # x = F.pad(x, (1, 1, 1, 1), mode="circular")
        # sum across all channels, if the cells have any values then they're good idrk
        alpha = x[:, :2, :, :]
        pool = (
            F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1)
            .abs()
            .sum(dim=1)
            .unsqueeze(1)
        )
        mask = pool > threshold
        return mask.float()

    def forward(self, x, steps=1, freeze_mask=None, update_rate=None):
        for i in range(steps):
            x = self.step(x, freeze_mask=freeze_mask, update_rate=update_rate)
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

    def bbox(self, text):
        return self.font.getbbox(text)


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
            # load sentences from a file with one sentence per line
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
        rows = [self.get_row() for i in range(self.pool_size)]
        sentences, seeds, freezes = zip(*rows)
        self.ys = torch.stack(sentences)
        self.xs = torch.stack(seeds)
        self.fs = torch.stack(freezes)

    def sample(self, batch_size=8, damaged=3):
        self.idxs = torch.randperm(self.pool_size)[:batch_size]
        self.y_batch = self.ys[self.idxs]
        self.x_batch = self.xs[self.idxs]
        self.f_batch = self.fs[self.idxs]

        losses = F.mse_loss(
            self.x_batch[:, :1, :, :],
            self.y_batch,
            reduction="none",
        ).sum(dim=(1, 2, 3))
        sorted_idxs = torch.argsort(losses)

        replace_idx = sorted_idxs[-1]
        sentence, seed, freeze_mask = self.get_row()
        self.y_batch[replace_idx] = sentence
        self.x_batch[replace_idx] = seed
        self.f_batch[replace_idx] = freeze_mask

        damaged_idxs = sorted_idxs[:damaged]
        self.x_batch[damaged_idxs] = self.damage(
            self.x_batch[damaged_idxs], self.f_batch[damaged_idxs]
        )

        return self.x_batch, self.f_batch

    def get_row(self):
        farm = "p'" + "w" * (self.dataset.bin_size * (self.bin + 0) + 1)
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

        freeze_bbox = self.renderer.bbox(seed)
        x1, y1, x2, y2 = freeze_bbox
        freeze_t = torch.ones(seed_t.shape)
        freeze_t[:c, y1 - 1 : y2 - 1, x1:x2] = 0.0

        return sentence_t, seed_t, freeze_t

    def damage(self, x_batch, f_batch):
        b, c, h, w = x_batch.shape
        grid = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        grid = torch.stack(grid, dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        center = torch.rand(b, 2, 1, 1) - 0.5
        radius = 0.3 * torch.rand(b, 1, 1, 1) + 0.1
        mask = ((grid - center) * (grid - center)).sum(1, keepdim=True).sqrt() > radius
        return x_batch * mask.float()

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


@dataclass
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
    backprop_chunk: int = 32
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

        self.nca = NCA(channels=self.config.channels)
        self.optimizer = optim.Adam(
            self.nca.parameters(), lr=self.config.lr, betas=self.config.betas
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, self.config.lr_gamma
        )

        self.loaded_epoch = 0

        if state is not None:
            self.nca = NCA(channels=self.config.channels)
            self.nca.load_state_dict(state["nca"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.loaded_epoch = state["curr_epoch"]

    def loss(self, x):
        mse_loss = self.poolpool.update(x)

        # alive_mask = (self.nca.get_alive_mask(x) > 0).expand(-1, 16, -1, -1)
        # alive_x = x[alive_mask]
        # hidden_loss = 0.01 / (torch.std(alive_x) + 1e-4)

        # return mse_loss + hidden_loss

        return mse_loss

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
                x, freeze_mask = self.poolpool.sample(self.config.batch_size)

                total_steps = random.randint(x.shape[3], int(x.shape[3] * 1.25))
                acc_loss = 0
                for step_idx in range(0, total_steps, self.config.backprop_chunk):
                    chunk_steps = min(
                        self.config.backprop_chunk, total_steps - step_idx
                    )
                    x = self.nca(x, chunk_steps, freeze_mask)
                    loss = self.loss(x)
                    loss.backward()
                    x = x.detach()
                    x = torch.clamp(x, -2.0, 2.0)
                    acc_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.nca.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                if i % 100 == 0:
                    tqdm.write(f"epoch {curr_epoch} loss: {acc_loss}")

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
    if len(sys.argv) in [2, 3]:
        state = torch.load(sys.argv[1], weights_only=False)
        config = state["config"]

        if len(sys.argv) == 3:
            new_epochs = int(sys.argv[2])
            config.epochs = new_epochs

        llnca = LLNCA(config, state)
    else:
        config = LLNCAConfig(
            name="beta",
            folder="models",
            sentences_file="data/norm/ezpzr.txt",
            font_name="/use/share/fonts/opentype/baby.otf",
            font_size=8,
            bin_size=16,
            trunc_ratio=3,
            epochs=8000,
            batch_size=8,
            channels=32,
            backprop_chunk=32,
        )
        llnca = LLNCA(config, None)

    llnca.train()
