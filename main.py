import functools
import json
import math
import posixpath
import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from corpus import (
    LLNCACorpus,
    LLNCACorpusConfig,
    LLNCACorpusSlidingConfig,
    LLNCACorpusTruncConfig,
)
from dataset import (
    LLNCADataSampler,
    LLNCADataSamplerConfig,
    LLNCADataset,
    LLNCADatasetConfig,
)
from embedding import LLNCAEmbeddings2, LLNCAEmbeddingsConfig
from nca import LLNCANCA, LLNCANCAConfig
from saver import (
    LLNCAName,
    LLNCASaver,
    LLNCASaverConfig,
)
from tokenizer2 import LLNCATokenizer2, LLNCATokenizer2Config, LLNCAVocab2

# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)


def log(folder: str = "", prefix: str = ""):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # _args = [arg if not isinstance(arg) for arg in args]
            # input = {"args": args, "kwargs": kwargs}
            print(func.__name__, args, kwargs)
            output = func(*args, **kwargs)
            print(f"\t\t{len(output)}")
            # print(f"\t\t{repr(output)[:200]}")
            # with open(f"{folder}/{prefix}-{func.__name__}", "w") as f:
            #     pass
            return output

        return wrapper

    return decorator


@dataclass
class LLNCAOptimConfig:
    max_lr: float
    min_lr: float
    weight_decay: float
    betas: tuple[float, float]


@dataclass
class LLNCAGenConfig:
    nca: LLNCANCAConfig
    optim: LLNCAOptimConfig
    steps: tuple[int, int]


@dataclass
class LLNCAAdvConfig:
    nca: LLNCANCAConfig
    optim: LLNCAOptimConfig
    steps: tuple[int, int]


@dataclass
class LLNCAConfig:
    saver: LLNCASaverConfig
    corpus: LLNCACorpusConfig
    dataset: LLNCADatasetConfig
    sampler: LLNCADataSamplerConfig
    embeddings: LLNCAEmbeddingsConfig
    tokenizer: LLNCATokenizer2Config
    vocab: LLNCAVocab2
    gen: LLNCAGenConfig
    adv: LLNCAAdvConfig
    n_epochs: int
    lambda_pxl: float
    lambda_gan: float


class LLNCA:
    def __init__(
        self,
        checkpoint: dict | None = None,
        config: LLNCAConfig | None = None,
        debug: bool = False,
    ):
        if checkpoint is None and config is None:
            raise RuntimeError(
                "\033[31m[error]\033[0m either checkpoint or config must be provided."
            )

        if checkpoint is not None and config is not None:
            checkpoint["config"] = config

        self.config: LLNCAConfig = (
            config if checkpoint is None else checkpoint["config"]
        )  # type: ignore

        self.writer = SummaryWriter(log_dir=self.config.saver.logs_folder)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        _epoch_str = ("-" + str(checkpoint["curr_epoch"] + 1)) if checkpoint else ""
        print(f"loading {self.config.saver.name.full}{_epoch_str}...", end=" ")

        self.saver = LLNCASaver(config=self.config.saver)
        print("\033[2msaver\033[0m ", end=" ")

        self.corpus = LLNCACorpus(config=self.config.corpus)
        print("\033[2mcorpus\033[0m ", end=" ")

        self.tokenizer = LLNCATokenizer2(
            config=self.config.tokenizer, vocab=self.config.vocab
        )
        print("\033[2mtokenizer\033[0m ", end=" ")

        self.embeddings = LLNCAEmbeddings2(
            self.tokenizer, config=self.config.embeddings
        )
        self.embeddings.embeddings.to(self.device)
        print("\033[2membeddings\033[0m ", end=" ")

        self.dataset = LLNCADataset(config=self.config.dataset)
        print("\033[2mdataset\033[0m ", end=" ")

        self.sampler = LLNCADataSampler(self.dataset, config=self.config.sampler)
        print("\033[2msampler\033[0m ", end=" ")

        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_sampler=self.sampler
        )
        print("\033[2mdataloader\033[0m ", end=" ")

        self.gen_nca = LLNCANCA(config=self.config.gen.nca).to(self.device)
        # self.gen_nca = torch.compile(self.gen_nca)
        print("\033[2mgen_nca\033[0m ", end=" ")

        self.adv_nca = LLNCANCA(config=self.config.adv.nca).to(self.device)
        # self.adv_nca = torch.compile(self.adv_nca)
        print("\033[2madv_nca\033[0m ", end=" ")

        gen_optim_conf = self.config.gen.optim
        adv_optim_conf = self.config.adv.optim

        self.gen_optim = optim.AdamW(
            list(self.gen_nca.parameters())
            + list(self.embeddings.embeddings.parameters()),
            lr=gen_optim_conf.max_lr,
            betas=gen_optim_conf.betas,
            weight_decay=gen_optim_conf.weight_decay,
        )
        print("\033[2mgen_optim\033[0m ", end=" ")

        self.adv_optim = optim.AdamW(
            self.adv_nca.parameters(),
            lr=adv_optim_conf.max_lr,
            betas=adv_optim_conf.betas,
            weight_decay=adv_optim_conf.weight_decay,
        )
        print("\033[2madv_optim\033[0m ", end=" ")

        self.gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.gen_optim,
            self.config.n_epochs,
            self.config.gen.optim.min_lr,
        )
        print("\033[2mgen_scheduler\033[0m ", end=" ")

        self.adv_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.gen_optim,
            self.config.n_epochs,
            self.config.gen.optim.min_lr,
        )
        print("\033[2madv_scheduler\033[0m ", end=" ")

        print()

        self.curr_epoch = 0

        if checkpoint is not None:
            self.embeddings.embeddings.load_state_dict(checkpoint["embeddings"])
            self.gen_nca.load_state_dict(checkpoint["gen_nca"])
            self.adv_nca.load_state_dict(checkpoint["adv_nca"])
            self.gen_optim.load_state_dict(checkpoint["gen_optim"])
            self.adv_optim.load_state_dict(checkpoint["adv_optim"])
            self.gen_scheduler.load_state_dict(checkpoint["gen_scheduler"])
            self.adv_scheduler.load_state_dict(checkpoint["adv_scheduler"])
            self.curr_epoch = checkpoint["curr_epoch"]

        with warnings.catch_warnings(), self.gen_nca.disable_update_mask():
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module="torch.jit.*"
            )
            self.writer.add_graph(self.gen_nca, self.get_nca_dummy(self.config.gen.nca))

    def get_state(self):
        state = {
            "config": self.config,
            "embeddings": self.embeddings.embeddings.state_dict(),
            "gen_nca": self.gen_nca.state_dict(),
            "adv_nca": self.adv_nca.state_dict(),
            "gen_optim": self.gen_optim.state_dict(),
            "adv_optim": self.adv_optim.state_dict(),
            "gen_scheduler": self.gen_scheduler.state_dict(),
            "adv_scheduler": self.adv_scheduler.state_dict(),
            "curr_epoch": self.curr_epoch,
        }
        return state

    def train(self):
        self.gen_nca.train()
        self.adv_nca.train()

        for epoch_i in tqdm(
            range(self.curr_epoch, self.config.n_epochs),
            initial=self.curr_epoch,
            total=self.config.n_epochs,
            leave=False,
            dynamic_ncols=True,
            unit="epoch",
            smoothing=0.9,
        ):
            self.curr_epoch = epoch_i
            loss_acc = torch.tensor(0.0, device=self.device)

            for x, y in self.dataloader:
                n_steps = random.randint(
                    self.config.gen.steps[0], self.config.gen.steps[1]
                )
                xs, _, y_idxs = self.tok_to_embed(x, y)
                xs = self.gen_nca.add_channels(xs)
                ys_pred = self.gen_nca(xs, steps=n_steps)
                ys_pred = ys_pred[:, : self.config.embeddings.n_dims, :]
                logits = self.embeddings.get_logits(ys_pred)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y_idxs.reshape(-1)
                )

                loss_acc += loss.detach()
                loss.backward()
                self.gen_optim.step()
                self.gen_optim.zero_grad(set_to_none=True)

            self.gen_scheduler.step()

            loss_avg = loss_acc / len(self.dataloader)
            loss_avg = loss_avg.cpu()

            self.writer.add_scalar("loss", loss_avg, epoch_i)
            self.writer.add_scalar("lr", self.adv_scheduler.get_last_lr()[0], epoch_i)

            if (epoch_i + 1) % self.config.saver.epochs_checkpts == 0:
                self.checkpoint()

            if (epoch_i + 1) % self.config.saver.epochs_histogram == 0:
                self.log_histogram()

            if (epoch_i + 1) % self.config.saver.epochs_embeddings == 0:
                self.log_embeddings()

        self.writer.flush()
        self.writer.close()
        self.saver.close()

    def log_histogram(self):
        for name, param in self.gen_nca.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(
                    tag=name,
                    values=param.detach().cpu(),
                    global_step=self.curr_epoch + 1,
                )

    def log_embeddings(self):
        weights = self.embeddings.embeddings.weight.detach().cpu().numpy()
        labels = self.tokenizer.vocab_arr
        self.writer.add_embedding(
            weights,
            labels,
            tag="embeddings",
            global_step=self.curr_epoch + 1,
        )

    def save(self, path):
        state = self.get_state()
        torch.save(state, path)

    def checkpoint(self):
        name = "".join(
            [
                self.config.saver.name.full,
                "-",
                str(self.curr_epoch + 1),
                ".pth",
            ]
        )
        path = posixpath.join(self.config.saver.checkpts_folder, name)
        self.save(path)

    @torch.no_grad()
    def eval(self):
        self.gen_nca.eval()
        self.adv_nca.eval()
        for x, y in self.dataloader:
            n_steps = (self.config.gen.steps[0] + self.config.gen.steps[1]) // 2
            xs, ys, _ = self.tok_to_embed(x, y)
            xs = self.gen_nca.add_channels(xs)
            y_pred = y
            for _ in range(n_steps):
                xs = self.gen_nca(xs, steps=1)
                ys_pred = xs[:, : self.config.embeddings.n_dims, :]
                idxs = self.nearest_embed(ys_pred)
                y_pred = self.reconstruct_str(idxs)
                yield x[0], y[0], xs[0], y_pred[0]

    @torch.no_grad()
    def eval_str(self, x: str):
        x = self.tokenizer.encode(x)
        self.gen_nca.eval()
        self.adv_nca.eval()
        n_steps = (self.config.gen.steps[0] + self.config.gen.steps[1]) // 2
        xs, _, _ = self.tok_to_embed([x], [])
        xs = self.gen_nca.add_channels(xs)
        for _ in range(n_steps):
            xs = self.gen_nca(xs, steps=1)
            ys_pred = xs[:, : self.config.embeddings.n_dims, :]
            idxs = self.nearest_embed(ys_pred)
            y_pred = self.reconstruct_str(idxs)
            yield xs[0], y_pred[0]

    @torch.no_grad()
    def eval_start(self, x: list[str]):
        x = [self.tokenizer.encode(s) for s in x]
        self.gen_nca.eval()
        self.adv_nca.eval()
        xs, _, _ = self.tok_to_embed(x, [""])
        xs = self.gen_nca.add_channels(xs)
        return xs

    @torch.no_grad()
    def eval_step(self, xs: torch.Tensor):
        self.gen_nca.eval()
        self.adv_nca.eval()
        xs = self.gen_nca(xs, steps=1)
        ys_pred = xs[:, : self.config.embeddings.n_dims, :]
        idxs = self.nearest_embed(ys_pred)
        y_pred = self.reconstruct_str(idxs)
        return xs, y_pred

    def reconstruct_str(self, idxs: torch.Tensor):
        # return idxs.cpu().tolist()
        idxs = idxs.cpu()
        strs = []
        for i in range(len(idxs)):
            strs.append(self.tokenizer._encode_toks(idxs[i].tolist()))
        return strs

    def tok_to_embed(self, x_strs: list[str], y_strs: list[str]):
        max_len = max(max(len(s) for s in x_strs), max(len(s) for s in y_strs))
        pad_len = self.get_padded_embed_len(max_len)

        x_idxs = [
            self.tokenizer._decode_toks(x_str)
            + [self.embeddings.pad_tok] * (pad_len - len(x_str))
            for x_str in x_strs
        ]
        y_idxs = [
            self.tokenizer._decode_toks(y_str)
            + [self.embeddings.pad_tok] * (pad_len - len(y_str))
            for y_str in y_strs
        ]

        x_idx_tensor = torch.tensor(x_idxs, dtype=torch.long, device=self.device)
        y_idx_tensor = torch.tensor(y_idxs, dtype=torch.long, device=self.device)

        xs = self.embeddings.embeddings(x_idx_tensor).permute(0, 2, 1)
        ys = self.embeddings.embeddings(y_idx_tensor).permute(0, 2, 1)

        return xs, ys, y_idx_tensor

    def get_padded_embed_len(self, x_len: int):
        bin_size = self.config.sampler.bin_interval
        return math.ceil(x_len / bin_size) * bin_size

    def nearest_embed(self, x: torch.Tensor):
        return self.embeddings.get_logits(x).argmax(dim=-1)

    def get_nca_dummy(self, nca: LLNCANCAConfig):
        return torch.rand(
            self.config.sampler.batch_len,
            nca.channels,
            self.config.sampler.bin_interval,
            device=self.device,
        )


if __name__ == "__main__":
    corptok_path = "data/harv/harv.corptok.txt"
    vocab_path = "data/harv/harv.tokenizer.json"

    corpus_config = LLNCACorpusConfig(
        LLNCACorpusSlidingConfig(
            window_len=15,
            window_space=5,
            word_level=True,
        ),
        LLNCACorpusTruncConfig(
            ratio=0.7,
        ),
    )

    dataset_config = LLNCADatasetConfig(file=corptok_path)

    sampler_config = LLNCADataSamplerConfig(
        bin_interval=18,
        batch_len=16,
        drop_last=False,
        shuffle=True,
    )

    embedding_config = LLNCAEmbeddingsConfig(
        n_dims=12,
    )

    tokenizer_config = LLNCATokenizer2Config(
        special_tokens=["<pad>"],
        max_token_length=4,
    )
    with open(vocab_path) as file:
        vocab = json.load(file)

    gen_config = LLNCAGenConfig(
        LLNCANCAConfig(
            channels=32,
            mlp_width=96,
            mlp_depth=8,
            activation_fn="ReLU",
            update_rate=0.5,
            alive_threshold=0.01,
        ),
        LLNCAOptimConfig(
            max_lr=1e-3,
            min_lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        ),
        steps=(20, 30),
    )

    adv_config = LLNCAAdvConfig(
        LLNCANCAConfig(
            channels=32,
            mlp_width=96,
            mlp_depth=8,
            activation_fn="ReLU",
            update_rate=0.5,
            alive_threshold=0.01,
        ),
        LLNCAOptimConfig(
            max_lr=1e-3,
            min_lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        ),
        steps=(20, 30),
    )

    name_config = LLNCAName(
        major="harv",
        minor="realpha",
    )

    saver_config = LLNCASaverConfig(
        name=name_config,
        s3_endpoint="https://s3.krazyorange.dev/",
        s3_uri="s3://bucket-0/",
        folder="runs",
        s3_folder="llnca/runs",
        s3_bucket="bucket-0",
        sync_interval=30,
        epochs_checkpts=100,
        epochs_histogram=100,
        epochs_embeddings=100,
    )

    config = LLNCAConfig(
        saver=saver_config,
        corpus=corpus_config,
        dataset=dataset_config,
        sampler=sampler_config,
        embeddings=embedding_config,
        tokenizer=tokenizer_config,
        vocab=vocab,
        gen=gen_config,
        adv=adv_config,
        n_epochs=15000,
        lambda_pxl=0.5,
        lambda_gan=0.5,
    )

    load_dotenv()

    # checkpoint = torch.load("models/harv-beta-3000.pth", weights_only=False)
    # checkpoint["config"].n_epochs = 10000

    llnca = LLNCA(config=config)
    # llnca = LLNCA(
    #     checkpoint=checkpoint,
    #     visdom=visdom,
    # )
    llnca.train()
