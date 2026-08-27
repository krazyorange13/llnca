import json
from dataclasses import dataclass

import torch
from torch import optim
from tqdm import tqdm

from corpus import LLNCACorpus, LLNCACorpusConfig
from dataset import (
    LLNCADataSampler,
    LLNCADataSamplerConfig,
    LLNCADataset,
    LLNCADatasetConfig,
)
from embedding import LLNCAEmbedding, LLNCAEmbeddingConfig
from nca import LLNCANCA, LLNCANCAConfig
from tokenizer import LLNCATokenizer, LLNCAVocab


@dataclass
class LLNCAOptimConfig:
    lr: float
    lr_gamma: float
    weight_decay: float
    betas: tuple[float, float]


@dataclass
class LLNCAGenConfig:
    nca: LLNCANCAConfig
    optim: LLNCAOptimConfig


@dataclass
class LLNCAAdvConfig:
    nca: LLNCANCAConfig
    optim: LLNCAOptimConfig


@dataclass
class LLNCAConfig:
    name: str
    folder: str
    corpus: LLNCACorpusConfig
    dataset: LLNCADatasetConfig
    sampler: LLNCADataSamplerConfig
    embedding: LLNCAEmbeddingConfig
    vocab: LLNCAVocab
    gen: LLNCAGenConfig
    adv: LLNCAAdvConfig
    # gen_nca: LLNCANCAConfig
    # adv_nca: LLNCANCAConfig
    # gen_optim: LLNCAOptimConfig
    # adv_optim: LLNCAOptimConfig
    n_epochs: int
    batch_size: int
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

        print("loading...", end=" ")
        self.corpus = LLNCACorpus(config=self.config.corpus)
        print("\033[2mcorpus\033[0m ", end=" ")
        self.tokenizer = LLNCATokenizer(vocab=self.config.vocab)
        print("\033[2mtokenizer\033[0m ", end=" ")
        self.embeddings = LLNCAEmbedding(self.tokenizer, config=self.config.embedding)
        print("\033[2membeddings\033[0m ", end=" ")
        self.dataset = LLNCADataset(config=self.config.dataset)
        print("\033[2mdataset\033[0m ", end=" ")
        self.sampler = LLNCADataSampler(self.dataset, config=self.config.sampler)
        print("\033[2msampler\033[0m ", end=" ")
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset, sampler=self.sampler
        )
        print("\033[2mdataloader\033[0m ", end=" ")

        self.gen_nca = LLNCANCA(config=self.config.gen.nca)
        print("\033[2mgen_nca\033[0m ", end=" ")
        self.adv_nca = LLNCANCA(config=self.config.adv.nca)
        print("\033[2madv_nca\033[0m ", end=" ")

        gen_optim_conf = self.config.gen.optim
        adv_optim_conf = self.config.adv.optim
        self.gen_optim = optim.AdamW(
            self.gen_nca.parameters(),
            lr=gen_optim_conf.lr,
            betas=gen_optim_conf.betas,
            weight_decay=gen_optim_conf.weight_decay,
        )
        print("\033[2mgen_optim\033[0m ", end=" ")
        self.adv_optim = optim.AdamW(
            self.adv_nca.parameters(),
            lr=adv_optim_conf.lr,
            betas=adv_optim_conf.betas,
            weight_decay=adv_optim_conf.weight_decay,
        )
        print("\033[2madv_optim\033[0m ", end=" ")
        self.gen_scheduler = optim.lr_scheduler.ExponentialLR(
            self.gen_optim, gen_optim_conf.lr_gamma
        )
        print("\033[2mgen_scheduler\033[0m ", end=" ")
        self.adv_scheduler = optim.lr_scheduler.ExponentialLR(
            self.adv_optim, adv_optim_conf.lr_gamma
        )
        print("\033[2madv_scheduler\033[0m ", end=" ")
        print()

        if checkpoint is not None:
            self.gen_nca.load_state_dict(checkpoint["gen_nca"])
            self.adv_nca.load_state_dict(checkpoint["adv_nca"])
            self.gen_optim.load_state_dict(checkpoint["gen_optim"])
            self.adv_optim.load_state_dict(checkpoint["adv_optim"])
            self.gen_scheduler.load_state_dict(checkpoint["gen_scheduler"])
            self.adv_scheduler.load_state_dict(checkpoint["adv_scheduler"])

        self.curr_epoch = 0

        print(f"name: {self.config.name}")

    def make_checkpoint(self):
        state = {
            "config": self.config,
            "gen_nca": self.gen_nca.state_dict(),
            "adv_nca": self.adv_nca.state_dict(),
            "gen_optim": self.gen_optim.state_dict(),
            "adv_optim": self.adv_optim.state_dict(),
            "gen_scheduler": self.gen_scheduler.state_dict(),
            "adv_scheduler": self.adv_scheduler.state_dict(),
            "curr_epoch": self.curr_epoch,
        }
        return state

    def save(self, path):
        state = self.make_checkpoint()
        torch.save(state, path)

    def train(self):
        self.gen_nca.train()
        self.adv_nca.train()

        for epoch_i in tqdm(
            range(self.curr_epoch, self.config.n_epochs),
            total=self.config.n_epochs,
            leave=True,
            dynamic_ncols=True,
            unit="epoch",
        ):
            for x, y in self.dataloader:
                print(x, y)


# load tokenizer
# load nca
# load gan
# loop:
#   build img
#   apply nca
#   apply gan
#   backprop
#   optimize

if __name__ == "__main__":
    corptok_path = "data/ezpz/ezpz.corptok.txt"
    vocab_path = "data/ezpz/ezpz.vocab.json"

    corpus_config = LLNCACorpusConfig(trunc_ratio=0.7, trunc_split=" ")
    dataset_config = LLNCADatasetConfig(file=corptok_path)
    sampler_config = LLNCADataSamplerConfig(
        batch_len=8,
        drop_last=False,
        shuffle=True,
        bin_interval=8,
    )
    embedding_config = LLNCAEmbeddingConfig(n_dims=16)

    tokenizer = LLNCATokenizer()
    with open(vocab_path) as file:
        tokenizer.load_vocab(json.load(file))
    vocab = tokenizer.vocab

    gen_config = LLNCAGenConfig(
        LLNCANCAConfig(
            channels=64,
            mlp_width=256,
            mlp_depth=8,
            activation_fn="ReLU",
            update_rate=0.25,
            alive_threshold=0.01,
        ),
        LLNCAOptimConfig(
            lr=1e-3,
            lr_gamma=0.9999,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        ),
    )

    adv_config = LLNCAAdvConfig(
        LLNCANCAConfig(
            channels=64,
            mlp_width=256,
            mlp_depth=8,
            activation_fn="ReLU",
            update_rate=0.25,
            alive_threshold=0.01,
        ),
        LLNCAOptimConfig(
            lr=1e-3,
            lr_gamma=0.9999,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        ),
    )

    config = LLNCAConfig(
        name="alpha",
        folder="models",
        corpus=corpus_config,
        dataset=dataset_config,
        sampler=sampler_config,
        embedding=embedding_config,
        vocab=vocab,
        gen=gen_config,
        adv=adv_config,
        n_epochs=1000,
        batch_size=8,
        lambda_pxl=0.5,
        lambda_gan=0.5,
    )

    llnca = LLNCA(config=config)
