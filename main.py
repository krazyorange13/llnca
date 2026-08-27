from dataclasses import dataclass

from corpus import LLNCACorpusConfig, LLNCACorpus
from dataset import (
    LLNCADatasetConfig,
    LLNCADataSamplerConfig,
    LLNCADataset,
    LLNCADataSampler,
)
from tokenizer import LLNCAVocab, LLNCATokenizer
from embedding import LLNCAEmbeddingConfig, LLNCAEmbedding
from nca import LLNCANCAConfig, LLNCANCA

import torch
from torch import optim


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
    epochs: int
    batch_size: int
    lambda_pxl: float
    lambda_gan: float


class LLNCA:
    def __init__(
        self, checkpoint: dict | None = None, config: LLNCAConfig | None = None
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

        self.corpus = LLNCACorpus(config=self.config.corpus)
        self.dataset = LLNCADataset(config=self.config.dataset)
        self.sampler = LLNCADataSampler(self.dataset, config=self.config.sampler)
        self.tokenizer = LLNCATokenizer(vocab=self.config.vocab)
        self.embeddings = LLNCAEmbedding(self.tokenizer, config=self.config.embedding)

        self.gen_nca = LLNCANCA(config=self.config.gen.nca)
        self.adv_nca = LLNCANCA(config=self.config.adv.nca)

        gen_optim_conf = self.config.gen.optim
        adv_optim_conf = self.config.adv.optim
        self.gen_optim = optim.AdamW(
            self.gen_nca.parameters(),
            lr=gen_optim_conf.lr,
            betas=gen_optim_conf.betas,
            weight_decay=gen_optim_conf.weight_decay,
        )
        self.adv_optim = optim.AdamW(
            self.adv_nca.parameters(),
            lr=adv_optim_conf.lr,
            betas=adv_optim_conf.betas,
            weight_decay=adv_optim_conf.weight_decay,
        )
        self.gen_scheduler = optim.lr_scheduler.ExponentialLR(
            self.gen_optim, gen_optim_conf.lr_gamma
        )
        self.adv_scheduler = optim.lr_scheduler.ExponentialLR(
            self.adv_optim, adv_optim_conf.lr_gamma
        )

        if checkpoint is not None:
            self.gen_nca.load_state_dict(checkpoint["gen_nca"])
            self.adv_nca.load_state_dict(checkpoint["adv_nca"])
            self.gen_optim.load_state_dict(checkpoint["gen_optim"])
            self.adv_optim.load_state_dict(checkpoint["adv_optim"])
            self.gen_scheduler.load_state_dict(checkpoint["gen_scheduler"])
            self.adv_scheduler.load_state_dict(checkpoint["adv_scheduler"])

        self.curr_epoch = 0

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


# load tokenizer
# load nca
# load gan
# loop:
#   build img
#   apply nca
#   apply gan
#   backprop
#   optimize
