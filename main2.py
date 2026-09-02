import json

import torch
from visdom import Visdom

from corpus import LLNCACorpus, LLNCACorpusConfig
from dataset import (
    LLNCADataSampler,
    LLNCADataSamplerConfig,
    LLNCADataset,
    LLNCADatasetConfig,
)
from embedding import LLNCAEmbeddings, LLNCAEmbeddingsConfig
from main import (
    LLNCA,
    LLNCAAdvConfig,
    LLNCACheckpointingConfig,
    LLNCAConfig,
    LLNCAGenConfig,
    LLNCAOptimConfig,
)
from nca import LLNCANCA, LLNCANCAConfig
from tokenizer import LLNCATokenizer, LLNCAVocab

if __name__ == "__main__":
    corptok_path = "data/ezpz2/ezpz2.corptok.txt"
    vocab_path = "data/ezpz2/ezpz2.vocab.json"

    corpus_config = LLNCACorpusConfig(trunc_ratio=0.7, trunc_split=" ")
    dataset_config = LLNCADatasetConfig(file=corptok_path)
    sampler_config = LLNCADataSamplerConfig(
        bin_interval=16,
        batch_len=16,
        drop_last=False,
        shuffle=True,
    )
    embedding_config = LLNCAEmbeddingsConfig(
        n_dims=8,
    )

    tokenizer = LLNCATokenizer()
    with open(vocab_path) as file:
        tokenizer.load_vocab(json.load(file))
    vocab = tokenizer.vocab

    gen_config = LLNCAGenConfig(
        LLNCANCAConfig(
            channels=24,
            mlp_width=64,
            mlp_depth=8,
            activation_fn="ReLU",
            update_rate=0.5,
            alive_threshold=0.01,
        ),
        LLNCAOptimConfig(
            lr=1e-3,
            lr_gamma=0.99999,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        ),
        steps=(20, 30),
    )

    adv_config = LLNCAAdvConfig(
        LLNCANCAConfig(
            channels=24,
            mlp_width=64,
            mlp_depth=8,
            activation_fn="ReLU",
            update_rate=0.5,
            alive_threshold=0.01,
        ),
        LLNCAOptimConfig(
            lr=1e-3,
            lr_gamma=0.99999,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        ),
        steps=(20, 30),
    )

    checkpointing_config = LLNCACheckpointingConfig(
        major_name="alpha",
        minor_name="c-micro",
        folder="models",
        freq=2000,
    )

    config = LLNCAConfig(
        corpus=corpus_config,
        dataset=dataset_config,
        sampler=sampler_config,
        embeddings=embedding_config,
        vocab=vocab,
        gen=gen_config,
        adv=adv_config,
        checkpointing=checkpointing_config,
        n_epochs=20000,
        lambda_pxl=0.5,
        lambda_gan=0.5,
    )

    visdom = Visdom(server="https://visdom.krazyorange.dev", port=443)
    if not visdom.check_connection():
        visdom = None

    llnca = LLNCA(config=config, visdom=visdom)
    llnca.train()
