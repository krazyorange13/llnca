import json
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from tokenizer import LLNCATokenizer


@dataclass
class LLNCAEmbeddingsConfig:
    n_dims: int


class LLNCAEmbeddings:
    def __init__(
        self,
        tokenizer: LLNCATokenizer,
        config: LLNCAEmbeddingsConfig,
        debug: bool = False,
    ):
        self.tokenizer = tokenizer
        self.config = config

        self.embeddings = nn.Embedding(
            num_embeddings=len(self.tokenizer),
            embedding_dim=self.config.n_dims,
            padding_idx=0,
        )

        self.tok_embed_map: dict[int, int] = {
            k: i for i, (k, v) in enumerate(sorted(self.tokenizer.vocab.items()))
        }
        self.i_tok_embed_map: dict[int, int] = {
            v: k for k, v in self.tok_embed_map.items()
        }
        self.tok_str_embed_map: dict[str, int] = {
            (k if isinstance(k, str) else k[0] + k[1]): self.tok_embed_map[v]
            for k, v in self.tokenizer.ivocab.items()
        }

        if debug:
            sorted_keys = sorted(self.tokenizer.vocab.keys())
            for tok in sorted_keys[:5]:
                self.print_embed(tok)
            print("  \033[2m...\033[0m")
            for tok in sorted_keys[-5:]:
                self.print_embed(tok)

            print(f"  \033[2mn embeds: {len(self.embeddings.weight)}\033[0m")
            print(f"  \033[2mn params: {self.embeddings.weight.numel()}\033[0m")

    def embed_from_tok(self, tok: int):
        embed_i = self.tok_embed_map[tok]
        return self.embeddings(
            torch.tensor(embed_i, device=self.embeddings.weight.device)
        )

    def print_embed(self, tok: int):
        with np.printoptions(linewidth=10000, precision=4, suppress=True):
            embedding_i = self.tok_embed_map[tok]
            embedding = self.embeddings.weight[embedding_i].detach().numpy()
            print(f"  \033[2m[{tok}]\t{embedding}\033[0m")

    def print_embed_str(self, tok_str: str):
        with np.printoptions(linewidth=10000, precision=4, suppress=True):
            embedding_i = self.tok_str_embed_map[tok_str]
            embedding = self.embeddings.weight[embedding_i].detach().numpy()
            print(f"  \033[2m[{tok_str}]\t{embedding}\033[0m")

    def load(self, state):
        self.embeddings.load_state_dict(state)

    def save(self):
        return self.embeddings.state_dict()


if __name__ == "__main__":
    # corp_path = "data/harv/harv.corp.txt"
    # corptok_path = "data/harv/harv.corptok.txt"
    vocab_path = "data/harv/harv.vocab.json"

    tokenizer = LLNCATokenizer()
    print("loading vocab...", end=" ")
    with open(vocab_path, "r") as file:
        tokenizer.load_vocab(json.load(file))
    print(f"\033[2m{vocab_path}\033[0m")

    print("loading embeddings...")
    config = LLNCAEmbeddingsConfig(n_dims=16)
    embedding = LLNCAEmbeddings(tokenizer=tokenizer, config=config, debug=True)
