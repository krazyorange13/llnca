from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from tokenizer import LLNCATokenizer, main


@dataclass
class LLNCAEmbeddingConfig:
    n_dims: int


class LLNCAEmbedding:
    def __init__(
        self,
        tokenizer: LLNCATokenizer,
        config: LLNCAEmbeddingConfig,
        debug: bool = False,
    ):
        self.tokenizer = tokenizer
        self.config = config

        self.embeddings = nn.Embedding(
            num_embeddings=len(self.tokenizer),
            embedding_dim=self.config.n_dims,
        )

        self.vocab_embed_map: dict[int, int] = {
            k: i for i, (k, v) in enumerate(self.tokenizer.vocab.items())
        }

        if debug:
            print("\033[2membeddings\033[0m")
            with np.printoptions(linewidth=10000, precision=4, suppress=True):

                def print_embed(tok):
                    embedding_i = self.vocab_embed_map[tok]
                    embedding = self.embeddings.weight[embedding_i].detach().numpy()
                    print(f"  \033[2m[{tok}]\t{embedding}\033[0m")

                for tok in list(self.tokenizer.vocab.keys())[:5]:
                    print_embed(tok)
                print("  \033[2m...\033[0m")
                for tok in list(self.tokenizer.vocab.keys())[-5:]:
                    print_embed(tok)

    def load(self, state):
        self.embeddings.load_state_dict(state)

    def save(self):
        return self.embeddings.state_dict()


if __name__ == "__main__":
    tokenizer = main()
    print("loading embeddings...")
    config = LLNCAEmbeddingConfig(n_dims=8)
    embedding = LLNCAEmbedding(tokenizer=tokenizer, config=config, debug=True)
