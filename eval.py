import sys
import time

import torch

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
    if len(sys.argv) < 2:
        print("\033[91merror:\033[0m model path required")
        sys.exit(1)

    path = sys.argv[1]

    llnca = LLNCA(
        checkpoint=torch.load(
            path,
            weights_only=False,
            map_location=torch.device("cpu"),
        )
    )

    # llnca1 = LLNCA(
    #     checkpoint=torch.load(
    #         "models/alpha-c-mini-20000.pth",
    #         weights_only=False,
    #         map_location=torch.device("cpu"),
    #     )
    # )
    # llnca2 = LLNCA(
    #     checkpoint=torch.load(
    #         "models/alpha-c-nano-20000.pth",
    #         weights_only=False,
    #         map_location=torch.device("cpu"),
    #     )
    # )
    # embds1 = llnca1.embeddings.embeddings.weight
    # embds2 = llnca2.embeddings.embeddings.weight
    # diff = embds2 - embds1
    # print("embds2 stats")
    # print(embds2.detach())
    # print("min", embds2.min().detach())
    # print("max", embds2.max().detach())
    # print("mean", embds2.mean().detach())
    # print("std", embds2.std().detach())
    # print("diff stats")
    # print(diff)
    # print("min", diff.min().detach())
    # print("max", diff.max().detach())
    # print("mean", diff.mean().detach())
    # print("std", diff.std().detach())

    while True:
        prev_layers = None
        prev_frame = ""
        for layers, frame in llnca.eval():
            frame_str = llnca.tokenizer.decode(frame)
            if prev_layers is None:
                prev_layers = layers
                layers_diff = None
            else:
                layers_diff = layers - prev_layers
                prev_layers = layers
            # print("\33[2K\r" + (f"(delta mean={layers_diff.mean().item()} std={layers_diff.std().item()}) " if layers_diff is not None else "") + frame_str, end="", flush=True)
            print("\33[2K\r" + frame_str, end="", flush=True)
            if prev_frame != frame_str:
                time.sleep(0.25)
            prev_frame = frame_str
