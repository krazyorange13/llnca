import sys
import threading
import time

import dearpygui.dearpygui as dpg
import numpy as np
import torch

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
    #         path,
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
        for x, y, xs, frame in llnca.eval():
            x_str = llnca.tokenizer.decode(x)
            y_str = llnca.tokenizer.decode(y)
            target_str = f"[\033[2m{y_str[: len(x_str)]}\033[0m{y_str[len(x_str) :]}]"
            frame_str = llnca.tokenizer.decode(frame)
            print("\33[2K\r" + f"{target_str} {frame_str}", end="", flush=True)
            if prev_frame != frame_str:
                time.sleep(0.25)
            prev_frame = frame_str
