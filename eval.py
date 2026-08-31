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

    llnca1 = LLNCA(
        checkpoint=torch.load(
            "models/alpha-a-25.pth",
            weights_only=False,
            map_location=torch.device("cpu"),
        )
    )
    llnca2 = LLNCA(
        checkpoint=torch.load(
            "models/alpha-a-1000.pth",
            weights_only=False,
            map_location=torch.device("cpu"),
        )
    )
    embds1 = llnca1.embeddings.embeddings.weight
    embds2 = llnca2.embeddings.embeddings.weight
    diff = embds2 - embds1
    print(diff)
    print("min", diff.min())
    print("max", diff.max())
    print("mean", diff.mean())
    print("std", diff.std())

    while True:
        prev_frame = ""
        for frame in llnca.eval():
            frame_str = llnca.tokenizer.decode(frame)
            print("\33[2K\r" + frame_str, end="", flush=True)
            if prev_frame != frame_str:
                time.sleep(0.25)
            prev_frame = frame_str
