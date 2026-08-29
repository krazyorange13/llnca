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
from main import LLNCA, LLNCAAdvConfig, LLNCAConfig, LLNCAGenConfig, LLNCAOptimConfig
from nca import LLNCANCA, LLNCANCAConfig
from tokenizer import LLNCATokenizer, LLNCAVocab

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\033[91merror:\033[0m model path required")

    llnca = LLNCA(
        checkpoint=torch.load(
            "models/alpha-1000.pth",
            weights_only=False,
            map_location=torch.device("cpu"),
        )
    )
    while True:
        prev_frame = ""
        for frame in llnca.eval():
            frame_str = llnca.tokenizer.decode(frame)
            print("\r" + " " * 32 + "\r" + frame_str, end="", flush=True)
            if prev_frame != frame_str:
                time.sleep(0.25)
            prev_frame = frame_str
