import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise

from tqdm import tqdm


def _count_pairwise(text: str) -> Counter:
    return Counter(pairwise(text))


class Tokenizer:
    def __init__(self):
        self.vocab: dict[int, str | tuple[str, str]] = {}
        self.n_workers = os.cpu_count() or 8
        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)

    def __del__(self):
        self.executor.shutdown(wait=False)

    def train(self, text: str):
        prev = time.perf_counter()
        n_merges = self._train(text)
        curr = time.perf_counter()
        diff = curr - prev
        print(f"made {n_merges} merges in {diff:.6f} seconds.")

    def _train(self, text: str):
        chars_unique = set(text)
        self.vocab = {}
        for char in chars_unique:
            tok = next(iter(char.encode("utf-8")))
            self.vocab[tok] = char

        toks_str = text.encode("utf-8").decode("latin-1")

        n_merges = 0
        next_tok = 256
        with tqdm(dynamic_ncols=True, leave=False, unit="merges") as pbar:
            while len(toks_str) > 1:
                counts = self.count_pairs(toks_str)
                if not counts:
                    break

                most_common = counts.most_common(1)[0]
                most_pair, most_count = most_common
                if most_count == 1:
                    break

                self.vocab[next_tok] = most_pair
                c1, c2 = most_pair
                toks_str = toks_str.replace(c1 + c2, chr(next_tok))

                next_tok += 1
                n_merges += 1
                pbar.update()

        return n_merges

    def count_pairs(self, toks_str: str):
        n_toks = len(toks_str)

        if n_toks < 100_000:
            return Counter(pairwise(toks_str))

        buf_sz = n_toks // self.n_workers
        bufs = []

        for i in range(self.n_workers):
            start = i * buf_sz
            end = start + buf_sz + (1 if i < self.n_workers - 1 else 0)
            bufs.append(toks_str[start:end])

        counts = Counter()
        for _counts in self.executor.map(_count_pairwise, bufs):
            counts.update(_counts)

        return counts


if __name__ == "__main__":
    print("loading text...")
    text = """"""
    with open("data/norm/sentences.txt", "r") as file:
        text = file.read()

    tokenizer = Tokenizer()
    print("training...")
    tokenizer.train(text)
