import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise

from tqdm import tqdm


def _count_pairwise(text: str) -> Counter:
    return Counter(pairwise(text))


def print_ttlr(text: str, lim: int = 120, dim=True):
    r_text = repr(text)[1:-1]
    l_r_text = len(r_text)
    l_r_r_text = len(r_text.replace("\\n", ""))
    tqdm.write(
        ("\033[2m" if dim else "")
        + f"[{l_r_r_text}] {r_text[:lim]}"
        + ("..." if l_r_text > lim else "")
        + ("\033[22m" if dim else "")
    )


type LLNCAVocab = dict[int, str | tuple[str, str]]
type LLNCAIVocab = dict[str | tuple[str, str], int]


class LLNCATokenizer:
    def __init__(self, vocab: LLNCAVocab | None = None):
        self.vocab: LLNCAVocab = vocab if vocab is not None else {}

        self.ivocab: LLNCAIVocab = {}
        self.build_ivocab()
        self.ivocab_dirty = False

        self.next_tok = max(255, max(self.vocab.keys())) + 1 if self.vocab else 256

        self.n_workers = os.cpu_count() or 8
        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)

        self.vocab[0] = "\x00"
        for i in range(32, 127):
            self.vocab[i] = chr(i)

    def load_vocab(self, vocab):
        self.vocab = vocab
        self.vocab = {
            (int(k) if isinstance(k, str) else k): (
                (v[0], v[1]) if isinstance(v, list) else v
            )
            for k, v in self.vocab.items()
        }

        self.build_ivocab()
        self.ivocab_dirty = False

        max_tok = max(self.vocab.keys())
        self.next_tok = max_tok + 1

    def train(self, text: str, debug: bool = False):
        prev = time.perf_counter()
        n_merges = self._train(text, debug)
        curr = time.perf_counter()
        diff = curr - prev
        print(f"\033[2mmade {n_merges} merges in {diff:.6f} seconds.\033[0m")

    def _train(self, text: str, debug: bool = False):
        self.ivocab_dirty = True

        chars_unique = set(text)
        for char in chars_unique:
            tok = next(iter(char.encode("utf-8")))
            self.vocab[tok] = char
        self.vocab[0] = "\x00"

        toks_str = text.encode("utf-8").decode("latin-1")
        if debug:
            print_ttlr(toks_str)

        n_merges = 0
        # with tqdm(dynamic_ncols=True, leave=False, unit="merges") as pbar:
        while len(toks_str) > 1:
            counts = self.count_pairs(toks_str)
            # tqdm.write(str(counts))
            if not counts:
                break

            # most_common = counts.most_common(1)[0]
            most_common = (("", ""), 0)
            for pair, count in counts.most_common():
                if count < most_common[1]:
                    break
                if "\n" not in pair:  # and " " not in pair:
                    most_common = (pair, count)

            most_pair, most_count = most_common
            if most_count <= 1:
                break

            self.vocab[self.next_tok] = most_pair
            c1, c2 = most_pair
            toks_str = toks_str.replace(c1 + c2, chr(self.next_tok))

            self.next_tok += 1
            n_merges += 1
            if debug:
                print_ttlr(toks_str)
            # pbar.update()

        return n_merges

    def encode(self, text: str, silent: bool = True, debug: bool = False):
        prev = time.perf_counter()
        toks_str, n_merges = self._encode(text, debug)
        curr = time.perf_counter()
        diff = curr - prev
        if not silent:
            print(f"\033[2mmade {n_merges} merges in {diff:.6f} seconds.\033[0m")
        return toks_str

    def _encode(self, text: str, debug: bool = False):
        if self.ivocab_dirty:
            self.build_ivocab()
            self.ivocab_dirty = False

        toks_str = text.encode("utf-8").decode("latin-1")
        if debug:
            print_ttlr(toks_str)

        n_merges = 0
        while len(toks_str) > 1:
            counts = self.count_pairs(toks_str)
            if not counts:
                break

            best_tok: int = float("inf")  # type: ignore
            best_pair = None

            for pair in counts:
                if pair in self.ivocab:
                    tok_id = self.ivocab[pair]
                    if tok_id < best_tok:
                        best_tok = tok_id
                        best_pair = pair

            if best_pair is None:
                break

            c1, c2 = best_pair
            toks_str = toks_str.replace(c1 + c2, chr(best_tok))

            n_merges += 1

            if debug:
                print_ttlr(toks_str)

        return toks_str, n_merges

    def count_pairs(self, toks_str: str) -> Counter[tuple[str, str]]:
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

    def decode(self, tok_str: str):
        return "".join([self.expand_tok(ord(tok)) for tok in tok_str])

    def expand_tok(self, tok: int):
        toks: list[int] = [tok]
        strs: list[str] = []
        expanded = True
        while expanded:
            new_toks: list[int] = []
            expanded = False
            for _tok in toks:
                sub = self.vocab[_tok]
                if isinstance(sub, tuple):
                    new_toks.extend([ord(sub_e) for sub_e in sub])
                    expanded = True
                elif isinstance(sub, str):
                    new_toks.append(_tok)
            toks = new_toks
        for _tok in toks:
            sub = self.vocab[_tok]
            if isinstance(sub, str):
                strs.append(sub)
            elif isinstance(sub, tuple):
                raise TypeError("token didn't expand to str as expected")
        return "".join(strs)

    def build_ivocab(self):
        self.ivocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

    def __del__(self):
        self.executor.shutdown(wait=False)


def main():
    corp_path = "data/harv/harv.corp.txt"
    corptok_path = "data/harv/harv.corptok.txt"
    vocab_path = "data/harv/harv.vocab.json"

    print("loading text...", end=" ")

    corp = """"""
    with open(corp_path, "r") as file:
        corp = file.read()
    print(f"\033[2m{corp_path}\033[0m")

    tokenizer = LLNCATokenizer()

    print("training...", end=" ")
    tokenizer.train(corp)

    # print("loading vocab...", end=" ")
    # with open(vocab_path, "r") as file:
    #     tokenizer.load_vocab(json.load(file))
    # print(f"\033[2m{vocab_path}\033[0m")

    # print(tokenizer.vocab)

    print("encoding...", end=" ")
    corptok = tokenizer.encode(corp, silent=False)

    print("writing...", end=" ")
    with open(corptok_path, "w") as file:
        file.write(corptok)
    print(f"\033[2m{corptok_path}\033[0m")

    print("saving...", end=" ")
    with open(vocab_path, "w") as file:
        json.dump(tokenizer.vocab, file)
    print(f"\033[2m{vocab_path}\033[0m")

    return tokenizer


if __name__ == "__main__":
    main()
