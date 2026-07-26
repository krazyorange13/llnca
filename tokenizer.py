import time
from collections import Counter
from itertools import pairwise
from typing import Literal


class Tokenizer:
    def __init__(self):
        self.vocab: dict[int, str | tuple[int, int]] = {}

    def train(self, text: str, verbosity: Literal[0, 1, 2]):
        prev = time.perf_counter()
        n_merges = self._train(text, verbosity)
        curr = time.perf_counter()
        diff = curr - prev
        print(f"made {n_merges} merges in {diff:.6f} seconds.")

    def _train(self, text: str, verbosity: Literal[0, 1, 2]):
        chars_unique = set(text)
        self.vocab = {}
        for char in chars_unique:
            tok = next(iter(char.encode("utf-8")))
            self.vocab[tok] = char

        toks = list(text.encode("utf-8"))

        n_merges = 0
        next_tok = 256
        while len(toks) > 1:
            if verbosity >= 2:
                print("|".join([self.expand_tok(tok) for tok in toks]))

            counts = self.count_pairs(toks)
            most_common = counts.most_common(1)[0]
            most_pair, most_count = most_common
            if most_count == 1:
                break

            self.vocab[next_tok] = most_pair
            toks = self.merge_pair(toks, next_tok, most_pair)

            if verbosity >= 1:
                print(
                    f"MERGE: ({self.expand_tok(most_pair[0])!r}, {self.expand_tok(most_pair[1])!r}) {most_pair} -> {next_tok}"
                )

            next_tok += 1
            n_merges += 1

        return n_merges

    def merge_pair(self, toks: list[int], new_tok: int, pair: tuple[int, int]):
        new_toks: list[int] = []
        i = 0
        while i < len(toks):
            if i != len(toks) - 1 and toks[i] == pair[0] and toks[i + 1] == pair[1]:
                new_toks.append(new_tok)
                i += 1
            else:
                new_toks.append(toks[i])
            i += 1
        return new_toks

    def count_pairs(self, toks: list[int]):
        counter: Counter[tuple[int, int]] = Counter()
        for pair in pairwise(toks):
            counter[pair] += 1
        return counter

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
                    new_toks.extend(sub)
                    expanded = True
                elif isinstance(sub, str):
                    new_toks.append(_tok)
            toks = new_toks
        for _tok in toks:
            sub = self.vocab[_tok]
            if isinstance(sub, str):
                strs.append(sub)
            elif isinstance(sub, tuple):
                raise TypeError(
                    ">w< oopsies not good! token didn't expand to str as expected :("
                )
        return "".join(strs)


if __name__ == "__main__":
    print("loading text...")
    text = """"""
    with open("data/norm/harvsents.txt", "r") as file:
        text = file.read()

    tokenizer = Tokenizer()
    print("training...")
    tokenizer.train(text, verbosity=0)
