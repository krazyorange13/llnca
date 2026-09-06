import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from itertools import pairwise

from tqdm import tqdm

NEWLINE_ID = 10
_SURROGATE_START = 0xD800
_SURROGATE_END = 0xDFFF
_MAX_CODEPOINT = 0x10FFFF


def _skip_surrogates(tok_id: int) -> int:
    if _SURROGATE_START <= tok_id <= _SURROGATE_END:
        return _SURROGATE_END + 1
    return tok_id


_worker_ivocab: dict = {}


def _init_encode_worker(ivocab: dict):
    global _worker_ivocab
    _worker_ivocab = ivocab


def _encode_line_worker(line: str) -> tuple[str, int]:
    if len(line) < 2:
        return line, 0

    ivocab = _worker_ivocab
    n_merges = 0
    while True:
        pairs = set(pairwise(line))
        available = {p: ivocab[p] for p in pairs if p in ivocab}

        if not available:
            break

        best_pair = min(available, key=available.get)  # type: ignore
        line = line.replace(best_pair[0] + best_pair[1], chr(available[best_pair]))
        n_merges += 1

    return line, n_merges


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
type LLNCAIVocab = dict[tuple[str, str], int]


class LLNCATokenizer:
    def __init__(self, vocab: LLNCAVocab | None = None):
        self.vocab: LLNCAVocab = vocab if vocab is not None else {}
        self.ivocab: LLNCAIVocab = {}
        self.ivocab_dirty = False

        self.next_tok = max(255, max(self.vocab.keys())) + 1 if self.vocab else 256
        self.next_tok = _skip_surrogates(self.next_tok)

        self.n_workers = os.cpu_count() or 8
        self._encode_executor: ProcessPoolExecutor | None = None
        self._encode_executor_vocab_id: int | None = None

        if not self.vocab:
            self.vocab[0] = "\x00"
            for i in range(32, 127):
                self.vocab[i] = chr(i)

        self.build_ivocab()

    def load_vocab(self, vocab):
        self.vocab = {
            (int(k) if isinstance(k, str) else k): (
                tuple(v) if isinstance(v, list) else v
            )
            for k, v in vocab.items()
        }

        self.build_ivocab()
        self.ivocab_dirty = False
        self.next_tok = max(self.vocab.keys()) + 1
        self.next_tok = _skip_surrogates(self.next_tok)

    def train(self, text: str, debug: bool = False):
        prev = time.perf_counter()
        n_merges = self._train(text, debug)
        curr = time.perf_counter()
        diff = curr - prev
        print(f"\033[2mmade {n_merges} merges in {diff:.6f} seconds.\033[0m")

    def _train(self, text: str, debug: bool = False):
        self.ivocab_dirty = True

        for char in set(text):
            b = next(iter(char.encode("utf-8")))
            self.vocab[b] = char
        self.vocab[0] = "\x00"

        toks_str = text.encode("utf-8").decode("latin-1")
        n = len(toks_str)

        tok = [ord(c) for c in toks_str]
        nxt = list(range(1, n)) + [-1]
        prev = [-1] + list(range(n - 1))
        dead = bytearray(n)

        pair_count: dict[tuple[int, int], int] = {}
        occ: dict[tuple[int, int], set] = {}
        buckets: dict[int, dict] = {}
        current_max = 0

        def note_pair(p, idx):
            nonlocal current_max
            old_c = pair_count.get(p, 0)
            new_c = old_c + 1
            if old_c:
                b = buckets.get(old_c)
                if b is not None:
                    b.pop(p, None)
            pair_count[p] = new_c
            buckets.setdefault(new_c, {})[p] = None
            s = occ.get(p)
            if s is None:
                occ[p] = s = set()
            s.add(idx)
            if new_c > current_max:  # noqa: PLR1730
                current_max = new_c

        def forget_pair(p, idx):
            old_c = pair_count.get(p, 0)
            s = occ.get(p)
            if s is not None:
                s.discard(idx)
            b = buckets.get(old_c)
            if b is not None:
                b.pop(p, None)
            new_c = old_c - 1
            if new_c <= 0:
                pair_count.pop(p, None)
                occ.pop(p, None)
            else:
                pair_count[p] = new_c
                buckets.setdefault(new_c, {})[p] = None

        for i in range(n - 1):
            a, b = tok[i], tok[i + 1]
            if a == NEWLINE_ID or b == NEWLINE_ID:
                continue
            note_pair((a, b), i)

        n_merges = 0
        next_tok = self.next_tok

        with tqdm(desc="training", unit="merge") as pbar:
            while current_max > 1:
                bucket = buckets.get(current_max)
                if not bucket:
                    current_max -= 1
                    continue

                if next_tok > _MAX_CODEPOINT:
                    break

                pair = next(iter(bucket))
                del bucket[pair]

                a, b = pair
                new_id = next_tok
                self.vocab[new_id] = (chr(a), chr(b))

                positions = sorted(occ.get(pair, ()))
                for i in positions:
                    if dead[i]:
                        continue
                    r = nxt[i]
                    if r == -1 or dead[r] or tok[i] != a or tok[r] != b:
                        continue

                    l = prev[i]
                    r2 = nxt[r]

                    if l != -1 and tok[l] != NEWLINE_ID:
                        forget_pair((tok[l], a), l)
                    if r2 != -1 and tok[r2] != NEWLINE_ID:
                        forget_pair((b, tok[r2]), r)

                    tok[i] = new_id
                    nxt[i] = r2
                    if r2 != -1:
                        prev[r2] = i
                    dead[r] = 1

                    if l != -1 and tok[l] != NEWLINE_ID:
                        note_pair((tok[l], new_id), l)
                    if r2 != -1 and tok[r2] != NEWLINE_ID:
                        note_pair((new_id, tok[r2]), i)

                final_c = pair_count.pop(pair, None)
                if final_c is not None:
                    b2 = buckets.get(final_c)
                    if b2 is not None:
                        b2.pop(pair, None)
                occ.pop(pair, None)

                next_tok += 1
                next_tok = _skip_surrogates(next_tok)
                n_merges += 1
                pbar.update(1)

                if debug:
                    print_ttlr(self._reconstruct(tok, nxt, dead, n))

        self.next_tok = next_tok
        return n_merges

    @staticmethod
    def _reconstruct(tok, nxt, dead, n):
        out = []
        i = 0
        while i != -1:
            if not dead[i]:
                out.append(chr(tok[i]))
            i = nxt[i]
        return "".join(out)

    def _ensure_encode_executor(self):
        vocab_id = id(self.ivocab) if not self.ivocab_dirty else None
        if self.ivocab_dirty:
            self.build_ivocab()
            self.ivocab_dirty = False
            vocab_id = id(self.ivocab)

        if self._encode_executor is None or self._encode_executor_vocab_id != vocab_id:
            if self._encode_executor is not None:
                self._encode_executor.shutdown(wait=True)
            self._encode_executor = ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=_init_encode_worker,
                initargs=(self.ivocab,),
            )
            self._encode_executor_vocab_id = vocab_id

        return self._encode_executor

    def encode(self, text: str, silent: bool = True, debug: bool = False):
        prev = time.perf_counter()
        toks_str, n_merges = self._encode(text, debug)
        curr = time.perf_counter()
        diff = curr - prev
        if not silent:
            print(f"\033[2mmade {n_merges} merges in {diff:.6f} seconds.\033[0m")
        return toks_str

    def _encode(self, text: str, debug: bool = False):
        executor = self._ensure_encode_executor()

        toks_str = text.encode("utf-8").decode("latin-1")
        lines = toks_str.split("\n")

        chunksize = max(1, len(lines) // (self.n_workers * 4))

        encoded_lines = []
        total_merges = 0

        for res_line, merges in tqdm(
            executor.map(_encode_line_worker, lines, chunksize=chunksize),
            total=len(lines),
            desc="encoding",
            leave=False,
        ):
            encoded_lines.append(res_line)
            total_merges += merges

        res_str = "\n".join(encoded_lines)

        if debug:
            print_ttlr(res_str)

        return res_str, total_merges

    def decode(self, tok_str: str):
        return "".join([self.expand_tok(ord(tok)) for tok in tok_str])

    @cache
    def expand_tok(self, tok: int):
        sub = self.vocab[tok]
        if isinstance(sub, str):
            return sub
        return "".join(self.expand_tok(ord(c)) for c in sub)

    def build_ivocab(self):
        self.ivocab = {v: k for k, v in self.vocab.items() if isinstance(v, tuple)}

    def __len__(self):
        return len(self.vocab)

    def __del__(self):
        if self._encode_executor is not None:
            self._encode_executor.shutdown(wait=False)


def main():
    corp_path = "data/plato/platos-republic.corp.txt"
    corptok_path = "data/plato/platos-republic.corptok.txt"
    vocab_path = "data/plato/platos-republic.vocab.json"

    print("loading text...", end=" ")

    corp = """"""
    with open(corp_path, "r") as file:
        corp = file.read()
    print(f"\033[2m{corp_path}\033[0m")

    tokenizer = LLNCATokenizer()

    print("training...", end=" ")
    tokenizer.train(corp)

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
