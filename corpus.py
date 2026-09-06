import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class LLNCACorpusSlidingConfig:
    window_len: int
    window_space: int
    word_level: bool


@dataclass
class LLNCACorpusTruncConfig:
    ratio: float = 0.7


@dataclass
class LLNCACorpusConfig:
    sliding: LLNCACorpusSlidingConfig
    trunc: LLNCACorpusTruncConfig


class LLNCACorpus:
    def __init__(self, config: LLNCACorpusConfig):
        self.config = config

    def normalize(self, in_file: str, out_file: str, sentences=False):
        print("normalizing corpus...")
        print(f"\033[2m  in: {in_file}\033[0m")

        with open(in_file) as file:
            contents = file.read()

        contents = re.sub(r"\s+", " ", contents)
        if sentences:
            contents = re.sub(r"[.!?]\s+", ".\n", contents)

        with open(out_file, "w") as file:
            file.write(contents)

        print(f"\033[2m out: {out_file}\033[0m")

    def generate_sliding(self, in_file: str, out_file: str | None = None):
        if out_file is None:
            parts = in_file.split(".")
            out_file = ".".join(parts[:-1]) + ".corp" + "." + parts[-1]

        print("generating corpus (sliding)...")
        print(f"\033[2m  in: {in_file}\033[0m")

        if self.config.sliding.word_level:

            def generator(text):  # type: ignore
                words = text.split()
                for i in range(
                    0,
                    len(words) - self.config.sliding.window_len,
                    self.config.sliding.window_space,
                ):
                    yield " ".join(words[i : i + self.config.sliding.window_len]) + "\n"
        else:

            def generator(text):
                for i in range(
                    0,
                    len(text) - self.config.sliding.window_len,
                    self.config.sliding.window_space,
                ):
                    yield text[i : i + self.config.sliding.window_len] + "\n"

        with open(in_file) as file:
            contents = file.read()

        with open(out_file, "w") as file:
            file.writelines(generator(contents))

        print(f"\033[2m out: {out_file}\033[0m")

    def generate_trunc(
        self,
        in_file: str,
        out_file: str | None = None,
    ):
        if out_file is None:
            parts = in_file.split(".")
            out_file = ".".join(parts[:-1]) + ".corp" + "." + parts[-1]

        print("generating corpus (trunc)...")
        print(f"\033[2m  in: {in_file}\033[0m")
        with (
            open(in_file, "r") as _in_file,
            open(out_file, "w") as _out_file,
        ):
            for line in _in_file:
                line = line.strip()
                words = line.split()
                trunc_i = int(len(words) * self.config.trunc.ratio)
                words_x = " ".join(words[:trunc_i])
                words_y = " ".join(words[trunc_i:])
                _out_file.write(f"{words_x} \n{words_y}\n")

        print(f"\033[2m out: {out_file}\033[0m")

    def verify(self, in_file: str, debug=False):
        print("verifying corpus...")

        with open(in_file) as file:
            lines = file.readlines()
            xs = lines[::2]
            ys = lines[1::2]
            uxs = set(xs)
            uys = set(ys)
            valid_xs = len(xs) == len(uxs)
            valid_ys = len(ys) == len(uys)
            valid_xs_str = "true" if valid_xs else "\033[0;91mfalse\033[0m"
            valid_ys_str = "true" if valid_ys else "\033[0;91mfalse\033[0m"

            print(f"\033[2m valid xs: {valid_xs_str}\033[0m")
            print(f"\033[2m valid ys: {valid_ys_str}\033[0m")

            if debug and not valid_xs:
                dup_xs = [x for x, c in Counter(xs).items() if c > 1]
                print(f"   dup xs: {dup_xs}")
            if debug and not valid_ys:
                dup_ys = [y for y, c in Counter(ys).items() if c > 1]
                print(f"   dup ys: {dup_ys}")

            return valid_xs, valid_ys


if __name__ == "__main__":
    in_file = "data/plato/platos-republic.txt"
    norm_file = "data/plato/platos-republic.norm.txt"
    slid_file = "data/plato/platos-republic.slid.txt"
    out_file = "data/plato/platos-republic.corp.txt"
    print("loading config...")
    config = LLNCACorpusConfig(
        LLNCACorpusSlidingConfig(
            window_len=15,
            window_space=5,
            word_level=True,
        ),
        LLNCACorpusTruncConfig(
            ratio=0.7,
        ),
    )
    corpus = LLNCACorpus(config=config)
    corpus.normalize(in_file=in_file, out_file=norm_file)
    corpus.generate_sliding(in_file=norm_file, out_file=slid_file)
    corpus.generate_trunc(in_file=slid_file, out_file=out_file)
    corpus.verify(in_file=out_file)
