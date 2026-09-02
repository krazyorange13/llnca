from dataclasses import dataclass


@dataclass
class LLNCACorpusConfig:
    trunc_ratio: float = 0.8
    trunc_split: str = " "


class LLNCACorpus:
    def __init__(self, config: LLNCACorpusConfig):
        self.config = config

    def generate(
        self,
        in_file: str,
        out_file: str | None = None,
    ):
        if out_file is None:
            parts = in_file.split(".")
            out_file = ".".join(parts[:-1]) + ".corp" + "." + parts[-1]

        print(f"\033[2m  in: {in_file}\033[0m")
        with (
            open(in_file, "r") as _in_file,
            open(out_file, "w") as _out_file,
        ):
            trunc_split = self.config.trunc_split
            for line in _in_file:
                line = line.strip()
                words = line.split(trunc_split)
                trunc_i = int(len(words) * self.config.trunc_ratio)
                words_x = trunc_split.join(words[:trunc_i])
                words_y = trunc_split.join(words[trunc_i:])
                _out_file.write(f"{words_x}{trunc_split}\n{words_y}\n")

        print(f"\033[2m out: {out_file}\033[0m")

    def verify(self, in_file: str):
        with open(in_file) as file:
            lines = file.readlines()
            xs = lines[::2]
            ys = lines[1::2]
            uxs = set(xs)
            uys = set(ys)
            valid_xs = len(xs) == len(uxs)
            valid_ys = len(ys) == len(uys)
            print(f"\033[2m valid xs: {valid_xs}\033[0m")
            print(f"\033[2m valid ys: {valid_ys}\033[0m")
            return valid_xs, valid_ys


if __name__ == "__main__":
    in_file = "data/harv/harv.txt"
    print("loading config...")
    config = LLNCACorpusConfig(
        trunc_ratio=0.7,
    )
    corpus = LLNCACorpus(config=config)
    print("generating corpus...")
    corpus.generate(in_file=in_file)
    corpus.verify(in_file=in_file)
