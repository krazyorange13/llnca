from dataclasses import dataclass


@dataclass
class LLNCACorpusConfig:
    trunc_ratio: float = 0.7
    trunc_split: str = " "


class LLNCACorpus:
    def __init__(self, config: LLNCACorpusConfig):
        self.config = config

    def generate(
        self,
        in_file: str,
        out_file: str | None = None,
        # test_file: str | None = None,
    ):
        if out_file is None:
            parts = in_file.split(".")
            out_file = ".".join(parts[:-1]) + ".corp" + "." + parts[-1]
        # if test_file is None:
        #     parts = in_file.split(".")
        #     test_file = ".".join(parts[:-1]) + ".corptest" + "." + parts[-1]
        print(f"\033[2m  in: {in_file}\033[0m")
        with (
            open(in_file, "r") as _in_file,
            open(out_file, "w") as _out_file,
            # open(test_file, "w") as _test_file,
        ):
            for line in _in_file:
                line = line.strip()
                words = line.split(self.config.trunc_split)
                trunc_i = int(len(words) * self.config.trunc_ratio)
                words_x = self.config.trunc_split.join(words[:trunc_i])
                words_y = self.config.trunc_split.join(words[trunc_i:])
                _out_file.write(f"{words_x}\n{words_y}\n")
                # _test_file.write(f"{words_y}\n")

        print(f"\033[2m out: {out_file}\033[0m")
        # print(f"\033[2mtest: {test_file}\033[0m")


if __name__ == "__main__":
    print("loading config...")
    config = LLNCACorpusConfig()
    corpus = LLNCACorpus(config=config)
    print("generating corpus...")
    corpus.generate(in_file="data/ezpz/ezpz.txt")
