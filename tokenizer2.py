from dataclasses import dataclass, field
from pathlib import Path

from bpeasy.tokenizer import _DEFAULT_REGEX_PATTERN, BPEasyTokenizer


@dataclass
class LLNCATokenizer2Config:
    name: str = ""
    regex_pattern: str = _DEFAULT_REGEX_PATTERN
    special_tokens: list[str] = field(default_factory=list)
    vocab_size: int = 32000
    max_token_length: int = 128
    fill_to_nearest_multiple_of_eight: bool = False
    batch_size: int = 1000


class LLNCATokenizer2Exception(Exception):
    pass


class LLNCATokenizer2:
    _NO_FILEPATH_EXISTS_MSG = (
        "file at path {} does not exist. perhaps you need to call `save`?"
    )
    _NO_TOKENIZER_MSG = (
        "no tokenizer is loaded. load the tokenizer using `train` or `load`."
    )

    def __init__(self, config: LLNCATokenizer2Config):
        self.config = config
        self.tokenizer = None

    def train(self, path: str):
        print("training...")
        print(f"\033[2m  in: {path}\033[0m")
        with open(path) as file:
            self.tokenizer = BPEasyTokenizer.train(
                file,
                self.config.vocab_size,
                self.config.max_token_length,
                self.config.regex_pattern,
                self.config.special_tokens,
                self.config.fill_to_nearest_multiple_of_eight,
                self.config.name,
                self.config.batch_size,
            )

    def encode(self, text: str) -> str:
        if self.tokenizer is None:
            raise LLNCATokenizer2Exception(LLNCATokenizer2._NO_TOKENIZER_MSG)

        return self._encode_toks(self.tokenizer.encode(text))

    def encode_file(self, in_path: str, out_path: str):
        if self.tokenizer is None:
            raise LLNCATokenizer2Exception(LLNCATokenizer2._NO_TOKENIZER_MSG)

        print("encoding file...")
        print(f"\033[2m  in: {in_path}\033[0m")

        with open(in_path, "r") as in_file, open(out_path, "w") as out_file:
            out_file.write(self._encode_toks(self.tokenizer.encode(in_file.read())))

        print(f"\033[2m out: {out_path}\033[0m")

    def decode(self, toks: list[int] | str) -> str | None:
        if self.tokenizer is None:
            raise LLNCATokenizer2Exception(LLNCATokenizer2._NO_TOKENIZER_MSG)

        if isinstance(toks, str):
            toks = self._decode_toks(toks)

        return self.tokenizer.decode(toks)

    def decode_file(self, in_path: str, out_path: str):
        if self.tokenizer is None:
            raise LLNCATokenizer2Exception(LLNCATokenizer2._NO_TOKENIZER_MSG)

        print("decoding file...")
        print(f"\033[2m  in: {in_path}\033[0m")

        with open(in_path, "r") as in_file, open(out_path, "w") as out_file:
            out_file.write(self.tokenizer.decode(self._decode_toks(in_file.read())))

        print(f"\033[2m out: {out_path}\033[0m")

    def _encode_toks(self, toks: list[int]) -> str:
        return "".join(map(chr, toks))

    def _decode_toks(self, toks: str) -> list[int]:
        return list(map(ord, toks))

    def save(self, path: str):
        if self.tokenizer is None:
            raise LLNCATokenizer2Exception(LLNCATokenizer2._NO_TOKENIZER_MSG)

        print("saving...")
        print(f"\033[2m out: {path}\033[0m")

        self.tokenizer.save(path)

    def load(self, path: str):
        if not Path(path).exists():
            raise LLNCATokenizer2Exception(
                LLNCATokenizer2._NO_FILEPATH_EXISTS_MSG.format(path)
            )

        print("loading...")
        print(f"\033[2m  in: {path}\033[0m")

        self.tokenizer = BPEasyTokenizer.from_file(path)


if __name__ == "__main__":
    corp_path = "data/plato/platos-republic.corp.txt"
    corptok_path = "data/plato/platos-republic.corptok.txt"
    tokenizer_path = "data/plato/platos-republic.tokenizer.json"

    config = LLNCATokenizer2Config()
    tokenizer = LLNCATokenizer2(config=config)
    tokenizer.train(corp_path)
    tokenizer.encode_file(corp_path, corptok_path)
    tokenizer.save(tokenizer_path)
