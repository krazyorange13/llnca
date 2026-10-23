import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bpeasy.tokenizer import BPEasyTokenizer

# version of bpeasy.tokenizer._DEFAULT_REGEX_PATTERN
# that creates hard breaks wherever there's newlines

_NO_NEWLINE_REGEX_PATTERN = (
    r"""[\r\n]+|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
_NO_NEWLINE_YES_PUNCTUATION_REGEX_PATTERN = r"""[\r\n]+|[^\r\n\p{L}\p{N}]?\p{L}+[.!?;:]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@dataclass
class LLNCATokenizer2Config:
    name: str = ""
    regex_pattern: str = _NO_NEWLINE_REGEX_PATTERN
    special_tokens: list[str] = field(default_factory=list)
    vocab_size: int = 32000
    max_token_length: int = 128
    fill_to_nearest_multiple_of_eight: bool = False
    batch_size: int = 1000


class LLNCATokenizer2Exception(Exception):
    pass


type LLNCAVocab2 = dict[str, str | list[str] | dict[str, int]]


class LLNCATokenizer2:
    _NO_FILEPATH_EXISTS_MSG = (
        "file at path {} does not exist. perhaps you need to call `save`?"
    )
    _NO_TOKENIZER_MSG = (
        "no tokenizer is loaded. load the tokenizer using `train` or `load`."
    )

    def __init__(self, config: LLNCATokenizer2Config, vocab: LLNCAVocab2 | None = None):
        self.config = config
        self.tokenizer = None
        if vocab is not None:
            self.load_data(vocab)

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

    def encode(
        self,
        text: str,
        allowed_special: set[str] | None = None,
        disallowed_special: set[str] | None = None,
    ) -> str:
        if allowed_special is None:
            allowed_special = set()
        if disallowed_special is None:
            disallowed_special = set()

        if self.tokenizer is None:
            raise LLNCATokenizer2Exception(LLNCATokenizer2._NO_TOKENIZER_MSG)

        return self._encode_toks(
            self.tokenizer.encode(
                text,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
            )
        )

    def encode_file(
        self,
        in_path: str,
        out_path: str,
        allowed_special: set[str] | None = None,
        disallowed_special: set[str] | None = None,
    ):
        if allowed_special is None:
            allowed_special = set()
        if disallowed_special is None:
            disallowed_special = set()

        if self.tokenizer is None:
            raise LLNCATokenizer2Exception(LLNCATokenizer2._NO_TOKENIZER_MSG)

        print("encoding file...")
        print(f"\033[2m  in: {in_path}\033[0m")

        with (
            open(in_path, "r", encoding="utf-8", newline="") as in_file,
            open(out_path, "w", encoding="utf-8", newline="") as out_file,
        ):
            out_file.write(
                self._encode_toks(
                    self.tokenizer.encode(
                        in_file.read(),
                        allowed_special=allowed_special,
                        disallowed_special=disallowed_special,
                    )
                )
            )

        print(f"\033[2m out: {out_path}\033[0m")

    def decode(self, toks: list[int] | str) -> str:
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

        with (
            open(in_path, "r", encoding="utf-8", newline="") as in_file,
            open(out_path, "w", encoding="utf-8", newline="") as out_file,
        ):
            out_file.write(self.tokenizer.decode(self._decode_toks(in_file.read())))

        print(f"\033[2m out: {out_path}\033[0m")

    def _encode_toks(self, toks: list[int]) -> str:
        return "".join(map(chr, toks))

    def _decode_toks(self, toks: str) -> list[int]:
        return list(map(ord, toks))

    def get_special_tok(self, tok: str) -> int:
        return self._decode_toks(self.encode(tok, allowed_special={tok}))[0]

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

    def load_data(self, data: Any):
        bytes_vocab = {
            base64.b64decode(key): value for key, value in data["vocab"].items()
        }
        instance = BPEasyTokenizer(
            name=data["name"],
            vocab=bytes_vocab,
            regex_pattern=data["regex_pattern"],
            special_tokens=data["special_tokens"],
        )
        self.tokenizer = instance


if __name__ == "__main__":
    # corp_path = "data/plato/platos-republic.corp.txt"
    # corptok_path = "data/plato/platos-republic.corptok.txt"
    # tokenizer_path = "data/plato/platos-republic.tokenizer.json"
    corp_path = "data/harv/harv.corp.txt"
    corptok_path = "data/harv/harv.corptok.txt"
    tokenizer_path = "data/harv/harv.tokenizer.json"

    config = LLNCATokenizer2Config(
        special_tokens=["<pad>"],
        max_token_length=4,
    )
    tokenizer = LLNCATokenizer2(config=config)
    tokenizer.train(corp_path)
    tokenizer.encode_file(corp_path, corptok_path)
    tokenizer.save(tokenizer_path)

    # print(
    #     enc := tokenizer.encode(
    #         "mathematician<pad><pad><pad>", allowed_special={"<pad>"}
    #     )
    # )
    # print(tokenizer.decode(enc))
    print(padtok := tokenizer.get_special_tok("<pad>"))
    # print(tokenizer.decode([padtok]))
