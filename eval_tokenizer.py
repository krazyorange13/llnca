import base64
import json
import sys

import dearpygui.dearpygui as dpg
from bpeasy.tokenizer import BPEasyTokenizer

from tokenizer2 import LLNCATokenizer2, LLNCATokenizer2Config

if __name__ == "__main__":
    # dpg.create_context()

    # if len(sys.argv) < 2:
    #     print("pls pass the vocab path: *.tokenizer.json")
    #     sys.exit(1)
    # else:
    #     vocab_path = sys.argv[1]

    vocab_path = "data/ezpz2/ezpz2.tokenizer.json"
    # vocab_path = "data/plato/platos-republic.tokenizer.json"

    config = LLNCATokenizer2Config(
        special_tokens=["<pad>"],
        max_token_length=128,
    )

    with open(vocab_path) as file:
        vocab = json.load(file)

    tokenizer = LLNCATokenizer2(config=config, vocab=vocab)
    # print(
    #     {
    #         int.from_bytes(base64.decodebytes(k.encode())): v
    #         for k, v in vocab["vocab"].items()
    #     }
    # )
    # print(tokenizer.encode(list(vocab["vocab"].values())))

    assert tokenizer.tokenizer is not None

    print("\nRAW VOCAB\n")
    print(tokenizer.tokenizer.vocab)

    corpus_path = "data/ezpz2/ezpz2.corp.txt"
    # corpus_path = "data/plato/platos-republic.corp.txt"
    with open(corpus_path) as f:
        corpus_text = f.read()
    corpus_words = corpus_text.split()

    ks = {}
    for k in sorted(tokenizer.tokenizer.vocab):
        ins = [k_ for k_ in tokenizer.tokenizer.vocab if k in k_]
        ins = [k_.decode(errors="backslashreplace") for k_ in ins]
        k = k.decode(errors="backslashreplace")
        ks[k] = ins

    print("\nTOKENS and their RELATIONSHIPS\n")

    for k, ins in ks.items():
        print(f"{k!r} in: {' '.join([f'{k_!r}' for k_ in ins])}")

    print("\nFINAL VALID TOKENS\n")

    ks_final: list[str] = []
    ks_final_saved: list[str] = []
    ks_final_all: list[str] = []
    ks_final_easy: list[str] = []

    for k, ins in ks.items():
        if k.strip() in corpus_words:
            ks_final_saved.append(k)
        if len(ins) > 1:
            continue
        if "\\x" in repr(k):
            continue
        ks_final.append(k)
        if k in ks_final_saved:
            ks_final_saved.remove(k)
        ks_final_all.append(k)

    for k, ins in ks.items():
        if k.strip() in corpus_words:
            ks_final_easy.append(k)

    for k in ks_final:
        print(f"{k!r}")

    print("\nFINAL VALID TOKENS (no single char)\n")

    ks_final_words = [k for k in ks_final if len(k) > 1]

    for k in ks_final_words:
        print(f"{k!r}")

    print()
    print(f"final tokens: {len(ks_final)}")
    print(f"final tokens (no single char): {len([k for k in ks_final if len(k) > 1])}")

    print("\nSAVED TOKENS\n")

    for k in ks_final_saved:
        print(f"{k!r}")

    print("\nKS ALL/EASY DIFFERENCE\n")
    kfa = set(ks_final_all)
    kfe = set(ks_final_easy)
    kfd = (kfa | kfe) - (kfa & kfe)
    print("all differences:", kfd)
    print("ks unique to kfa:", kfa - kfe)
    print("ks unique to kfe:", kfe - kfa)
    print(ks_final_all == ks_final_easy)

    # with dpg.window(label="main"):  # type: ignore
    #     pass

    # dpg.setup_dearpygui()
    # dpg.create_viewport()
    # dpg.show_viewport()
    # dpg.start_dearpygui()
    # dpg.destroy_context()
