import argparse
import os
import time

import numpy as np
import tiktoken
from tqdm import tqdm


def tokenize_txt(filedir: str):
    """Tokenize txt file with tiktoken gpt2 tokenizer"""
    files = os.listdir(filedir)
    print(f"found {len(files)} files in {filedir}")
    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = []
    for file in tqdm(files):
        with open(f"{filedir}/{file}", "r") as f:
            data = f.read()
        input_ids.extend(tokenizer.encode(data, allowed_special={"<|endoftext|>"}))
    return np.array(input_ids).astype(np.uint16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize txt file with tiktoken",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--txt",
        type=str,
        default="data/text",
        help="Folder with txt files",
    )
    parser.add_argument(
        "--out", type=str, default="data/train", help="File to write tokens to"
    )
    args = parser.parse_args()

    t = time.time()
    print(f"Starting to tokenize {args.txt} using gpt2 tokenizer..")
    input_ids = tokenize_txt(args.txt)
    np.save(args.out, input_ids)
    print(f"Number of trainings token: {len(input_ids):,}")
    print(f"took {time.time()-t:.2f}s")
