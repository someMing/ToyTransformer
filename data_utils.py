import random
import numpy as np

def load_parallel_data(filepath):
    zh_sentences = []
    en_sentences = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue

            zh, en = parts
            zh_sentences.append(zh.strip())
            en_sentences.append(en.strip())

    return zh_sentences, en_sentences


def get_batches(src_ids, tgt_ids, targets, batch_size, shuffle=True):
    n = len(src_ids)
    indices = list(range(n))

    if shuffle:
        random.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]

        yield (
            src_ids[batch_idx],
            tgt_ids[batch_idx],
            targets[batch_idx]
        )

def build_dataset(zh_sentences, en_sentences, zh_tokenizer, en_tokenizer, src_max_len, tgt_max_len):
    src_ids = zh_tokenizer.encode_batch(
        zh_sentences,
        is_zh=True,
        add_bos=False,
        add_eos=True,
        max_len=src_max_len
    )

    tgt_ids = en_tokenizer.encode_batch(
        en_sentences,
        is_zh=False,
        add_bos=True,
        add_eos=False,
        max_len=tgt_max_len
    )

    targets = en_tokenizer.encode_batch(
        en_sentences,
        is_zh=False,
        add_bos=False,
        add_eos=True,
        max_len=tgt_max_len
    )

    return src_ids, tgt_ids, targets