import re
import numpy as np
from collections import Counter
from typing import Any

class Tokenizer:
    def __init__(self, max_vocab_size=3000):
        self.max_vocab_size = max_vocab_size

        # 词表
        self.word2idx = {}
        self.idx2word = {}

        # 特殊符号
        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"

    # =========================
    # 分词函数
    # =========================
    def tokenize_zh(self, text: object) -> list[Any]:
        # 中文：按字切
        return list(text.strip())

    def tokenize_en(self, text):
        # 英文：按空格 + 简单清洗
        text = text.lower()
        text = re.sub(r"[^a-zA-Z']+", " ", text)
        return text.strip().split()

    # =========================
    # 构建词表
    # =========================
    def build_vocab(self, sentences: object, is_zh: object = True) -> None:
        counter = Counter()

        for sent in sentences:
            tokens = self.tokenize_zh(sent) if is_zh else self.tokenize_en(sent)
            counter.update(tokens)

        # 保留最常见的词
        most_common = counter.most_common(self.max_vocab_size - 4)

        # 计算词汇覆盖率
        total_tokens = sum(counter.values())
        kept_tokens = sum(freq for word, freq in most_common)
        coverage = kept_tokens / total_tokens
        print( ("汉语" if is_zh else "英文") + "覆盖率" )
        print(coverage)

        # 加入特殊token
        vocab = [self.PAD, self.UNK, self.BOS, self.EOS]
        vocab += [word for word, _ in most_common]

        # 构建映射
        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}

    # =========================
    # 编码（句子 → id）
    # =========================
    def encode(self, sentence, is_zh=True, add_bos=False, add_eos=False):
        tokens = self.tokenize_zh(sentence) if is_zh else self.tokenize_en(sentence)

        if add_bos:
            tokens = [self.BOS] + tokens
        if add_eos:
            tokens = tokens + [self.EOS]

        ids = []
        for token in tokens:
            ids.append(self.word2idx.get(token, self.word2idx[self.UNK]))

        return ids

    def encode_batch(self, sentences, is_zh=True, add_bos=False, add_eos=False, max_len = None):
        batch_ids = []

        for sent in sentences:
            ids = self.encode(sent, is_zh, add_bos, add_eos)

            if max_len is not None:
                ids = self.pad(ids, max_len)

            batch_ids.append(ids)

        return np.array(batch_ids)

    # =========================
    # 解码（id → 句子）
    # =========================
    def decode(self, ids, is_zh = True):
        tokens = []
        for i in ids:
            word = self.idx2word.get(i, self.UNK)
            if word in [self.PAD, self.BOS, self.EOS]:
                continue
            tokens.append(word)

        if is_zh:
            return "".join(tokens)
        else:
            return " ".join(tokens)

    def decode_batch(self, batch_ids, is_zh=True):
        sentences = []

        for ids in batch_ids:
            sent = self.decode(ids, is_zh=is_zh)
            sentences.append(sent)

        return sentences

    # =========================
    # padding
    # =========================
    def pad(self, ids, max_len):
        if len(ids) < max_len:
            ids += [self.word2idx[self.PAD]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    def create_padding_mask(self, batch_ids):
        """
        batch_ids: (B, seq_len)
        return: (B, seq_len)  1表示有效token，0表示PAD
        """
        return (batch_ids != self.word2idx[self.PAD]).astype(np.float32)

# =========================
# 主函数（测试用）
# =========================
def main():
    # 读取数据
    train_file = "train.txt"

    zh_sentences = []
    en_sentences = []

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            zh, en = line.strip().split("\t")
            zh_sentences.append(zh)
            en_sentences.append(en)

    # max_len = max(len(s.split()) for s in en_sentences)
    # print(max_len)
    #
    # max_len = max(len(s) for s in zh_sentences)
    # print(max_len)

    # 初始化 tokenizer
    zh_tokenizer = Tokenizer(max_vocab_size=3000)
    en_tokenizer = Tokenizer(max_vocab_size=4000)

    # 构建词表
    zh_tokenizer.build_vocab(zh_sentences, is_zh=True)
    en_tokenizer.build_vocab(en_sentences, is_zh=False)

    print("中文词表大小:", len(zh_tokenizer.word2idx))
    print("英文词表大小:", len(en_tokenizer.word2idx))

    # 测试句子
    test_zh = "我爱你"
    test_en = "i love you"

    print("\n=== 编码测试 ===")

    zh_ids = zh_tokenizer.encode(test_zh, is_zh=True)
    en_ids = en_tokenizer.encode(test_en, is_zh=False, add_bos=True, add_eos=True)

    print("中文:", test_zh)
    print("编码:", zh_ids)

    print("英文:", test_en)
    print("编码:", en_ids)

    # padding
    max_len = 10
    zh_pad = zh_tokenizer.pad(zh_ids, max_len)
    en_pad = en_tokenizer.pad(en_ids, max_len)

    print("\n=== Padding ===")
    print("中文pad:", zh_pad)
    print("英文pad:", en_pad)

    # 解码
    print("\n=== 解码测试 ===")
    print("中文 decode:", zh_tokenizer.decode(zh_pad, True))
    print("英文 decode:", en_tokenizer.decode(en_pad))

    print("\n ****** batch test *******")
    # ===== 构造简单数据 =====
    zh_sentences_test = [
        "我爱你",
        "你好世界",
        "机器学习很有趣"
    ]

    en_sentences_test = [
        "i love you",
        "hello world",
        "machine learning is fun"
    ]

    # ===== 初始化 tokenizer =====
    zh_tokenizer = Tokenizer(max_vocab_size=3000)
    en_tokenizer = Tokenizer(max_vocab_size=3000)

    # ===== 构建词表 =====
    zh_tokenizer.build_vocab(zh_sentences, is_zh=True)
    en_tokenizer.build_vocab(en_sentences, is_zh=False)

    print("\n词表大小：")
    print("ZH:", len(zh_tokenizer.word2idx))
    print("EN:", len(en_tokenizer.word2idx))

    # ===== batch encode =====
    max_len = 10

    zh_batch = zh_tokenizer.encode_batch(
        zh_sentences_test,
        is_zh=True,
        add_bos=True,
        add_eos=True,
        max_len=max_len
    )

    en_batch = en_tokenizer.encode_batch(
        en_sentences_test,
        is_zh=False,
        add_bos=True,
        add_eos=True,
        max_len=max_len
    )

    print("\n=== Batch Encode ===")
    print("ZH batch shape:", zh_batch.shape)
    print(zh_batch)

    print("\nEN batch shape:", en_batch.shape)
    print(en_batch)

    # ===== batch decode =====
    zh_decoded = zh_tokenizer.decode_batch(zh_batch, is_zh=True)
    en_decoded = en_tokenizer.decode_batch(en_batch, is_zh=False)

    print("\n=== Batch Decode ===")
    print("ZH decoded:")
    for s in zh_decoded:
        print(s)

    print("\nEN decoded:")
    for s in en_decoded:
        print(s)


if __name__ == "__main__":
    main()