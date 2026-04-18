import os
import json
import numpy as np

from tokenizer import Tokenizer
from toyTransformer import Transformer


def save_tokenizer(tokenizer, filepath):
    data = {
        "max_vocab_size": tokenizer.max_vocab_size,
        "word2idx": tokenizer.word2idx,
        "idx2word": {str(k): v for k, v in tokenizer.idx2word.items()},
        "PAD": tokenizer.PAD,
        "UNK": tokenizer.UNK,
        "BOS": tokenizer.BOS,
        "EOS": tokenizer.EOS,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_tokenizer(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = Tokenizer(max_vocab_size=data["max_vocab_size"])
    tokenizer.word2idx = data["word2idx"]
    tokenizer.idx2word = {int(k): v for k, v in data["idx2word"].items()}
    tokenizer.PAD = data["PAD"]
    tokenizer.UNK = data["UNK"]
    tokenizer.BOS = data["BOS"]
    tokenizer.EOS = data["EOS"]

    return tokenizer


def save_checkpoint(model, zh_tokenizer, en_tokenizer, config, save_dir):
    """
    保存：
    1. 模型权重
    2. tokenizer
    3. 配置
    """
    os.makedirs(save_dir, exist_ok=True)

    # ===== 1. 保存模型参数 =====
    weights_path = os.path.join(save_dir, "model_weights.npz")
    params = model.get_parameters_dict()
    np.savez_compressed(weights_path, **params)

    # ===== 2. 保存 tokenizer =====
    save_tokenizer(zh_tokenizer, os.path.join(save_dir, "zh_tokenizer.json"))
    save_tokenizer(en_tokenizer, os.path.join(save_dir, "en_tokenizer.json"))

    # ===== 3. 保存 config =====
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"checkpoint saved to: {save_dir}")


def load_checkpoint(save_dir):
    """
    加载：
    1. config
    2. tokenizer
    3. model
    4. 权重
    """
    config_path = os.path.join(save_dir, "config.json")
    zh_tok_path = os.path.join(save_dir, "zh_tokenizer.json")
    en_tok_path = os.path.join(save_dir, "en_tokenizer.json")
    weights_path = os.path.join(save_dir, "model_weights.npz")

    # ===== 1. load config =====
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # ===== 2. load tokenizer =====
    zh_tokenizer = load_tokenizer(zh_tok_path)
    en_tokenizer = load_tokenizer(en_tok_path)

    # ===== 3. rebuild model =====
    model = Transformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        heads=config["heads"],
        d_ff=config["d_ff"],
        max_len=config["max_len"]
    )

    # ===== 4. load weights =====
    npz_file = np.load(weights_path, allow_pickle=True)
    params = {k: npz_file[k] for k in npz_file.files}
    model.load_parameters_dict(params)

    print(f"checkpoint loaded from: {save_dir}")

    return model, zh_tokenizer, en_tokenizer, config