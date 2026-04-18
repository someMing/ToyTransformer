import os
import argparse
import numpy as np

from tokenizer import Tokenizer
from data_utils import load_parallel_data, get_batches, build_dataset
from toyTransformer import Transformer
from checkpoint_utils import save_checkpoint, load_checkpoint


# =====================================
# 你当前默认模型配置
# 想改模型规模，只改这里
# =====================================
MODEL_CONFIG = {
    "d_model": 128,
    "num_layers": 2,
    "heads": 4,
    "d_ff": 256,
    "warmup_steps": 1000,
    "batch_size": 32,
    "dropout_rate": 0.1,   # 只是记录；如果你的 FFN/模型内部已固定，也没关系
    "src_vocab_limit": 3000,
    "tgt_vocab_limit": 3000,
}


def print_model_info(config, train_size):
    print("===== Model Info =====")
    print(f"train size   : {train_size}")
    print(f"layers       : {config['num_layers']}")
    print(f"d_model      : {config['d_model']}")
    print(f"heads        : {config['heads']}")
    print(f"d_ff         : {config['d_ff']}")
    print(f"warmup_steps : {config['warmup_steps']}")
    print("=" * 30)


def get_transformer_lr(step_num, d_model, warmup_steps=1000):
    """
    Transformer论文中的学习率策略：
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    step_num = max(step_num, 1)
    return (d_model ** -0.5) * min(
        step_num ** -0.5,
        step_num * (warmup_steps ** -1.5)
    )


def evaluate_loss(model, src_ids, tgt_ids, targets, src_pad_id, tgt_pad_id, batch_size=32):
    losses = []

    for batch_src, batch_tgt, batch_targets in get_batches(
        src_ids, tgt_ids, targets, batch_size=batch_size, shuffle=False
    ):
        logits = model.forward(batch_src, batch_tgt, src_pad_id, tgt_pad_id)
        loss = model.cross_entropy_loss(logits, batch_targets, tgt_pad_id)
        losses.append(loss)

    return float(np.mean(losses))


def build_everything(train_file, valid_file):
    # ===== 1. 读取数据 =====
    train_zh, train_en = load_parallel_data(train_file)
    valid_zh, valid_en = load_parallel_data(valid_file)

    # ===== 2. tokenizer =====
    zh_tokenizer = Tokenizer(max_vocab_size=MODEL_CONFIG["src_vocab_limit"])
    en_tokenizer = Tokenizer(max_vocab_size=MODEL_CONFIG["tgt_vocab_limit"])

    zh_tokenizer.build_vocab(train_zh, is_zh=True)
    en_tokenizer.build_vocab(train_en, is_zh=False)

    # ===== 3. 最大长度 =====
    src_max_len = max(len(s) for s in train_zh) + 1
    tgt_max_len = max(len(s.split()) for s in train_en) + 1

    # ===== 4. 构造数据 =====
    train_src_ids, train_tgt_ids, train_targets = build_dataset(
        train_zh, train_en,
        zh_tokenizer, en_tokenizer,
        src_max_len, tgt_max_len
    )

    valid_src_ids, valid_tgt_ids, valid_targets = build_dataset(
        valid_zh, valid_en,
        zh_tokenizer, en_tokenizer,
        src_max_len, tgt_max_len
    )

    # ===== 5. token ids =====
    src_pad_id = zh_tokenizer.word2idx[zh_tokenizer.PAD]
    tgt_pad_id = en_tokenizer.word2idx[en_tokenizer.PAD]
    bos_id = en_tokenizer.word2idx[en_tokenizer.BOS]
    eos_id = en_tokenizer.word2idx[en_tokenizer.EOS]

    # ===== 6. config =====
    config = {
        "src_vocab_size": len(zh_tokenizer.word2idx),
        "tgt_vocab_size": len(en_tokenizer.word2idx),
        "d_model": MODEL_CONFIG["d_model"],
        "num_layers": MODEL_CONFIG["num_layers"],
        "heads": MODEL_CONFIG["heads"],
        "d_ff": MODEL_CONFIG["d_ff"],
        "max_len": max(src_max_len, tgt_max_len),
        "src_max_len": src_max_len,
        "tgt_max_len": tgt_max_len,
        "warmup_steps": MODEL_CONFIG["warmup_steps"],
        "batch_size": MODEL_CONFIG["batch_size"],
        "dropout_rate": MODEL_CONFIG["dropout_rate"],
        "train_size": len(train_zh),
    }

    # ===== 7. model =====
    model = Transformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        heads=config["heads"],
        d_ff=config["d_ff"],
        max_len=config["max_len"]
    )

    return {
        "model": model,
        "zh_tokenizer": zh_tokenizer,
        "en_tokenizer": en_tokenizer,
        "config": config,
        "train_zh": train_zh,
        "train_en": train_en,
        "valid_zh": valid_zh,
        "valid_en": valid_en,
        "train_src_ids": train_src_ids,
        "train_tgt_ids": train_tgt_ids,
        "train_targets": train_targets,
        "valid_src_ids": valid_src_ids,
        "valid_tgt_ids": valid_tgt_ids,
        "valid_targets": valid_targets,
        "src_pad_id": src_pad_id,
        "tgt_pad_id": tgt_pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
    }


def run_train(args):
    pack = build_everything(args.train_file, args.valid_file)

    model = pack["model"]
    zh_tokenizer = pack["zh_tokenizer"]
    en_tokenizer = pack["en_tokenizer"]
    config = pack["config"]

    train_src_ids = pack["train_src_ids"]
    train_tgt_ids = pack["train_tgt_ids"]
    train_targets = pack["train_targets"]

    valid_src_ids = pack["valid_src_ids"]
    valid_tgt_ids = pack["valid_tgt_ids"]
    valid_targets = pack["valid_targets"]

    src_pad_id = pack["src_pad_id"]
    tgt_pad_id = pack["tgt_pad_id"]

    print_model_info(config, config["train_size"])

    save_dir = os.path.join("checkpoints", args.save_dir)
    best_dir = os.path.join(save_dir, "best_model")
    final_dir = os.path.join(save_dir, "final_model")

    best_valid_loss = float("inf")
    patience = 3
    bad_epochs = 0

    step_num = 0
    d_model = config["d_model"]
    warmup_steps = config["warmup_steps"]
    batch_size = config["batch_size"]

    print(f"epochs       : {args.epochs}")
    print(f"save folder  : {save_dir}")
    print()

    for epoch in range(args.epochs):
        train_losses = []

        for batch_src, batch_tgt, batch_targets in get_batches(
            train_src_ids, train_tgt_ids, train_targets,
            batch_size=batch_size,
            shuffle=True
        ):
            step_num += 1
            lr = get_transformer_lr(step_num, d_model, warmup_steps)

            loss = model.train_step(
                src_ids=batch_src,
                tgt_ids=batch_tgt,
                targets=batch_targets,
                src_pad_id=src_pad_id,
                tgt_pad_id=tgt_pad_id,
                lr=lr
            )
            train_losses.append(loss)

        train_loss = float(np.mean(train_losses))
        valid_loss = evaluate_loss(
            model,
            valid_src_ids, valid_tgt_ids, valid_targets,
            src_pad_id, tgt_pad_id,
            batch_size=batch_size
        )

        print(f"epoch {epoch + 1}, lr = {lr:.8f}, train_loss = {train_loss:.6f}, valid_loss = {valid_loss:.6f}")

        # ===== 保存 best model =====
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            bad_epochs = 0

            save_checkpoint(
                model=model,
                zh_tokenizer=zh_tokenizer,
                en_tokenizer=en_tokenizer,
                config=config,
                save_dir=best_dir
            )
            print(f"best model updated, best_valid_loss = {best_valid_loss:.6f}")
        else:
            bad_epochs += 1
            print(f"no improvement for {bad_epochs} epoch(s)")

        # ===== early stopping =====
        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # ===== 保存最终模型 =====
    save_checkpoint(
        model=model,
        zh_tokenizer=zh_tokenizer,
        en_tokenizer=en_tokenizer,
        config=config,
        save_dir=final_dir
    )
    print(f"training finished. final model saved to: {final_dir}")


def run_translate(args):
    model_dir = os.path.join("checkpoints", args.model_dir)
    best_dir = os.path.join(model_dir, "best_model")

    # 优先加载 best_model；如果没有，再尝试直接加载 model_dir
    load_dir = best_dir if os.path.exists(best_dir) else model_dir

    model, zh_tokenizer, en_tokenizer, config = load_checkpoint(load_dir)

    train_size = config.get("train_size", -1)
    print_model_info(config, train_size)

    src_pad_id = zh_tokenizer.word2idx[zh_tokenizer.PAD]
    tgt_pad_id = en_tokenizer.word2idx[en_tokenizer.PAD]
    bos_id = en_tokenizer.word2idx[en_tokenizer.BOS]
    eos_id = en_tokenizer.word2idx[en_tokenizer.EOS]

    print("进入翻译模式。输入中文句子后按回车翻译。")
    print("输入 exit / quit 退出。")
    print()

    while True:
        text = input("中文> ").strip()

        if text.lower() in ["exit", "quit"]:
            print("已退出翻译模式。")
            break

        if text == "":
            continue

        src = zh_tokenizer.encode_batch(
            [text],
            is_zh=True,
            add_bos=False,
            add_eos=True,
            max_len=config["src_max_len"]
        )

        pred_ids = model.predict(
            src_ids=src,
            bos_id=bos_id,
            eos_id=eos_id,
            src_pad_id=src_pad_id,
            tgt_pad_id=tgt_pad_id,
            max_len=config["tgt_max_len"]
        )

        pred_text = en_tokenizer.decode_batch(pred_ids, is_zh=False)[0]

        print("英文>", pred_text)
        print()


def main():
    parser = argparse.ArgumentParser(description="Toy Transformer 中文到英文")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ===== train =====
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--epochs", type=int, required=True, help="训练轮数")
    train_parser.add_argument("--save_dir", type=str, required=True, help="保存文件夹名称（位于 checkpoints/ 下）")
    train_parser.add_argument("--train_file", type=str, default="train.txt", help="训练集文件")
    train_parser.add_argument("--valid_file", type=str, default="valid.txt", help="验证集文件")

    # ===== translate =====
    translate_parser = subparsers.add_parser("translate", help="加载模型并进入翻译模式")
    translate_parser.add_argument("--model_dir", type=str, required=True, help="模型文件夹名称（位于 checkpoints/ 下）")

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "translate":
        run_translate(args)


if __name__ == "__main__":
    main()