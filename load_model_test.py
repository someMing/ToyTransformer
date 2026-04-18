import numpy as np
from checkpoint_utils import load_checkpoint

def main():
    model, zh_tokenizer, en_tokenizer, config = load_checkpoint("checkpoints/final_model")

    src_pad_id = zh_tokenizer.word2idx[zh_tokenizer.PAD]
    tgt_pad_id = en_tokenizer.word2idx[en_tokenizer.PAD]
    bos_id = en_tokenizer.word2idx[en_tokenizer.BOS]
    eos_id = en_tokenizer.word2idx[en_tokenizer.EOS]

    text = "你会骑马吗"

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

    print("ZH       :", text)
    print("PRED_IDS :", pred_ids)
    print("PRED_TEXT:", pred_text)

if __name__ == "__main__":
    main()