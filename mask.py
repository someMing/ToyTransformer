import numpy as np

def create_padding_mask(batch_ids, pad_id):
    """
    batch_ids: (B, N)
    return: (B, 1, 1, N)
    """
    mask = (batch_ids != pad_id).astype(np.float32)   # 1=有效，0=pad
    mask = (1.0 - mask) * (-1e9)                      # pad位置 -> -1e9
    mask = mask[:, None, None, :]                     # (B,1,1,N)
    return mask


def create_causal_mask(seq_len):
    """
    return: (1, 1, seq_len, seq_len)
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = mask * (-1e9) # 或 -np.inf
    mask = mask[None, None, :, :]
    return mask


def create_decoder_mask(batch_ids, pad_id):
    """
    padding + causal
    return: (B, 1, N, N)
    """
    B, N = batch_ids.shape

    pad_mask = create_padding_mask(batch_ids, pad_id)   # (B,1,1,N)
    causal_mask = create_causal_mask(N)                # (1,1,N,N)

    mask = pad_mask + causal_mask                      # broadcast
    return mask

if __name__ == "__main__":
    B = 2
    N = 4

    token_ids = np.array([
        [1, 5, 10, 0],
        [2, 6, 0, 0]
    ])

    print("padding mask 测试:\n", create_padding_mask(token_ids, 0))
    print("causal mask 测试:\n", create_causal_mask(4))
    print("create_decoder_maske测试:\n", create_decoder_mask(token_ids, 0))