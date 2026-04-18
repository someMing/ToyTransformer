import numpy as np

class InputEmbedding:
    def __init__(self, vocab_size, d_model, max_len=25):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.dW = None

        # ===== 1. Embedding 参数 =====
        self.W = np.random.randn(vocab_size, d_model) * 0.01

        # ===== 2. 预计算 Positional Encoding =====
        self.PE = self.build_positional_encoding()

    def build_positional_encoding(self):
        """
            生成 shape = (max_len, d_model)
        """
        PE = np.zeros((self.max_len, self.d_model))

        for pos in range(self.max_len):
            for i in range(0, self.d_model, 2):
                angle = pos / (10000 ** (i / self.d_model))

                PE[pos, i] = np.sin(angle)

                if i + 1 < self.d_model:
                    PE[pos, i + 1] = np.cos(angle)

        return PE

    def forward(self, token_ids):
        """
        token_ids: (B, seq_len)
        return: (B, seq_len, d_model)
        """

        self.token_ids = token_ids  # cache
        B, seq_len = token_ids.shape
        self.seq_len = seq_len

        # ===== 1. Embedding lookup =====
        x = self.W[token_ids]  # (B, seq_len, d_model)

        # ===== 2. scaling =====
        x = x * np.sqrt(self.d_model)

        # ===== 3. 加 positional encoding =====
        x = x + self.PE[:seq_len]

        return x


    def backward(self, token_ids, dout):
        """
        token_ids: (B, seq_len)
        dout: (B, seq_len, d_model)

        return:
            dW: (vocab_size, d_model)
        """

        # =========================
        # 1. backward through scaling
        # forward: x = lookup * sqrt(d_model)
        #
        # 公式:
        # y = c * x
        # dx = dy * c
        # =========================
        dlookup = dout * np.sqrt(self.d_model)

        # =========================
        # 2. backward through embedding lookup
        # x = W[token_ids]
        # 梯度按 token_id 累加回 W
        # =========================
        dW = np.zeros_like(self.W)

        B, seq_len = token_ids.shape
        for b in range(B):
            for t in range(seq_len):
                token_id = token_ids[b, t]
                dW[token_id] += dlookup[b, t]

        return dW

if __name__ == "__main__":
    vocab_size = 10
    d_model = 4
    max_len = 6

    emb = InputEmbedding(vocab_size, d_model, max_len)

    token_ids = np.array([
        [1, 2, 3],
        [2, 4, 1]
    ])

    out = emb.forward(token_ids)
    print("out shape:", out.shape)

    dout = np.random.randn(*out.shape)
    dW = emb.backward(dout)

    print("dW shape:", dW.shape)
    print("token_ids:")
    print(token_ids)

    print("\n非零梯度行（对应出现过的 token）:")
    used_tokens = np.unique(token_ids)
    for tid in used_tokens:
        print(f"token {tid}, grad norm = {np.linalg.norm(dW[tid]):.6f}")