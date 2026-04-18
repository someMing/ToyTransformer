import numpy as np

def relu(x):
    return np.maximum(0, x)

def dropout(x, p, training=True):
    """
    x: 任意 shape 的 numpy 数组
    p: dropout rate，例如 0.1
    training: 训练时 True，预测时 False
    """
    if (not training) or p <= 0.0:
        mask = np.ones_like(x, dtype=np.float32)
        return x, mask

    keep_prob = 1.0 - p
    mask = (np.random.rand(*x.shape) < keep_prob).astype(np.float32)
    out = x * mask / keep_prob
    return out, mask

class PositionwiseFFN:
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        self.d_model = d_model
        self.d_ff = d_ff

        # 参数初始化
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

        # backward缓存变量
        self.X = None
        self.hidden_linear = None
        self.hidden = None

        # 方便后面优化
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

        # dropout
        self.dropout_rate = dropout_rate

    def forward(self, X, training=True):
        """
        X: (B, N, d_model)
        return: (B, N, d_model)
        """
        self.X = X

        # ===== 1. 第一个线性层 =====
        self.hidden_linear = X @ self.W1 + self.b1  # (B, N, d_ff)

        # ===== 2. 激活 =====
        self.hidden = relu(self.hidden_linear)  # (B, N, d_ff)

        # ===== 3. Dropout =====
        self.hidden_drop, self.dropout_mask = dropout(
            self.hidden,
            p=self.dropout_rate,
            training=training
        )

        # ===== 4. 第二个线性层 =====
        out = self.hidden_drop @ self.W2 + self.b2

        return out

    def backward(self, dout):
        """
        dout: (B, N, d_model)
        return: dx
        """

        B, N, d_model = dout.shape

        # ===== flatten =====
        X = self.X.reshape(-1, d_model)  # (B*N, d_model)
        hidden_linear = self.hidden_linear.reshape(-1, self.d_ff)  # (B*N, d_ff)
        hidden_drop = self.hidden_drop.reshape(-1, self.d_ff)  # (B*N, d_ff)
        dropout_mask = self.dropout_mask.reshape(-1, self.d_ff)  # (B*N, d_ff)
        dout = dout.reshape(-1, d_model)  # (B*N, d_model)

        # =========================
        # 1. Linear2 backward
        # out = hidden_drop @ W2 + b2
        # =========================
        self.dW2 = hidden_drop.T @ dout
        self.db2 = np.sum(dout, axis=0)

        dhidden_drop = dout @ self.W2.T

        # =========================
        # 2. Dropout backward
        # hidden_drop = hidden * mask / keep_prob
        # =========================
        if self.dropout_rate > 0:
            keep_prob = 1.0 - self.dropout_rate
            dhidden = dhidden_drop * dropout_mask / keep_prob
        else:
            dhidden = dhidden_drop

        # =========================
        # 3. ReLU backward
        # =========================
        dhidden[hidden_linear <= 0] = 0

        # =========================
        # 4. Linear1 backward
        # hidden_linear = X @ W1 + b1
        # =========================
        self.dW1 = X.T @ dhidden
        self.db1 = np.sum(dhidden, axis=0)

        dx = dhidden @ self.W1.T
        dx = dx.reshape(B, N, d_model)

        return dx


if __name__ == "__main__":
    B, N, d_model = 2, 3, 4
    d_ff = 5

    X = np.random.randn(B, N, d_model)

    ffn = PositionwiseFFN(d_model, d_ff)

    # ===== forward =====
    out = ffn.forward(X)
    print("forward 输出:", out.shape)

    # ===== 假设 loss 梯度 =====
    dout = np.random.randn(B, N, d_model)

    # ===== backward =====
    dx = ffn.backward(dout)

    print("dx shape:", dx.shape)
    print("dW1 shape:", ffn.dW1.shape)
    print("dW2 shape:", ffn.dW2.shape)