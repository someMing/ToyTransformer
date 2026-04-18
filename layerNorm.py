import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        # backward需要
        self.X = None
        self.mean = None
        self.var = None
        self.std = None
        self.X_norm = None

        self.d_model = d_model
        self.eps = eps

        # 可学习参数
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, X):
        """
        X: (B, N, d_model)
        """
        # ===== 保存输入 =====
        self.X = X

        # ===== 1. 均值 =====
        self.mean = np.mean(X, axis=-1, keepdims=True)

        # ===== 2. 方差 =====
        self.var = np.var(X, axis=-1, keepdims=True)

        # ===== 3. 标准差 =====
        self.std = np.sqrt(self.var + self.eps)

        # ===== 4. 标准化 =====
        self.X_norm = (X - self.mean) / self.std

        # ===== 5. 缩放 + 平移 =====
        out = self.gamma * self.X_norm + self.beta

        return out


    def backward(self, dout):
        """
        dout: (B, N, d_model)
        return:
            dx: (B, N, d_model)
        """

        B, N, D = dout.shape

        # ===== 1. gamma / beta 梯度 =====
        self.dgamma = np.sum(dout * self.X_norm, axis=(0, 1))  # (d_model,)
        self.dbeta = np.sum(dout, axis=(0, 1))  # (d_model,)

        # ===== 2. dxhat =====
        dxhat = dout * self.gamma  # (B,N,D)

        # ===== 3. 三项 =====
        sum_dxhat = np.sum(dxhat, axis=-1, keepdims=True)  # (B,N,1)
        sum_dxhat_xhat = np.sum(dxhat * self.X_norm, axis=-1, keepdims=True)

        # ===== 4. 核心公式 =====
        dx = (dxhat
              - sum_dxhat / D
              - self.X_norm * sum_dxhat_xhat / D) / self.std

        return dx


if __name__ == "__main__":
    B, N, d_model = 2, 4, 6

    X = np.random.randn(B, N, d_model)

    ln = LayerNorm(d_model)

    out = ln.forward(X)

    print("输入 shape:", X.shape)
    print("输出 shape:", out.shape)

    # 验证均值≈0，方差≈1
    print("\n检查每个 token：")
    print("mean:", np.mean(out, axis=-1))
    print("var :", np.var(out, axis=-1))