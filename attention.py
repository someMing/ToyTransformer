import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / (sum_exp + 1e-9) # 数值稳定

def generate_subsequent_mask(N):
    mask = np.triu(np.ones((N, N)), k=1)  # 上三角（不含对角线）
    mask = mask * (-1e9)  # 或 -np.inf
    return mask

class MultiHeadAttention:
    def __init__(self, d_model, heads):
        self.d_model = d_model
        self.heads = heads
        self.d_head = d_model // heads

        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01

        self.Wo = np.random.randn(d_model, d_model) * 0.01


    def forward(self, Q_in, K_in, V_in, mask=None):
        """
        Q_in: (B, N_q, d_model)
        K_in: (B, N_k, d_model)
        V_in: (B, N_k, d_model)
        """

        B, N_q, _ = Q_in.shape
        _, N_k, _ = K_in.shape

        # ===== 保存输入 =====
        self.Q_in = Q_in
        self.K_in = K_in
        self.V_in = V_in
        self.mask = mask

        # ===== 1. 线性变换 =====
        # 可以理解为特意化维度信息为 q k v
        self.Q_linear = Q_in @ self.Wq  # (B, N_q, d_model)
        self.K_linear = K_in @ self.Wk  # (B, N_k, d_model)
        self.V_linear = V_in @ self.Wv  # (B, N_k, d_model)

        # ===== 2. reshape =====
        Q = self.Q_linear.reshape(B, N_q, self.heads, self.d_head)
        K = self.K_linear.reshape(B, N_k, self.heads, self.d_head)
        V = self.V_linear.reshape(B, N_k, self.heads, self.d_head)

        # ===== 3. transpose =====
        self.Q = Q.transpose(0, 2, 1, 3)  # (B, heads, N_q, d_head)
        self.K = K.transpose(0, 2, 1, 3)  # (B, heads, N_k, d_head)
        self.V = V.transpose(0, 2, 1, 3)  # (B, heads, N_k, d_head)

        # ===== 4. attention =====
        # scores = Q K^T / sqrt(d_head)
        # 注意力分数 自注意力机制中可以理解为对该句中某个词的注意力
        # cross注意力机制中 q来自encoder，而k v来自decoder，注意力分数可以理解为对已生成的pre_tgt的兴趣
        self.scores = self.Q @ self.K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)
        # (B, heads, N_q, N_k)

        if mask is not None:
            self.scores_masked = self.scores + mask
        else:
            self.scores_masked = self.scores

        self.weights = softmax(self.scores_masked)  # (B, heads, N_q, N_k)

        self.attn_out = self.weights @ self.V  # (B, heads, N_q, d_head)

        # ===== 5. concat =====
        self.attn_out_transposed = self.attn_out.transpose(0, 2, 1, 3)  # (B, N_q, heads, d_head)
        self.attn_out_concat = self.attn_out_transposed.reshape(B, N_q, self.d_model)

        # ===== 6. 输出映射 =====
        # 将注意力分数得到的内容（V）最终转化为N_q每个维度的一个值
        out = self.attn_out_concat @ self.Wo  # (B, N_q, d_model)

        return out

    def self_attn_forward(self, X, mask=None):
        return self.forward(X, X, X, mask)

    def cross_attn_forward(self, Q_in, K_in, V_in, mask=None):
        return self.forward(Q_in, K_in, V_in, mask)


    def backward(self, dout):
        """
        dout: (B, N_q, d_model)

        return:
            dQ_in: (B, N_q, d_model)
            dK_in: (B, N_k, d_model)
            dV_in: (B, N_k, d_model)
        """

        B, N_q, _ = dout.shape
        N_k = self.K_in.shape[1]

        # =========================
        # 1. 输出层 backward
        # out = attn_out_concat @ Wo
        #
        # 公式:
        # y = xW
        # dW = x^T @ dy
        # dx = dy @ W^T
        # =========================
        attn_out_concat_flat = self.attn_out_concat.reshape(-1, self.d_model)   # (B*N_q, d_model)
        dout_flat = dout.reshape(-1, self.d_model)                               # (B*N_q, d_model)

        self.dWo = attn_out_concat_flat.T @ dout_flat                            # (d_model, d_model)
        dattn_out_concat_flat = dout_flat @ self.Wo.T                            # (B*N_q, d_model)
        dattn_out_concat = dattn_out_concat_flat.reshape(B, N_q, self.d_model)   # (B, N_q, d_model)

        # =========================
        # 2. concat backward
        # attn_out_concat = transpose(attn_out).reshape(...)
        #
        # 这里只是 reshape + transpose 的逆操作
        # =========================
        dattn_out_transposed = dattn_out_concat.reshape(B, N_q, self.heads, self.d_head)
        dattn_out = dattn_out_transposed.transpose(0, 2, 1, 3)   # (B, heads, N_q, d_head)

        # =========================
        # 3. attn_out = weights @ V
        #
        # 公式:
        # Y = A @ B
        # dA = dY @ B^T
        # dB = A^T @ dY
        # =========================
        dweights = dattn_out @ self.V.transpose(0, 1, 3, 2)      # (B, heads, N_q, N_k)
        dV = self.weights.transpose(0, 1, 3, 2) @ dattn_out      # (B, heads, N_k, d_head)

        # =========================
        # 4. softmax backward
        # weights = softmax(scores_masked)
        #
        # 公式:
        # dS = P * (dP - sum(dP * P))
        # 其中:
        # P = softmax(S)
        # =========================
        sum_term = np.sum(dweights * self.weights, axis=-1, keepdims=True)
        dscores_masked = self.weights * (dweights - sum_term)

        # 注意:
        # mask 是常数，不参与训练
        # scores_masked = scores + mask
        # 所以:
        # dscores = dscores_masked
        dscores = dscores_masked

        # =========================
        # 5. scores = Q @ K^T / sqrt(d_head)
        #
        # 公式:
        # S = QK^T / sqrt(d)
        # dQ = dS @ K / sqrt(d)
        # dK = dS^T @ Q / sqrt(d)
        # =========================
        scale = np.sqrt(self.d_head)
        dQ = dscores @ self.K / scale                            # (B, heads, N_q, d_head)
        dK = dscores.transpose(0, 1, 3, 2) @ self.Q / scale     # (B, heads, N_k, d_head)

        # =========================
        # 6. transpose + reshape backward
        # 把 dQ, dK, dV 从多头形式还原回线性层输出形式
        # =========================
        dQ_linear = dQ.transpose(0, 2, 1, 3).reshape(B, N_q, self.d_model)   # (B, N_q, d_model)
        dK_linear = dK.transpose(0, 2, 1, 3).reshape(B, N_k, self.d_model)   # (B, N_k, d_model)
        dV_linear = dV.transpose(0, 2, 1, 3).reshape(B, N_k, self.d_model)   # (B, N_k, d_model)

        # =========================
        # 7. 输入线性层 backward
        # Q_linear = Q_in @ Wq
        # K_linear = K_in @ Wk
        # V_linear = V_in @ Wv
        #
        # 公式:
        # y = xW
        # dW = x^T @ dy
        # dx = dy @ W^T
        # =========================
        Q_in_flat = self.Q_in.reshape(-1, self.d_model)             # (B*N_q, d_model)
        K_in_flat = self.K_in.reshape(-1, self.d_model)             # (B*N_k, d_model)
        V_in_flat = self.V_in.reshape(-1, self.d_model)             # (B*N_k, d_model)

        dQ_linear_flat = dQ_linear.reshape(-1, self.d_model)
        dK_linear_flat = dK_linear.reshape(-1, self.d_model)
        dV_linear_flat = dV_linear.reshape(-1, self.d_model)

        self.dWq = Q_in_flat.T @ dQ_linear_flat                     # (d_model, d_model)
        self.dWk = K_in_flat.T @ dK_linear_flat                     # (d_model, d_model)
        self.dWv = V_in_flat.T @ dV_linear_flat                     # (d_model, d_model)

        dQ_in_flat = dQ_linear_flat @ self.Wq.T
        dK_in_flat = dK_linear_flat @ self.Wk.T
        dV_in_flat = dV_linear_flat @ self.Wv.T

        dQ_in = dQ_in_flat.reshape(B, N_q, self.d_model)
        dK_in = dK_in_flat.reshape(B, N_k, self.d_model)
        dV_in = dV_in_flat.reshape(B, N_k, self.d_model)

        return dQ_in, dK_in, dV_in

if __name__ == "__main__":
    B = 2
    N = 4
    d_model = 8
    heads = 2

    X = np.random.randn(B, N, d_model)
    dout = np.random.randn(B, N, d_model)

    attn = MultiHeadAttention(d_model, heads)

    out = attn.self_attn_forward(X)
    dQ, dK, dV = attn.backward(dout)

    print("输入 shape:", X.shape)
    print("输出 shape:", out.shape)
    print("dQ shape:", dQ.shape)
    print("dK shape:", dK.shape)
    print("dV shape:", dV.shape)
    print("dWq shape:", attn.dWq.shape)
    print("dWk shape:", attn.dWk.shape)
    print("dWv shape:", attn.dWv.shape)
    print("dWo shape:", attn.dWo.shape)