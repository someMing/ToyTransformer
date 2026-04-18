import numpy as np
import attention as at
import ffn as FFN
import layerNorm as LN
import embedding as emb
from mask import create_padding_mask, create_decoder_mask


class EncoderLayer:
    def __init__(self, d_model, heads, d_ff):
        self.attn = at.MultiHeadAttention(d_model, heads)
        self.ffn = FFN.PositionwiseFFN(d_model, d_ff)

        self.norm1 = LN.LayerNorm(d_model)
        self.norm2 = LN.LayerNorm(d_model)

    def forward(self, x, src_mask=None, training=True):
        """
        x: (B, N, d_model)
        """

        # ===== cache 输入 =====
        self.x0 = x
        self.src_mask = src_mask

        # ===== 1. Attention =====
        self.attn_out = self.attn.self_attn_forward(x, src_mask)

        # ===== 2. Residual + Norm =====
        self.y1 = self.x0 + self.attn_out
        self.x1 = self.norm1.forward(self.y1)

        # ===== 3. FFN =====
        self.ffn_out = self.ffn.forward(self.x1, training=training)

        # ===== 4. Residual + Norm =====
        self.y2 = self.x1 + self.ffn_out
        self.x2 = self.norm2.forward(self.y2)

        return self.x2

    def backward(self, dout):
        """
        dout: (B, N, d_model)
        return:
            dx: (B, N, d_model)
        """

        # Step 1. Norm2 backward
        dy2 = self.norm2.backward(dout)

        # =====================================
        # Step 2. residual split
        # y2 = x1 + ffn_out
        #
        # 公式:
        # d(x1 + ffn_out)/dx1 = 1
        # d(x1 + ffn_out)/dffn_out = 1
        #
        # 所以:
        # dx1_from_residual = dy2
        # dffn_out = dy2
        # =====================================
        dx1_from_residual = dy2
        dffn_out = dy2

        # Step 3. FFN backward
        dx1_from_ffn = self.ffn.backward(dffn_out)

        # =====================================
        # Step 4. 合并到 x1
        # x1 一方面直接走残差
        # 一方面进入 FFN
        #
        # 所以:
        # dx1 = dx1_from_residual + dx1_from_ffn
        # =====================================
        dx1 = dx1_from_residual + dx1_from_ffn

        # =====================================
        # Step 5. Norm1 backward
        # x1 = norm1(y1)
        # 已知 dx1，求 dy1
        # =====================================
        dy1 = self.norm1.backward(dx1)

        # =====================================
        # Step 6. residual split
        # y1 = x0 + attn_out
        #
        # 所以:
        # dx0_from_residual = dy1
        # dattn_out = dy1
        # =====================================
        dx0_from_residual = dy1
        dattn_out = dy1

        # =====================================
        # Step 7. Attention backward
        # attn_out = self_attn(x0, x0, x0)
        #
        # backward 返回:
        # dQ_in, dK_in, dV_in
        #
        # 因为 self-attention 三个输入都是同一个 x0，
        # 所以要把三路梯度加起来
        # =====================================
        dQ_in, dK_in, dV_in = self.attn.backward(dattn_out)

        dx0_from_attn = dQ_in + dK_in + dV_in

        # =====================================
        # Step 8. 合并到 x0
        # =====================================
        dx0 = dx0_from_residual + dx0_from_attn

        return dx0

class DecoderLayer:
    def __init__(self, d_model, heads, d_ff):
        self.self_attn = at.MultiHeadAttention(d_model, heads)
        self.cross_attn = at.MultiHeadAttention(d_model, heads)
        self.ffn = FFN.PositionwiseFFN(d_model, d_ff)

        self.norm1 = LN.LayerNorm(d_model)
        self.norm2 = LN.LayerNorm(d_model)
        self.norm3 = LN.LayerNorm(d_model)

    def forward(self, x, encoder_out, tgt_mask=None, src_mask=None, training=True):
        """
        x: (B, N_tgt, d_model)
        encoder_out: (B, N_src, d_model)
        """

        self.x0 = x
        self.encoder_out = encoder_out
        self.tgt_mask = tgt_mask
        self.src_mask = src_mask

        # ===== 1. Masked Self-Attention =====
        self.attn1 = self.self_attn.self_attn_forward(self.x0, tgt_mask)
        self.y1 = self.x0 + self.attn1
        self.x1 = self.norm1.forward(self.y1)

        # ===== 2. Cross Attention =====
        self.attn2 = self.cross_attn.cross_attn_forward(self.x1, encoder_out, encoder_out, src_mask)
        self.y2 = self.x1 + self.attn2
        self.x2 = self.norm2.forward(self.y2)

        # ===== 3. FFN =====
        self.ffn_out = self.ffn.forward(self.x2, training=training)
        self.y3 = self.x2 + self.ffn_out
        self.x3 = self.norm3.forward(self.y3)

        return self.x3

    def backward(self, dout):
        """
        dout: (B, N_tgt, d_model)

        return:
            dx: (B, N_tgt, d_model)          # 回传给 decoder 输入
            dencoder_out: (B, N_src, d_model)  # 回传给 encoder_out
        """

        # Step 1. Norm3 backward
        dy3 = self.norm3.backward(dout)

        # Step 2. residual split
        dx2_from_residual = dy3
        dffn_out = dy3

        # Step 3. FFN backward
        dx2_from_ffn = self.ffn.backward(dffn_out)

        # Step 4. merge to x2
        dx2 = dx2_from_residual + dx2_from_ffn

        # Step 5. Norm2 backward
        dy2  = self.norm2.backward(dx2)

        # Step 6. residual split
        # y2 = x1 + attn2
        dx1_from_residual = dy2
        dattn2 = dy2

        # =====================================
        # Step 7. Cross-Attention backward
        # attn2 = cross_attn(x1, encoder_out, encoder_out)
        #
        # 返回:
        # dQ_in -> 回到 x1
        # dK_in, dV_in -> 回到 encoder_out
        # 因为 K_in 和 V_in 都是 encoder_out
        # 所以 encoder_out 的梯度要相加
        # =====================================
        dQ_cross, dK_cross, dV_cross = self.cross_attn.backward(dattn2)

        dx1_from_cross = dQ_cross
        dencoder_out_from_cross = dK_cross + dV_cross

        # =====================================
        # Step 8. merge to x1
        # =====================================
        dx1 = dx1_from_residual + dx1_from_cross

        # =====================================
        # Step 9. Norm1 backward
        # x1 = norm1(y1)
        # =====================================
        dy1 = self.norm1.backward(dx1)

        # =====================================
        # Step 10. residual split
        # y1 = x0 + attn1
        #
        # dx0_from_residual = dy1
        # dattn1 = dy1
        # =====================================
        dx0_from_residual = dy1
        dattn1 = dy1

        # =====================================
        # Step 11. Self-Attention backward
        # attn1 = self_attn(x0, x0, x0)
        #
        # 三路输入都是 x0
        # 所以梯度要相加
        # =====================================
        dQ_self, dK_self, dV_self = self.self_attn.backward(dattn1)
        dx0_from_self = dQ_self + dK_self + dV_self

        # =====================================
        # Step 12. merge to x0
        # =====================================
        dx0 = dx0_from_residual + dx0_from_self

        return dx0, dencoder_out_from_cross

# 堆叠
class Encoder:
    def __init__(self, num_layers, d_model, heads, d_ff):
        self.layers = [EncoderLayer(d_model, heads, d_ff) for _ in range(num_layers)]

    def forward(self, x, src_mask=None, training=True):
        for layer in self.layers:
            x = layer.forward(x, src_mask, training=training)
        return x

    def backward(self, dout):
        """
        dout: (B, N, d_model)
        return:
            dx: (B, N, d_model)
        """

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

class Decoder:
    def __init__(self, num_layers, d_model, heads, d_ff):
        self.layers = [DecoderLayer(d_model, heads, d_ff) for _ in range(num_layers)]

    def forward(self, x, encoder_out, tgt_mask, src_mask, training=True):
        """
        x: (B, N_tgt, d_model)
        encoder_out: (B, N_src, d_model)
        """
        self.encoder_out = encoder_out
        for layer in self.layers:
            x = layer.forward(x, encoder_out, tgt_mask, src_mask, training=training)
        return x

    def backward(self, dout):
        """
        dout: (B, N_tgt, d_model)

        return:
            dx: (B, N_tgt, d_model)
            dencoder_out_total: (B, N_src, d_model)
        """

        dencoder_out_total = np.zeros_like(self.encoder_out)

        for layer in reversed(self.layers):
            dout, dencoder_out = layer.backward(dout)
            dencoder_out_total += dencoder_out

        return dout, dencoder_out_total

class OutputLayer:
    def __init__(self, d_model, vocab_size):
        self.W = np.random.randn(d_model, vocab_size) * 0.01

    def forward(self, x):
        """
        x: (B, N, d_model)
        return: (B, N, vocab_size)
        """
        self.x = x
        return x @ self.W

    def backward(self, dlogits):
        """
        dlogits: (B, N, vocab_size)
        return:
            dx: (B, N, d_model)
        """

        B, N, d_model = self.x.shape
        V = dlogits.shape[-1]

        x_flat = self.x.reshape(-1, d_model)           # (B*N, d_model)
        dlogits_flat = dlogits.reshape(-1, V)          # (B*N, V)

        self.dW = x_flat.T @ dlogits_flat              # (d_model, V)
        dx_flat = dlogits_flat @ self.W.T              # (B*N, d_model)
        dx = dx_flat.reshape(B, N, d_model)

        return dx

class Transformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, heads, d_ff, max_len=20):
        # ===== src / tgt embedding 分离 =====
        self.src_embedding = emb.InputEmbedding(src_vocab_size, d_model, max_len)
        self.tgt_embedding = emb.InputEmbedding(tgt_vocab_size, d_model, max_len)

        self.encoder = Encoder(num_layers, d_model, heads, d_ff)
        self.decoder = Decoder(num_layers, d_model, heads, d_ff)

        # ===== 输出层对应 tgt 词表 =====
        self.output_layer = OutputLayer(d_model, tgt_vocab_size)

        #adam
        self.adam_states = {}
        self.adam_step = 0

    def forward(self, src_ids, tgt_ids, src_pad_id, tgt_pad_id):
        # ===== cache =====
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

        # ===== 1. embedding =====
        self.src = self.src_embedding.forward(src_ids)
        self.tgt = self.tgt_embedding.forward(tgt_ids)

        # ===== 2. mask =====
        self.src_mask = create_padding_mask(src_ids, src_pad_id)   # encoder mask
        self.tgt_mask = create_decoder_mask(tgt_ids, tgt_pad_id)   # decoder mask

        # ===== 3. encoder =====
        #self.encoder_out = self.encoder.forward(self.src, self.src_mask)
        self.encoder_out = self.encoder.forward(self.src, self.src_mask, training=True)

        # ===== 4. decoder =====
        #self.decoder_out = self.decoder.forward(self.tgt, self.encoder_out, self.tgt_mask, self.src_mask)
        self.decoder_out = self.decoder.forward(self.tgt, self.encoder_out, self.tgt_mask, self.src_mask, training=True)

        # ===== 5. 输出 =====
        self.logits = self.output_layer.forward(self.decoder_out)

        return self.logits

    def cross_entropy_loss(self, logits, targets, tgt_pad_id):
        """
        logits: (B, N, tgt_vocab_size)
        targets: (B, N)
        tgt_pad_id: 目标语言 PAD token 的 id
        """
        B, N, V = logits.shape

        # ===== 1. 数值稳定 logsumexp =====
        max_logits = np.max(logits, axis=-1, keepdims=True)  # (B,N,1)
        logsumexp = max_logits + np.log(
            np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True)
        )  # (B,N,1)

        # ===== 2. 正确类别 logits =====
        correct_logits = logits[np.arange(B)[:, None], np.arange(N), targets]  # (B,N)

        # ===== 3. 基础 loss =====
        loss = -correct_logits + logsumexp.squeeze(-1)  # (B,N)

        # ===== 4. PAD mask =====
        mask = (targets != tgt_pad_id).astype(np.float32)
        loss = loss * mask

        # ===== 5. 平均 loss =====
        denom = np.sum(mask)
        denom = max(denom, 1e-9)
        return np.sum(loss) / denom

    def cross_entropy_backward(self, logits, targets, tgt_pad_id):
        """
        logits: (B, N, V)
        targets: (B, N)
        tgt_pad_id: 目标语言 PAD token id
        return: dlogits, same shape as logits
        """
        B, N, V = logits.shape

        # ===== 1. stable softmax =====
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # ===== 2. one-hot targets =====
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(B)[:, None], np.arange(N), targets] = 1

        # ===== 3. mask =====
        mask = (targets != tgt_pad_id).astype(np.float32)  # (B,N)

        # ===== 4. gradient =====
        dlogits = (probs - one_hot) * mask[:, :, None]

        # ===== 5. normalize =====
        denom = np.sum(mask)
        denom = max(denom, 1e-9)
        dlogits /= denom

        return dlogits

    def backward(self, targets):
        """
        targets: (B, N_tgt)
        """

        # =====================================
        # Step 1. loss backward
        # =====================================
        dlogits = self.cross_entropy_backward(self.logits, targets, self.tgt_pad_id)

        # =====================================
        # Step 2. output layer backward
        # =====================================
        d_decoder_out = self.output_layer.backward(dlogits)

        # =====================================
        # Step 3. decoder backward
        # 返回:
        # d_tgt_embed, d_encoder_out
        # =====================================
        d_tgt_embed, d_encoder_out = self.decoder.backward(d_decoder_out)

        # =====================================
        # Step 4. encoder backward
        # =====================================
        d_src_embed = self.encoder.backward(d_encoder_out)

        # =====================================
        # Step 5. embedding backward
        # src / tgt embedding 已经分离，不再需要相加
        # =====================================
        self.src_embedding.dW = self.src_embedding.backward(self.src_ids, d_src_embed)
        self.tgt_embedding.dW = self.tgt_embedding.backward(self.tgt_ids, d_tgt_embed)

        grads = {
            "src_embedding.W": self.src_embedding.dW,
            "tgt_embedding.W": self.tgt_embedding.dW,
            "output.W": self.output_layer.dW,
        }

        return grads

    def update_parameters(self, lr):
        """
        使用 Adam 更新所有参数
        """

        # 每调用一次 update，就表示第 t 步
        self.adam_step += 1

        # =====================================
        # 1. Embedding
        # =====================================
        # _adam_update(self, param, grad, name, lr, beta1=0.9, beta2=0.999, eps=1e-8)
        self._adam_update(
            self.src_embedding.W,
            self.src_embedding.dW,
            "src_embedding.W",
            lr
        )

        self._adam_update(
            self.tgt_embedding.W,
            self.tgt_embedding.dW,
            "tgt_embedding.W",
            lr
        )

        # =====================================
        # 2. Output layer
        # =====================================
        self._adam_update(
            self.output_layer.W,
            self.output_layer.dW,
            "output_layer.W",
            lr
        )

        # =====================================
        # 3. Encoder layers
        # =====================================
        for i, layer in enumerate(self.encoder.layers):
            # ---- Attention ----
            self._adam_update(layer.attn.Wq, layer.attn.dWq, f"encoder.layers.{i}.attn.Wq", lr)
            self._adam_update(layer.attn.Wk, layer.attn.dWk, f"encoder.layers.{i}.attn.Wk", lr)
            self._adam_update(layer.attn.Wv, layer.attn.dWv, f"encoder.layers.{i}.attn.Wv", lr)
            self._adam_update(layer.attn.Wo, layer.attn.dWo, f"encoder.layers.{i}.attn.Wo", lr)

            # ---- FFN ----
            self._adam_update(layer.ffn.W1, layer.ffn.dW1, f"encoder.layers.{i}.ffn.W1", lr)
            self._adam_update(layer.ffn.b1, layer.ffn.db1, f"encoder.layers.{i}.ffn.b1", lr)
            self._adam_update(layer.ffn.W2, layer.ffn.dW2, f"encoder.layers.{i}.ffn.W2", lr)
            self._adam_update(layer.ffn.b2, layer.ffn.db2, f"encoder.layers.{i}.ffn.b2", lr)

            # ---- LayerNorm ----
            self._adam_update(layer.norm1.gamma, layer.norm1.dgamma, f"encoder.layers.{i}.norm1.gamma", lr)
            self._adam_update(layer.norm1.beta, layer.norm1.dbeta, f"encoder.layers.{i}.norm1.beta", lr)
            self._adam_update(layer.norm2.gamma, layer.norm2.dgamma, f"encoder.layers.{i}.norm2.gamma", lr)
            self._adam_update(layer.norm2.beta, layer.norm2.dbeta, f"encoder.layers.{i}.norm2.beta", lr)

        # =====================================
        # 4. Decoder layers
        # =====================================
        for i, layer in enumerate(self.decoder.layers):
            # ---- Self Attention ----
            self._adam_update(layer.self_attn.Wq, layer.self_attn.dWq, f"decoder.layers.{i}.self_attn.Wq", lr)
            self._adam_update(layer.self_attn.Wk, layer.self_attn.dWk, f"decoder.layers.{i}.self_attn.Wk", lr)
            self._adam_update(layer.self_attn.Wv, layer.self_attn.dWv, f"decoder.layers.{i}.self_attn.Wv", lr)
            self._adam_update(layer.self_attn.Wo, layer.self_attn.dWo, f"decoder.layers.{i}.self_attn.Wo", lr)

            # ---- Cross Attention ----
            self._adam_update(layer.cross_attn.Wq, layer.cross_attn.dWq, f"decoder.layers.{i}.cross_attn.Wq", lr)
            self._adam_update(layer.cross_attn.Wk, layer.cross_attn.dWk, f"decoder.layers.{i}.cross_attn.Wk", lr)
            self._adam_update(layer.cross_attn.Wv, layer.cross_attn.dWv, f"decoder.layers.{i}.cross_attn.Wv", lr)
            self._adam_update(layer.cross_attn.Wo, layer.cross_attn.dWo, f"decoder.layers.{i}.cross_attn.Wo", lr)

            # ---- FFN ----
            self._adam_update(layer.ffn.W1, layer.ffn.dW1, f"decoder.layers.{i}.ffn.W1", lr)
            self._adam_update(layer.ffn.b1, layer.ffn.db1, f"decoder.layers.{i}.ffn.b1", lr)
            self._adam_update(layer.ffn.W2, layer.ffn.dW2, f"decoder.layers.{i}.ffn.W2", lr)
            self._adam_update(layer.ffn.b2, layer.ffn.db2, f"decoder.layers.{i}.ffn.b2", lr)

            # ---- LayerNorm ----
            self._adam_update(layer.norm1.gamma, layer.norm1.dgamma, f"decoder.layers.{i}.norm1.gamma", lr)
            self._adam_update(layer.norm1.beta, layer.norm1.dbeta, f"decoder.layers.{i}.norm1.beta", lr)
            self._adam_update(layer.norm2.gamma, layer.norm2.dgamma, f"decoder.layers.{i}.norm2.gamma", lr)
            self._adam_update(layer.norm2.beta, layer.norm2.dbeta, f"decoder.layers.{i}.norm2.beta", lr)
            self._adam_update(layer.norm3.gamma, layer.norm3.dgamma, f"decoder.layers.{i}.norm3.gamma", lr)
            self._adam_update(layer.norm3.beta, layer.norm3.dbeta, f"decoder.layers.{i}.norm3.beta", lr)


    def train_step(self, src_ids, tgt_ids, targets, src_pad_id, tgt_pad_id, lr):
        """
        单步训练：
        forward -> loss -> backward -> update
        """

        # ===== 1. forward =====
        logits = self.forward(src_ids, tgt_ids, src_pad_id, tgt_pad_id)

        # ===== 2. loss =====
        loss = self.cross_entropy_loss(logits, targets, tgt_pad_id)

        # ===== 3. backward =====
        self.backward(targets)

        # ===== 4. update =====
        self.update_parameters(lr)

        return loss

    def predict(self, src_ids, bos_id, eos_id, src_pad_id, tgt_pad_id, max_len=20):
        """
        greedy decode

        src_ids: (B, N_src)
        return:
            generated_ids: (B, <= max_len)
        """

        B = src_ids.shape[0]

        # ===== 1. encoder =====
        src = self.src_embedding.forward(src_ids)
        src_mask = create_padding_mask(src_ids, src_pad_id)
        encoder_out = self.encoder.forward(src, src_mask, training=False)

        # ===== 2. 初始 decoder 输入：全是 BOS =====
        generated = np.full((B, 1), bos_id, dtype=np.int32)

        # ===== 3. 逐步生成 =====
        for _ in range(max_len - 1):
            tgt = self.tgt_embedding.forward(generated)
            tgt_mask = create_decoder_mask(generated, tgt_pad_id)

            decoder_out = self.decoder.forward(tgt, encoder_out, tgt_mask, src_mask, training=False)
            logits = self.output_layer.forward(decoder_out)  # (B, cur_len, tgt_vocab_size)

            # 只取最后一个时间步
            next_token_logits = logits[:, -1, :]  # (B, tgt_vocab_size)
            next_token = np.argmax(next_token_logits, axis=-1)  # (B,)

            # 拼接到 generated 后面
            generated = np.concatenate(
                [generated, next_token[:, None]],
                axis=1
            )

            # 如果全部都生成 eos，就提前停止
            if np.all(next_token == eos_id):
                break

        return generated

    def _adam_update(self, param, grad, name, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        对单个参数做 Adam 更新

        param: 参数本身（numpy数组）
        grad: 对应梯度（numpy数组）
        name: 该参数的唯一名字（字符串）
        """

        # 如果这个参数第一次出现，就初始化 m 和 v
        if name not in self.adam_states:
            self.adam_states[name] = {
                "m": np.zeros_like(param),
                "v": np.zeros_like(param)
            }

        m = self.adam_states[name]["m"]
        v = self.adam_states[name]["v"]

        # ===== 1. 更新一阶矩 =====
        # m_t = beta1 * m_{t-1} + (1-beta1) * g_t
        m[:] = beta1 * m + (1.0 - beta1) * grad

        # ===== 2. 更新二阶矩 =====
        # v_t = beta2 * v_{t-1} + (1-beta2) * (g_t^2)
        v[:] = beta2 * v + (1.0 - beta2) * (grad * grad)

        # ===== 3. bias correction =====
        m_hat = m / (1.0 - beta1 ** self.adam_step)
        v_hat = v / (1.0 - beta2 ** self.adam_step)

        # ===== 4. 参数更新 =====
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def get_parameters_dict(self):
        """
        导出所有模型参数，返回一个 dict
        之后可直接用于 np.savez
        """
        params = {}

        # =====================================
        # Embedding
        # =====================================
        params["src_embedding.W"] = self.src_embedding.W
        params["tgt_embedding.W"] = self.tgt_embedding.W

        # =====================================
        # Output layer
        # =====================================
        params["output_layer.W"] = self.output_layer.W

        # =====================================
        # Encoder
        # =====================================
        for i, layer in enumerate(self.encoder.layers):
            prefix = f"encoder.layers.{i}"

            # Attention
            params[f"{prefix}.attn.Wq"] = layer.attn.Wq
            params[f"{prefix}.attn.Wk"] = layer.attn.Wk
            params[f"{prefix}.attn.Wv"] = layer.attn.Wv
            params[f"{prefix}.attn.Wo"] = layer.attn.Wo

            # FFN
            params[f"{prefix}.ffn.W1"] = layer.ffn.W1
            params[f"{prefix}.ffn.b1"] = layer.ffn.b1
            params[f"{prefix}.ffn.W2"] = layer.ffn.W2
            params[f"{prefix}.ffn.b2"] = layer.ffn.b2

            # LayerNorm
            params[f"{prefix}.norm1.gamma"] = layer.norm1.gamma
            params[f"{prefix}.norm1.beta"] = layer.norm1.beta
            params[f"{prefix}.norm2.gamma"] = layer.norm2.gamma
            params[f"{prefix}.norm2.beta"] = layer.norm2.beta

        # =====================================
        # Decoder
        # =====================================
        for i, layer in enumerate(self.decoder.layers):
            prefix = f"decoder.layers.{i}"

            # Self Attention
            params[f"{prefix}.self_attn.Wq"] = layer.self_attn.Wq
            params[f"{prefix}.self_attn.Wk"] = layer.self_attn.Wk
            params[f"{prefix}.self_attn.Wv"] = layer.self_attn.Wv
            params[f"{prefix}.self_attn.Wo"] = layer.self_attn.Wo

            # Cross Attention
            params[f"{prefix}.cross_attn.Wq"] = layer.cross_attn.Wq
            params[f"{prefix}.cross_attn.Wk"] = layer.cross_attn.Wk
            params[f"{prefix}.cross_attn.Wv"] = layer.cross_attn.Wv
            params[f"{prefix}.cross_attn.Wo"] = layer.cross_attn.Wo

            # FFN
            params[f"{prefix}.ffn.W1"] = layer.ffn.W1
            params[f"{prefix}.ffn.b1"] = layer.ffn.b1
            params[f"{prefix}.ffn.W2"] = layer.ffn.W2
            params[f"{prefix}.ffn.b2"] = layer.ffn.b2

            # LayerNorm
            params[f"{prefix}.norm1.gamma"] = layer.norm1.gamma
            params[f"{prefix}.norm1.beta"] = layer.norm1.beta
            params[f"{prefix}.norm2.gamma"] = layer.norm2.gamma
            params[f"{prefix}.norm2.beta"] = layer.norm2.beta
            params[f"{prefix}.norm3.gamma"] = layer.norm3.gamma
            params[f"{prefix}.norm3.beta"] = layer.norm3.beta

        return params

    def load_parameters_dict(self, params):
        """
        从 dict 中加载所有模型参数
        """

        # =====================================
        # Embedding
        # =====================================
        self.src_embedding.W = params["src_embedding.W"]
        self.tgt_embedding.W = params["tgt_embedding.W"]

        # =====================================
        # Output layer
        # =====================================
        self.output_layer.W = params["output_layer.W"]

        # =====================================
        # Encoder
        # =====================================
        for i, layer in enumerate(self.encoder.layers):
            prefix = f"encoder.layers.{i}"

            layer.attn.Wq = params[f"{prefix}.attn.Wq"]
            layer.attn.Wk = params[f"{prefix}.attn.Wk"]
            layer.attn.Wv = params[f"{prefix}.attn.Wv"]
            layer.attn.Wo = params[f"{prefix}.attn.Wo"]

            layer.ffn.W1 = params[f"{prefix}.ffn.W1"]
            layer.ffn.b1 = params[f"{prefix}.ffn.b1"]
            layer.ffn.W2 = params[f"{prefix}.ffn.W2"]
            layer.ffn.b2 = params[f"{prefix}.ffn.b2"]

            layer.norm1.gamma = params[f"{prefix}.norm1.gamma"]
            layer.norm1.beta = params[f"{prefix}.norm1.beta"]
            layer.norm2.gamma = params[f"{prefix}.norm2.gamma"]
            layer.norm2.beta = params[f"{prefix}.norm2.beta"]

        # =====================================
        # Decoder
        # =====================================
        for i, layer in enumerate(self.decoder.layers):
            prefix = f"decoder.layers.{i}"

            layer.self_attn.Wq = params[f"{prefix}.self_attn.Wq"]
            layer.self_attn.Wk = params[f"{prefix}.self_attn.Wk"]
            layer.self_attn.Wv = params[f"{prefix}.self_attn.Wv"]
            layer.self_attn.Wo = params[f"{prefix}.self_attn.Wo"]

            layer.cross_attn.Wq = params[f"{prefix}.cross_attn.Wq"]
            layer.cross_attn.Wk = params[f"{prefix}.cross_attn.Wk"]
            layer.cross_attn.Wv = params[f"{prefix}.cross_attn.Wv"]
            layer.cross_attn.Wo = params[f"{prefix}.cross_attn.Wo"]

            layer.ffn.W1 = params[f"{prefix}.ffn.W1"]
            layer.ffn.b1 = params[f"{prefix}.ffn.b1"]
            layer.ffn.W2 = params[f"{prefix}.ffn.W2"]
            layer.ffn.b2 = params[f"{prefix}.ffn.b2"]

            layer.norm1.gamma = params[f"{prefix}.norm1.gamma"]
            layer.norm1.beta = params[f"{prefix}.norm1.beta"]
            layer.norm2.gamma = params[f"{prefix}.norm2.gamma"]
            layer.norm2.beta = params[f"{prefix}.norm2.beta"]
            layer.norm3.gamma = params[f"{prefix}.norm3.gamma"]
            layer.norm3.beta = params[f"{prefix}.norm3.beta"]

if __name__ == "__main__":
    np.random.seed(42)

    B = 2
    N_src = 5
    N_tgt = 4

    src_vocab_size = 20
    tgt_vocab_size = 20

    src_pad_id = 0
    tgt_pad_id = 0
    bos_id = 1
    eos_id = 2

    epochs = 1000
    lr = 0.001

    # ===== toy 数据 =====
    # src 是“中文侧 id”，tgt/targets 是“英文侧 id”
    src_ids = np.array([
        [1, 4, 5, 0, 0],
        [1, 6, 7, 8, 0],
    ], dtype=np.int32)

    tgt_ids = np.array([
        [1, 9, 10, 2],   # <bos> ...
        [1, 11, 12, 2],
    ], dtype=np.int32)

    targets = np.array([
        [9, 10, 2, 0],   # ... <eos>
        [11, 12, 2, 0],
    ], dtype=np.int32)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=8,
        num_layers=2,
        heads=2,
        d_ff=16,
        max_len=10
    )

    for epoch in range(epochs):
        loss = model.train_step(
            src_ids=src_ids,
            tgt_ids=tgt_ids,
            targets=targets,
            src_pad_id=src_pad_id,
            tgt_pad_id=tgt_pad_id,
            lr=lr
        )

        if epoch % 5 == 0:
            print(f"epoch {epoch}, loss = {loss:.6f}")

    # ===== 简单测试 predict =====
    pred_ids = model.predict(
        src_ids=src_ids,
        bos_id=bos_id,
        eos_id=eos_id,
        src_pad_id=src_pad_id,
        tgt_pad_id=tgt_pad_id,
        max_len=10
    )

    print("\npred_ids:")
    print(pred_ids)