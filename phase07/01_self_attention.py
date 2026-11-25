import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    # 扣掉每一列的最大值，避免 exp(x) 溢位（Numerical Stability）
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Q, K, V shape: (seq_len, d_k)
    causal=True 時套用上三角遮罩（未來 token 權重歸零）
    回傳: (attention_output, attention_weights)
    """
    d_k = Q.shape[-1]

    # 1. 算兩兩 token 的相似度分數 -> shape: (seq_len, seq_len)
    scores = Q @ K.T

    # 2. 除以 sqrt(d_k)，防止維度變大時點積結果太大、把 softmax 推到梯度飽和區
    scaled_scores = scores / np.sqrt(d_k)

    # 3. Causal Mask：把右上角（未來位置）填入 -inf，softmax 後權重就會變成 0
    if causal:
        seq_len = Q.shape[0]
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scaled_scores = np.where(mask, -np.inf, scaled_scores)

    # 4. 轉成機率分布（每一列總和為 1）
    weights = softmax(scaled_scores)

    # 5. 用權重對 V 做加權平均 -> shape: (seq_len, d_k)
    output = weights @ V
    return output, weights


if __name__ == "__main__":
    np.random.seed(42)
    # 模擬 4 個 token，每個 token 維度為 8
    X = np.random.randn(4, 8)
    # 先簡化：直接用 X 當 Q, K, V（實務上會各自乘上可學習的權重矩陣 W_q, W_k, W_v）
    _, w_bidir = scaled_dot_product_attention(X, X, X, causal=False)
    _, w_causal = scaled_dot_product_attention(X, X, X, causal=True)

    np.set_printoptions(precision=3, suppress=True)
    print("=== Bidirectional (Encoder / BERT 類型，看得到未來) ===")
    print(w_bidir)
    print("\n=== Causal (Decoder / GPT 類型，看不到未來) ===")
    print(w_causal)
