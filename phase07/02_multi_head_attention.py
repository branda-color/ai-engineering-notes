import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """
    將 (seq_len, d_model) 拆成 (num_heads, seq_len, d_k)
    """
    seq_len, d_model = x.shape
    assert d_model % num_heads == 0, "d_model 必須能被 num_heads 整除"
    d_k = d_model // num_heads

    # 先在最後一維切開: (seq_len, num_heads, d_k)
    x = x.reshape(seq_len, num_heads, d_k)
    # 把 num_heads 拉到最前面，讓每個 head 可以各自平行做矩陣運算
    return x.transpose(1, 0, 2)


def combine_heads(x: np.ndarray) -> np.ndarray:
    """
    將 (num_heads, seq_len, d_k) 拼回 (seq_len, d_model)
    """
    num_heads, seq_len, d_k = x.shape
    # 先把 seq_len 換回第一維: (seq_len, num_heads, d_k)
    x = x.transpose(1, 0, 2)
    return x.reshape(seq_len, num_heads * d_k)


def multi_head_attention(
    X: np.ndarray, num_heads: int, causal: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    X shape: (seq_len, d_model)
    回傳: (output, all_heads_weights)
    """
    # 1. 把 (seq_len, d_model) 拆成 (num_heads, seq_len, d_k)
    Q = split_heads(X, num_heads)
    K = split_heads(X, num_heads)
    V = split_heads(X, num_heads)

    d_k = Q.shape[-1]

    # 2. 每個頭平行算相似度：只轉置最後兩維 (seq_len, d_k) -> (d_k, seq_len)
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)

    # 3. 若為 Causal 模式，所有頭套用同樣的上三角遮罩
    if causal:
        seq_len = X.shape[0]
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    weights = softmax(scores)

    # 4. 每個頭各自對 V 加權平均 -> (num_heads, seq_len, d_k)
    head_outputs = weights @ V

    # 5. 拼回 (seq_len, d_model)
    output = combine_heads(head_outputs)
    return output, weights


if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_model, num_heads = 4, 8, 2

    X = np.random.randn(seq_len, d_model)
    output, weights = multi_head_attention(X, num_heads=num_heads, causal=True)

    np.set_printoptions(precision=3, suppress=True)
    print("輸入 shape:", X.shape)
    print("輸出 shape:", output.shape)
    print("每個頭的權重 shape:", weights.shape)
    print("\nHead 0 注意力權重:")
    print(weights[0])
    print("\nHead 1 注意力權重:")
    print(weights[1])
