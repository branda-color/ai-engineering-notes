import time
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def step_no_cache(tokens: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> np.ndarray:
    """
    無快取：每多生 1 個字，都把「從頭到現在所有的字」重新乘一次權重並算整張 Attention
    tokens shape: (t, d_model)
    回傳最後一個字的輸出向量: (1, d_model)
    """
    Q = tokens @ W_q  # (t, d_model)
    K = tokens @ W_k  # (t, d_model)
    V = tokens @ W_v  # (t, d_model)

    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    mask = np.triu(np.ones((tokens.shape[0], tokens.shape[0]), dtype=bool), k=1)
    weights = softmax(np.where(mask, -np.inf, scores))
    output = weights @ V
    return output[-1:]  # 只取最新這個字的輸出


def step_with_kv_cache(
    new_token: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    k_cache: np.ndarray | None,
    v_cache: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    有 KV Cache：只算「最新這 1 個字」的 q, k, v，並把 k, v 接在快取後面
    new_token shape: (1, d_model)
    """
    q_new = new_token @ W_q  # (1, d_model)
    k_new = new_token @ W_k  # (1, d_model)
    v_new = new_token @ W_v  # (1, d_model)

    # 接在原本的快取後面（如果是第一個字就直接當起點）
    k_cache = k_new if k_cache is None else np.concatenate([k_cache, k_new], axis=0)
    v_cache = v_new if v_cache is None else np.concatenate([v_cache, v_new], axis=0)

    # 只拿 (1, d_model) 的 q_new 去跟整張 k_cache (t, d_model) 算分數，完全不用 Mask
    d_k = q_new.shape[-1]
    scores = q_new @ k_cache.T / np.sqrt(d_k)  # (1, t)
    weights = softmax(scores)
    output = weights @ v_cache  # (1, d_model)
    return output, k_cache, v_cache


if __name__ == "__main__":
    np.random.seed(42)
    total_steps, d_model = 200, 64

    # 模擬權重與依序生成的 200 個 token
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    all_tokens = np.random.randn(total_steps, d_model)

    # 1. 暴力重算組
    t0 = time.perf_counter()
    out_no_cache = []
    for t in range(1, total_steps + 1):
        out = step_no_cache(all_tokens[:t], W_q, W_k, W_v)
        out_no_cache.append(out)
    time_no_cache = time.perf_counter() - t0

    # 2. KV Cache 增量組
    t0 = time.perf_counter()
    out_with_cache = []
    k_cache, v_cache = None, None
    for t in range(total_steps):
        out, k_cache, v_cache = step_with_kv_cache(
            all_tokens[t : t + 1], W_q, W_k, W_v, k_cache, v_cache
        )
        out_with_cache.append(out)
    time_with_cache = time.perf_counter() - t0

    print(f"生成 {total_steps} 個字：")
    print(f"無 Cache 耗時: {time_no_cache * 1000:.2f} ms")
    print(f"有 Cache 耗時: {time_with_cache * 1000:.2f} ms")
    print(f"加速倍率: {time_no_cache / time_with_cache:.1f}x")
    print("兩者每一步輸出是否完全一致:", np.allclose(out_no_cache, out_with_cache))
