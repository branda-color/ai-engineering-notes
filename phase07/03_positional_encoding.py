import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def attention_output(X: np.ndarray) -> np.ndarray:
    d_k = X.shape[-1]
    scores = X @ X.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ X


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    產生 (seq_len, d_model) 的位置號碼牌矩陣
    每個位置 pos 用不同頻率的 sin / cos 波組合成唯一指紋
    """
    pos = np.arange(seq_len)[:, np.newaxis]        # (seq_len, 1)
    dim = np.arange(0, d_model, 2)[np.newaxis, :]  # (1, d_model // 2)
    angles = pos / np.power(10000.0, dim / d_model)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)  # 偶數欄放 sin
    pe[:, 1::2] = np.cos(angles)  # 奇數欄放 cos
    return pe


if __name__ == "__main__":
    np.random.seed(7)
    d_model = 8
    dog = np.random.randn(d_model)
    bite = np.random.randn(d_model)
    human = np.random.randn(d_model)

    # 兩句話字完全一樣，只有順序顛倒
    sent_a = np.stack([dog, bite, human])  # ["狗", "咬", "人"]
    sent_b = np.stack([human, bite, dog])  # ["人", "咬", "狗"]

    # 1. 不加位置編碼：抓出中間 index=1 的 "咬" 來比對
    out_a_no_pe = attention_output(sent_a)[1]
    out_b_no_pe = attention_output(sent_b)[1]
    print("不加位置編碼，兩句話的 '咬' 是否完全一樣:", np.allclose(out_a_no_pe, out_b_no_pe))

    # 2. 蓋上位置號碼牌（直接相加）
    pe = sinusoidal_positional_encoding(seq_len=3, d_model=d_model)
    out_a_with_pe = attention_output(sent_a + pe)[1]
    out_b_with_pe = attention_output(sent_b + pe)[1]
    print("加上位置編碼後，兩句話的 '咬' 是否完全一樣:", np.allclose(out_a_with_pe, out_b_with_pe))
