# AI Engineering 學習紀錄

- **教材來源**：[ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
- **學習策略**：跳過已掌握的基礎工具與 CNN（Phase 0–4），從 Phase 7（Transformers 核心機制）補齊後，推進至 LLM 工程與 Agent 系統（Phase 11–17）。
- **練習環境**：雲端 Linux VM（Python 3.12 + `uv`，純 CPU 驗證最小實作）。

---

## 進度總覽

| 日期 | Phase | 主題 | 狀態 | 練習檔案 | 核心備註 |
|---|---|---|---|---|---|
| 2026-11-25 | Phase 7 | Lesson 02: Self-Attention & Causal Mask | ✅ 完成 | `phase07/01_self_attention.py` | Q/K/V、$\sqrt{d_k}$ 縮放、Numerical Stability、Causal vs Bidirectional |
| 2026-11-25 | Phase 7 | Lesson 03: Multi-Head Attention | ✅ 完成 | `phase07/02_multi_head_attention.py` | `reshape` + `transpose` 切頭、多頭平行捕捉不同語意關聯 |
| 2026-11-25 | Phase 7 | Lesson 04: Positional Encoding & RoPE | ✅ 完成 | `phase07/03_positional_encoding.py` | 排列不變性驗證、Sinusoidal vs RoPE 相對距離、與 Embedding 流程關係 |
| 2026-11-25 | Phase 7 | Lesson 12: KV Cache | ✅ 完成 | `phase07/04_kv_cache.py` | Prefill vs Decode 瓶頸、增量快取實測 19.7x 加速、PagedAttention / Prefix Caching 概念 |
| 待續 | Phase 11 | LLM Engineering (RAG & Eval) | 🚧 待開始 | - | 下一站主戰場（建議開新對話視窗銜接） |

---

## 詳細筆記

### 2026-11-25 — Phase 7: Self-Attention 核心與 Causal Mask
- **核心引擎**：`Q @ K.T / sqrt(d_k)` → `softmax` → `@ V`，單層即可讓序列中每個 token 直接計算兩兩關聯，複雜度為 $O(N^2)$。
- **工程細節**：
  - 除以 $\sqrt{d_k}$ 避免點積變異數過大導致 softmax 梯度飽和。
  - `softmax` 前先扣掉每列最大值防止 `exp` 溢位。
  - Causal Mask 必須在進 `softmax` 前對上三角填入 `-inf`（填 `0` 會使 `exp(0)=1` 而失效）。
- **模型路線差異**：
  - **GPT 類型（Decoder / Causal）**：逐字生成任務，禁止看到未來 token（開 Causal Mask）。
  - **BERT 類型（Encoder / Bidirectional）**：語意理解與 Embedding 任務，需雙向參考全文（不開 Causal Mask）。

### 2026-11-25 — Phase 7: Multi-Head Attention
- **核心概念**：把每個 token 的特徵維度對半（或多等分）拆成多個 Head，讓不同 Head 用不同視角平行計算 Attention（例如有的看上下文動詞、有的看字面本身），最後再無損拼回原本維度。
- **維度轉換關鍵**：
  - 必須先 `reshape(seq_len, num_heads, d_k)` 再 `transpose(1, 0, 2)` 把 `num_heads` 拉到最前面，不能直接 `reshape(num_heads, seq_len, d_k)`，否則會把句子劈斷。
  - 3 維張量轉置需用 `swapaxes(-1, -2)`，只對調每個頭內部的列與欄，不動到最外層的頭數維度。

### 2026-11-25 — Phase 7: Positional Encoding & RoPE
- **為什麼需要**：Self-Attention 具備排列不變性（Permutation Invariant），沒加位置訊號時「狗咬人」與「人咬狗」算出的輸出完全一樣。
- **演進路線**：
  - 早期 **Sinusoidal**：用 `sin / cos` 產生 `[-1, 1]` 區間的絕對位置號碼牌，直接加在輸入向量上。
  - 現代 **RoPE**：在 Attention 階段對 `Q`、`K` 做角度旋轉，使 `Q @ K.T` 內積結果只取決於兩者相對距離（如 `rope_theta` 決定長文本外推）。
- **與 Embedding 的關係**：Token Embedding（查字義）→ Position/RoPE（揉入順序）→ 穿過多層 Transformer → 最終輸出句意 Embedding（RAG 向量搜尋所用）。

### 2026-11-25 — Phase 7: KV Cache 與推論系統架構
- **核心原理**：因 Causal Mask 保證過去的字不受未來影響，已生成的 `K`、`V` 向量不會再變動；推論時只算最新 1 個字的 `q_new, k_new, v_new`，將 `k_new, v_new` 接在既有快取後即可（連 Mask 都能省掉，200 步實測加速 19.7x 且結果完全等價）。
- **推論兩階段瓶頸**：
  - **Prefill**：一次算完輸入 Prompt 並填滿初始 KV Cache，決定 TTFT（首字延遲），吃算力。
  - **Decode**：逐字生成，每步都要讀取整份 KV Cache，決定吐字速度，吃爆 GPU VRAM 容量與記憶體頻寬。
- **生產級記憶體管理**：vLLM 等引擎採用類似 OS 分頁與 Memory Pool 的 **PagedAttention** 管理 KV Block；開啟 **Prefix Caching** 時透過 Hash Table 與 LRU 淘汰策略，跨 Request 復用共用的 System Prompt 快取。
