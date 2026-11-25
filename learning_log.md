# AI Engineering 學習紀錄

- **教材來源**：[ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
- **學習策略**：跳過已掌握的基礎工具與 CNN（Phase 0–4），從 Phase 7（Transformers 核心機制）補齊後，推進至 LLM 工程與 Agent 系統（Phase 11–17）。
- **練習環境**：雲端 Linux VM（Python 3.12 + `uv`，純 CPU 驗證最小實作）。

---

## 進度總覽

| 日期 | Phase | 主題 | 狀態 | 練習檔案 | 核心備註 |
|---|---|---|---|---|---|
| 2026-11-25 | Phase 7 | Lesson 02: Self-Attention & Causal Mask | ✅ 完成 | `phase07/01_self_attention.py` | Q/K/V、$\sqrt{d_k}$ 縮放、Numerical Stability、Causal vs Bidirectional |
| 2026-11-25 | Phase 7 | Lesson 03: Multi-Head Attention | 🚧 待開始 | - | 下一步 |

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
