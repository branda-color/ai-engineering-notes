# AI Engineering 學習紀錄

- **教材來源**：[ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
- **學習策略**：跳過已掌握的基礎工具與 CNN（Phase 0–4），從 Phase 7（Transformers 核心機制）補齊後，推進至 LLM 工程與 Agent 系統（Phase 11–17）。
- **練習環境**：雲端 Linux VM（Python 3.12 + `uv`，純 CPU 驗證最小實作）。

---

## 進度總覽

| 日期 | Phase | 主題 | 狀態 | 練習檔案 | 核心備註 |
|---|---|---|---|---|---|
| 2026-11-25 | Phase 7 | Transformers Deep Dive（準備開始） | 🚧 進行中 | - | 預計聚焦 Self-Attention、RoPE、KV Cache |

---

## 詳細筆記

### 2026-11-25 — 開局對齊
- 確認學習節奏：問題對齊 → 拆小片段從零手寫 → 實測與故意破壞驗證 → 對照生產環境架構。
- 第一站選定 **Phase 7: Transformers Deep Dive**，先釐清模型內部與 CNN 的核心差異。
