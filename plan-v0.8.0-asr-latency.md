# Plan — v0.8.0 ASR latency reduction candidates

> ⏳ **Disposable planning doc.** 實作完成 + v0.8.0 commit 之後刪掉(`git rm`)。不長期留在 repo。Mirrors the v0.7.0 plan pattern.

## Why this exists

v0.7.1 shipped Windows polish 改善(PLD + prefix-cache + cuDNN benchmark)後,polish 跑得很快(長文 ~1.9s)。但測試發現**長音訊 ASR 變成新瓶頸** — 40 秒音訊要 ~10s ASR,讓使用者感受到「按完還要等 10+ 秒」。

v0.7.2 嘗試「GPU mel spectrogram」優化失敗:bench 證實 mel preprocessing 在 transformers 4.57.6 + Qwen3-ASR 路徑下只花 6ms(研究 agent 預估 1-3s,**錯了 300x**)。Tier 1A 全 revert。

**結論**:**v0.7.1 是 transformers + qwen-asr stack 的 ceiling**。要進一步降低 ASR 延遲必須換 inference stack 或裝 Flash Attention 2 — 都是 v0.8.0 等級的工程。

## v0.7.1 baseline 數據(寫死作為對照基準)

| 場景 | ASR | Polish | 總等待 | 備註 |
|------|-----|--------|--------|------|
| 短(~14 字 / ~3-5s 音訊)| ~0.5-1s | 0.25s | < 2s | 日常常用,無瓶頸 |
| 中(~50-100 字 / ~10-15s 音訊)| ~2-3s | 0.7-1.0s | ~3-5s | 仍快 |
| 長(~200 字 / ~30s 音訊)| ~7-8s | ~3-4s | ~10-13s | 慢但堪用 |
| 超長(~280 字 / ~40s 音訊)| **~10s** | ~5s | **~15s** | stress test,長段 hold-to-talk |

**Polish 在 v0.7.1 已 metal-efficient**:byte-identical 4B output、decode -55% vs v0.7.0 4B baseline。不要再優化 polish。

**Mac path 跟 v0.7.0 完全等價**(MlxLocalLlmPolisher 未動)— 不在本 plan 範圍。

## Real bottleneck(實測拆解)

40s 音訊 / 10s ASR 內部:
- Mel preprocessing: ~10ms(已驗證)
- **Encoder forward**(40s mel → hidden states):~3-5s on GPU ← attack vector
- **Decoder autoregressive generate**(~300 tokens):~5-7s on GPU ← attack vector

兩個都已在 GPU 上跑,但 transformers 的 attention impl(sdpa)跟 Python decode loop overhead 限制了天花板。

## Candidates(由 ROI 排序)

### A. Flash Attention 2(compile from source)

**What**:Qwen team 官方推薦 Qwen3-ASR 用 `attn_implementation="flash_attention_2"`。Encoder + decoder 都受益。

**Expected**:1.3-1.6x ASR(10s → 6-8s),quality lossless

**Cost**:
- 需 Visual Studio Build Tools 2022 + MSVC + CUDA Toolkit 12.8(不是 runtime,是 nvcc compiler)
- Compile 30-60 分鐘
- 配環境 + debug 可能再 30 分鐘
- 完成後改 `_Qwen3TorchImpl.from_pretrained` 加一行 kwarg(同時改 `TorchLocalLlmPolisher` 也能受益)

**Risk**:
- 中-高 — compile from source 在 Windows 上常碰 MSVC vs CUDA 版本不對、header path 問題
- 即使 compile 成功,Blackwell sm_120 的 kernel coverage 可能不完整(已知 issue #1683)
- 若 daemon 啟動失敗會 fall back 到 NoopPolisher / faster-whisper,但 debug 路徑長

**Decision criteria**:
- 先確認本機有 VS 2022 Build Tools + CUDA Toolkit 12.8 nvcc 可用
- 若沒,先裝環境(各 ~2 GB),總 timeline 1-2 小時
- 若有,先試 compile —— 30 分鐘內成功 → ship;失敗 → 跳到候選 B

**Code change pointer**:
- `scripts/stt-daemon.py:615` `Qwen3ASRModel.from_pretrained(...)` — 加 `attn_implementation="flash_attention_2"`
- `scripts/text_polisher.py:_PREFERRED_ATTN` framework 已存在(v0.7.0 加的),裝完 flash-attn 後 polish 也會自動切換

### B. 換 inference stack 到 llama.cpp + Qwen3-ASR-0.6B-GGUF Q8_0

**What**:Alibaba 官方 ship `ggml-org/Qwen3-ASR-0.6B-GGUF` Q8_0(805 MB)。用 llama-cpp-python wrap llama-server HTTP endpoint。

**Expected**:2-3x ASR(10s → 3-5s),Q8 ≈ bf16 for ASR per ASR paper

**Cost**:4-8 hours
- 新依賴 `llama-cpp-python[cuda]`(Windows + CUDA 已成熟)
- 下載 GGUF model ~805 MB
- 寫 `LlamaCppQwen3AsrBackend` class 實作 STTBackend interface
- 整合 daemon 的 backend dispatch
- A/B bench 20 個 zh/zh-en 樣本驗證 quality 沒退化

**Risk**:中
- llama.cpp 跟 transformers 的 audio preprocessing 可能略有不同 → 確認輸出對齊
- 啟動時間多一個 server process(或 in-process)→ daemon 啟動流程改變
- 失敗時 fallback 機制要設計

**Decision criteria**:
- 若 A(FA2 compile)失敗或 compile 環境設定太麻煩 → 走 B
- 若使用者需要 long-form ASR 大幅加速(2x+)→ 走 B
- 若只要 1.3-1.6x 加速,A 較輕量

**Code change pointer**:
- 新檔 `scripts/llama_cpp_qwen3_asr.py` 或新 class 在 `stt-daemon.py:_Qwen3TorchImpl` 旁邊
- `Qwen3AsrBackend.__init__` 加 `_LlamaCppImpl` 跟 `_Qwen3TorchImpl` 並列
- 加 config toggle 讓 user 切換

### C. Streaming ASR via vLLM + WSL2(本 plan 不深入,只交叉參考)

**Status**:earlier research established this is v0.8.0+ feasible but daemon 需要 cross-WSL2 IPC layer。中文場景品質損失只 +0.52 WER(小)。

**Cost**:12-20 hours total(vLLM setup + WSL2 networking + daemon refactor)

**Expected**:長文 wall-clock 體驗大幅改善(等放開按鍵後只需處理 tail chunk)

**Decision criteria**:
- 只在 A 跟 B 都不夠用、且 user 高頻 long-form 使用時才考慮
- 不建議當 v0.8.0 首選 — 工程量 / 風險最高

## Already ruled out — don't re-research

| 路徑 | 為何 ruled out | 來源 |
|------|---------------|------|
| **flash-attn pre-built Windows wheel for torch 2.11 + cu128 + py 3.12** | 沒有對齊 stack 的 wheel 存在,ABI 嚴格 | 2026-05-24 multi-source verify(marcorez8 HF / White2Hand HF / loscrossos GitHub) |
| **GPU mel spectrogram patch (Tier 1A)** | bench 實測只省 3ms,placebo level | 本機 bench-mel.py(已刪)|
| **bitsandbytes NF4 INT4 (polish 或 ASR)** | Blackwell sm_120 上 67% 慢,bnb issue #1851 已 open 無 maintainer 回應 | v0.7.0 實測 + bitsandbytes#1851 |
| **torch.compile on Windows without triton** | Inductor 退回 no-op,triton 在 Windows 無官方 wheel(triton-windows fork archived 2026-02-18)| v0.7.0 stage 2 實測 |
| **faster-whisper INT8 on RTX 5080** | Blackwell sm_120 CUBLAS_STATUS_NOT_SUPPORTED crash,只能用 fp16(速度退一半)| SubtitleEdit issue #10180 |
| **distil-whisper / NVIDIA Canary / NVIDIA Parakeet** | 英文 only / 無中文支援 | 各 model card |
| **TensorRT-LLM Windows native** | RTX 50 系列 Whisper 0.17 直接 crash,Blackwell 支援未到 | NVIDIA/TensorRT-LLM#2847 |
| **CTranslate2 Qwen3-ASR support** | CTranslate2 沒 Qwen3-ASR arch | 2026 community check |
| **ExllamaV2 / EXL2 quants** | 對 0.6B 小模型不是 sweet spot,跟 vLLM-AWQ-Marlin 差距小 | 2026 community |
| **Speculative decoding with Qwen3-0.6B as draft for Qwen3-4B-ASR** | acceptance 太低,net negative;AWS benchmark verified | aws/spec-decode blog |
| **Hybrid fast-preview + slow refine(clipboard 替換)** | Windows clipboard 一旦 user 開始編輯就無法 retroactively replace | UX 共識 |
| **PLD on ASR decoder** | ASR input 是 audio,沒 text 可 lookup;前 transcript 不算 stable lookup source | 概念分析 |
| **Prefix-cache on ASR** | 每段 audio context 不同,沒 stable prefix | 概念分析 |
| **Switch to Qwen3-ASR-1.7B** | 1.7B > 0.6B,反而更慢 | model card |
| **Switch to Qwen3-ASR-0.6B-MLX-4bit** | Mac only | mlx-qwen3-asr |
| **Official FP8 Qwen3-ASR checkpoint** | Alibaba 沒 ship,不像 Qwen3-4B-FP8(polish 用)有官方 ship | HF model tree 2026-05 |

## Investigation order if you re-engage

1. **檢查環境 readiness for FA2 compile**:
   ```powershell
   where cl.exe           # MSVC compiler in PATH?
   where nvcc.exe         # CUDA Toolkit compiler in PATH?
   nvcc --version         # 12.8?
   ```
   - 都有 → 試 A(成本最低、quality 最穩、可逆)
   - 沒 → 跳 B(避免裝 Build Tools 的 ~3 GB)

2. **試 A**:
   ```powershell
   pip install ninja                          # speed up compile
   pip install flash-attn --no-build-isolation
   ```
   - 30-60 min 後若成功 → import test → 加 kwarg → bench → ship v0.8.0
   - 失敗(常見原因:CUDA 版本不對、MSVC 太舊、torch headers 衝突)→ 跳 B

3. **試 B**(在 A 失敗或 user 要 2x+ 改善時):
   - 詳細 plan 在實作前先寫(類似本 plan 的格式)
   - 不要直接動手 — 換 backend 涉及 daemon dispatch / fallback / config / 啟動流程

## Bench reproduction(再次驗證 v0.7.1 baseline 用)

需要時可重建這個快速 bench(類似 v0.7.1 開發時的 `bench-v071.py`):

```python
import importlib.util, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("stt_daemon", "scripts/stt-daemon.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from text_polisher import build_polisher

p = build_polisher(True, mod.POLISH_MODEL, mod.POLISH_PROMPT)
# Warmup, then time 3 runs each of short/medium/long
# Reference: v0.7.1 long-mixed (99 char) = 1.90s
```

對 ASR 的 bench 需要實際音訊檔(或用 daemon log 的 `zh X.XXs+polish Y.YYs` 觀察)。

## When to re-open this plan

- User 開始頻繁 long-form 使用(會議轉錄、口述長文)→ 投入 A 或 B
- 上游 PyTorch / qwen-asr / flash-attn 有 Blackwell 對齊 release → 重評 A 成本
- llama.cpp / vLLM 在 Windows native 對 ASR 模型成熟度提升 → 重評 B
- 否則,**v0.7.1 already shipped 是合理 ceiling**,本 plan 可一直留著等

## File locations for reference

- ASR backend impl:`scripts/stt-daemon.py:579-642` (`_Qwen3TorchImpl`)
- Polish backend impl:`scripts/text_polisher.py:153-410` (`TorchLocalLlmPolisher`)
- Backend factory:`scripts/stt-daemon.py:645-660` (`build_backend`)
- Polish factory:`scripts/text_polisher.py:412-490` (`build_polisher`)
- Config defaults:`scripts/stt-daemon.py:74-145`
- Reference v0.7.1 commit:`8bb6157`
