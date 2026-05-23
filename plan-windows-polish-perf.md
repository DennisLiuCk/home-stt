# Plan — v0.7.0 Windows polish latency optimization

> ⏳ **Disposable planning doc.** 實作完成 + v0.7.0 commit 之後 **刪掉這份檔案**。
> 不要長期留在 repo,不歸 README 管理範圍。

## 目標一句話

讓 Windows v0.7.0 polish 階段在長文情境下的延遲**降到接近 Mac MLX 等級**(~0.5-1.0s for 100-char output),解掉現在「RTX 5080 跑得比 Apple Silicon MLX 慢」的反直覺體驗。

## 為什麼會反直覺(v0.6.0 觀察基礎)

長文(~20 秒語音 → ~200 字輸出)實測延遲:

| 平台 | 硬體 | ASR | Polish | 總計 |
|------|------|-----|--------|------|
| Mac M-series | unified memory 16+ GB,~400 GB/s | 2.17 s | 1.24 s | **3.41 s** |
| Windows RTX 5080 | 16 GB VRAM,~960 GB/s | (TODO 實測) | (TODO 實測 ≥ 3 s based on user feedback) | ≥ 5-6 s |

RTX 5080 raw FLOPS 跟 memory bandwidth 都比 M-series 高 2-3 倍。理論上 polish 該快 — 但實際相反。根因:

1. **`transformers.generate()` Python overhead** — 每次 call 約 10-50 ms Python/autograd context 開銷。短輸出 (~100 tokens) 時佔比 20-40%。
2. **Eager attention 預設** — `AutoModelForCausalLM.from_pretrained(...)` 不指定 `attn_implementation` 預設用 eager。對長 sequence decode 是 O(n²) 不必要慢。
3. **bf16 沒利用 RTX 的 INT8/INT4 tensor cores** — 1.5B bf16 = 3 GB / step,4-bit 量化能砍到 ~750 MB / step,memory-bandwidth bound 的 small-model decode 直接受益。
4. **Mac 那邊 MLX 是 Apple 自寫 inference engine** — Metal kernels 直接調用,Python 層接近透明。
5. **Mac 用 4-bit MLX 量化**(Qwen3-4B-Instruct-2507-MLX-4bit ~2 GB 權重),Windows 用 bf16(Qwen2.5-1.5B-Instruct ~3 GB) — Mac 每 token 移動的 memory **更少**,跟 bandwidth 比值反而有利。

簡言之:**small-model autoregressive decode 是 memory-bandwidth-bound + Python-overhead-bound,不是 FLOPS-bound**。RTX 5080 算力用不到。

## 不會動到的部分(放心)

- ASR backend (`Qwen3AsrBackend` + `_Qwen3TorchImpl`) 已在 v0.6.0 對齊
- `TextPostProcessor` ABC + `MlxLocalLlmPolisher`(Mac 路徑零變動)
- Pasteboard + listener + paste pipeline
- `NoopPolisher` fallback 行為
- Config 結構(POLISH_ENABLED / POLISH_MODEL / POLISH_LANGUAGES / POLISH_PROMPT)

只動 `TorchLocalLlmPolisher` 內部的 model load + generate 路徑,跟必要時的 install 步驟。

---

## 優化選項清單(由便宜到貴排序)

每一階段都做完 benchmark,再決定要不要進下一階段。預期單獨某一階段可能就夠。

### 階段 1:`attn_implementation="sdpa"`(scaled dot-product attention)

PyTorch 2.0+ 內建,**不必裝任何套件**。預期 30-50% decode 改善。

```python
self._model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",  # ← 加這行
)
```

風險:模型若未支援 sdpa 會 fall back 到 eager(transformers warning,不會 crash)。Qwen2.5/Qwen3 系列都已支援。

### 階段 2:`torch.compile(model)`

把整個 forward graph JIT-compile。預期再 20-30% 改善。

```python
self._model = torch.compile(self._model, mode="reduce-overhead")
```

風險:
- 首次推論 warmup ~30-60 秒(編譯 graph + autotune kernels)— 啟動慢但跑得快
- 某些模型操作不支援 `reduce-overhead` mode,要 fall back `"default"` 或 `"max-autotune"`
- transformers + torch.compile 在 generate() 路徑歷史上有 bug,需要 PyTorch ≥ 2.4。先確認版本

### 階段 3:Flash Attention 2

```powershell
# 需要 CUDA toolkit (不只是 runtime) 來編譯
# Pre-built wheel 通常涵蓋 CUDA 11.8 / 12.1 / 12.4 + Python 3.10/3.11/3.12
pip install flash-attn --no-build-isolation
```

```python
self._model = AutoModelForCausalLM.from_pretrained(
    ...
    attn_implementation="flash_attention_2",  # 取代 sdpa
)
```

預期再 20-40% 改善(尤其長 context)。階段 1 + 階段 3 加總接近原始 50-100% 加速。

風險:
- pip install flash-attn 在 Windows 沒 prebuilt wheel 時要 from source build,需要 CUDA toolkit + MSVC。**可能 30-60 分鐘編譯時間**或直接失敗
- 部分 GPU 架構不支援(Ampere / Ada Lovelace / Hopper 都行;RTX 5080 是 Blackwell,需確認支援 — 應該有)
- 如果 flash-attn 裝不上,**保留階段 1 + 2 即可**,不必執著

### 階段 4:bitsandbytes INT4 量化

```powershell
pip install bitsandbytes
```

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",  # nf4 比 fp4 品質好
)
self._model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda:0",
    attn_implementation="sdpa",  # 或 flash_attention_2
)
```

預期再 20-40% 改善 + VRAM 砍掉 70%(1.5B bf16 ~3GB → INT4 ~750 MB)。

風險:
- bitsandbytes 在 Windows 歷史上 flaky,需要對的 CUDA 版本對應 wheel
- 量化可能讓 polish 品質微降。實測比較 raw 跟 polished 輸出
- 跟 torch.compile 組合不一定相容,先一個一個來

### 階段 5(可選):換更小 polish 模型

如果上面組合 1+2+3+4 都跑了還是慢,降到 `Qwen/Qwen2.5-0.5B-Instruct`(~1 GB bf16)或 `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` 等更激進的選項。

預期會多一些品質損失,但延遲應顯著改善。**這是最後手段,前面組合做完應該已經夠用**。

### 階段 6(若都跑完仍不滿意):vLLM 重寫

```powershell
pip install vllm
```

vLLM 是專業 LLM inference engine,有 PagedAttention + continuous batching + tensor parallelism。預期 50-100% 額外改善。

風險:
- 需要重寫 `TorchLocalLlmPolisher` — 不再用 transformers.AutoModelForCausalLM,改用 `vllm.LLM` API
- vLLM Windows native 支援還不穩(主要在 Linux),可能要走 WSL2
- 啟動時間比 transformers 長,memory footprint 大

**只在前面 5 階段都做完仍不滿意才走這條**。

---

## 改動清單

### 改動 1:`scripts/text_polisher.py` — `TorchLocalLlmPolisher.__init__`

把 model load 路徑改成可配置 attention + 可選量化。建議用 module-level constants 而不是塞進 Config block,讓主要 daemon Config 保持乾淨。

```python
class TorchLocalLlmPolisher(TextPostProcessor):
    # Attention implementation. Probed in this order:
    #   "flash_attention_2" — fastest if flash-attn is installed
    #   "sdpa"              — PyTorch 2.0+ built-in, no extra dep
    #   "eager"             — last resort, slow on long sequences
    _PREFERRED_ATTN = ("flash_attention_2", "sdpa", "eager")

    # 4-bit NF4 quantization via bitsandbytes. Set to True after
    # `pip install bitsandbytes` is confirmed working — cuts VRAM ~75%
    # and decode time ~20-40% on small models.
    _USE_4BIT_QUANT = False  # toggle per machine

    # torch.compile the model after load. Adds ~30-60s startup warmup
    # but gives ~20-30% decode improvement once warm.
    _USE_TORCH_COMPILE = False  # toggle per machine

    def __init__(self, model_name, system_prompt, max_tokens=256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TorchLocalLlmPolisher requires CUDA. ..."
            )

        # Choose best available attention impl
        from transformers.utils import is_flash_attn_2_available
        attn = "eager"
        for candidate in self._PREFERRED_ATTN:
            if candidate == "flash_attention_2" and not is_flash_attn_2_available():
                continue
            attn = candidate
            break

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation=attn,
        )
        if self._USE_4BIT_QUANT:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs.pop("torch_dtype")  # quant config overrides

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs,
        )
        self._model.eval()

        if self._USE_TORCH_COMPILE:
            # mode="reduce-overhead" for decode workloads;
            # falls back gracefully if unsupported.
            self._model = torch.compile(self._model, mode="reduce-overhead")

        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._model_name = model_name

        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA device"
        quant_label = "INT4-NF4" if self._USE_4BIT_QUANT else "bfloat16"
        compile_label = " + torch.compile" if self._USE_TORCH_COMPILE else ""
        self.device_label = (
            f"{model_name} (PyTorch {quant_label} + {attn}{compile_label} "
            f"@ NVIDIA {gpu_name}, ≤{max_tokens} tok)"
        )
```

### 改動 2:`__version__` bump

```python
__version__ = "0.6.0"  → "0.7.0"
```

### 改動 3:README updates(改動較少這次)

1. **Badge** v0.6.0 → v0.7.0
2. **Windows GPU 加速套件表**:加 `flash-attn (optional)` + `bitsandbytes (optional)` 兩個可選項目,標清楚「不必裝,但裝了能加速 polish」
3. **Windows 一鍵安裝**:加一個「v0.7.0 進階加速(選用)」小節,描述三個 toggle:
   ```powershell
   # 進階:安裝 flash-attn 加速 polish(從 source 編譯,~30-60 分鐘)
   pip install flash-attn --no-build-isolation
   # 進階:安裝 bitsandbytes 4-bit 量化
   pip install bitsandbytes
   ```
4. **疑難排解**:加一段 "Polish 在 Windows 比 macOS 慢" 的條目,引到 v0.7.0 toggle

---

## Windows 端實作步驟順序

```powershell
# 0. fetch 最新
cd C:\path\to\home-stt
git fetch origin
git pull origin main

# 1. 讀這份計劃
notepad plan-windows-polish-perf.md

# 2. Baseline 量測 — 在不改任何東西的情況下 bench 一輪
#    (建議寫個 scripts/bench-polish.ps1 之類的,測 10 個樣本平均延遲)
python -c "
import time, sys
sys.path.insert(0, 'scripts')
from text_polisher import build_polisher
prompt = '...'  # 從 stt-daemon.py 複製 POLISH_PROMPT
p = build_polisher(True, 'Qwen/Qwen2.5-1.5B-Instruct', prompt)
samples = ['短文','...','長文(200字)']
for s in samples:
    t0 = time.time()
    out = p.polish(s)
    print(f'{time.time()-t0:.2f}s | len(in)={len(s)} len(out)={len(out)}')
"
# 記下 baseline 數字,例如 '長文 baseline = 3.2s'

# 3. 階段 1:加 attn_implementation="sdpa"
#    - 改 scripts/text_polisher.py 套用「改動 1」(只開 sdpa,4bit + compile 都 False)
#    - 重跑 bench,記下「+sdpa = X.Xs」

# 4. 階段 2:加 torch.compile
#    - _USE_TORCH_COMPILE = True
#    - 首次啟動會慢(編譯 graph),記下 warmup time
#    - 重跑 bench,記下「+compile = X.Xs」

# 5. 階段 3:安裝 flash-attn
pip install flash-attn --no-build-isolation
#    - text_polisher.py 的 _PREFERRED_ATTN 已經會自動偵測 flash_attention_2
#    - 重跑 bench,記下「+flash-attn = X.Xs」

# 6. 階段 4(可選):bitsandbytes 4-bit
pip install bitsandbytes
#    - _USE_4BIT_QUANT = True
#    - 注意:可能跟 torch.compile 不相容,組合測試
#    - 重跑 bench + 確認 polish 品質沒明顯退化

# 7. 確定 baseline 數字達標(長文 < 1.5s)後:
#    - __version__ → "0.7.0"
#    - README 改 (改動 3)
#    - 刪本檔
#    - commit + tag + push

# 8. (可選)階段 5/6 — 換小模型 / vLLM,只在前面都跑完仍不滿意時做
```

---

## 驗證 checklist

每階段都做這個流程,記在 bench 表格裡:

### Benchmark protocol

固定 5 個樣本(短/中/長 各覆蓋),每個跑 3 次取平均:

| 樣本長度 | 樣本內容(例) |
|---------|--------------|
| 短(20-30 字) | `那個我們等等就是說要去吃飯吧` |
| 中(60-80 字) | `呃我覺得這個 Python function 的設計可以再優化一下,然後就是說標點的部分也想要 polish 一下` |
| 長(150-200 字) | `整體來說我覺得長文字的處理在 Mac 上的體驗會比在 Windows 上感覺更快...`(完整版用 v0.6.0 那條長文 sample) |

### Benchmark 表格(實作時填)

| 階段 | 配置 | 短文 polish | 中文 polish | 長文 polish | warmup time | VRAM peak | 品質感受 |
|------|------|-----------|------------|------------|------------|-----------|---------|
| 0 | baseline (v0.6.0,eager attention) | ? s | ? s | ? s | ? s | ? GB | OK |
| 1 | +sdpa | ? | ? | ? | ? | ? | OK |
| 2 | +sdpa +compile | ? | ? | ? | ? | ? | OK |
| 3 | +flash-attn +compile | ? | ? | ? | ? | ? | OK |
| 4 | +flash-attn +compile +4bit | ? | ? | ? | ? | ? | 確認沒退步 |

**目標**:長文 polish < 1.5s,理想 < 1.0s。

### 達標後再做

- [ ] 整套 daemon 重啟,看完整啟動 log 是否乾淨(沒新 warning)
- [ ] tail log 跑 3-5 段真實 hold-to-talk,確認 `[stt] zh X.XXs+polish Y.YYs` 跟 bench 數字吻合
- [ ] 比較 polish 輸出品質沒退化(同樣的 raw → 同樣的 polished),特別是 4-bit 量化開啟後
- [ ] Mac 端也要再 sanity check 一次(`bash scripts/stt-start.sh`,確認 MLX 路徑零退步)— 別忘了

---

## 風險 + 已知問題

1. **`flash-attn` 在 Windows pip install 可能要從 source 編譯** — 需要 CUDA toolkit + MSVC。如果你 PC 沒裝這些,先看 PyPI 有沒有 prebuilt wheel(`pip install flash-attn==X.X.X` 試特定版本)。實在裝不上就**跳過階段 3**,光階段 1+2 就有顯著改善。

2. **`torch.compile` 跟 transformers `.generate()` 的歷史 bug** — 需要 PyTorch ≥ 2.4。如果 compile 後出現 generation 結果空 / 亂碼,降回 `_USE_TORCH_COMPILE = False` 並去 PyTorch GitHub issue 追蹤。

3. **`bitsandbytes` 4-bit 量化可能輕微降低 polish 品質** — 對 1.5B 小模型影響應該 minor,但要實測比對。如果品質明顯退化,降回 bf16。

4. **RTX 5080 是 Blackwell 架構,某些優化 library 可能 lag** — flash-attn / bitsandbytes 對 Blackwell 的支援可能比 Ada/Hopper 晚到。看到 unsupported architecture 錯誤先去 GitHub 看 issue。

5. **改完 Mac 端要 sanity 確認** — `text_polisher.py` 的改動是 PyTorch path(`TorchLocalLlmPolisher`),理論上 Mac 端(`MlxLocalLlmPolisher`)零影響,但因為都在同個檔案,要重啟 Mac daemon 一次確認沒誤改到 MLX 路徑。

6. **warmup 時間變長**:torch.compile 開啟後首次啟動會多 30-60 秒。對 daemon use case(啟動一次跑很久)還可接受,但要在 README / 啟動訊息提示使用者。

---

## 完成後清理(別忘了)

1. `git rm plan-windows-polish-perf.md`
2. `git add scripts/text_polisher.py scripts/stt-daemon.py README.md`
3. `git commit -m "v0.7.0: Windows polish latency optimization (sdpa / flash-attn / torch.compile / int4)"`
4. `git tag -a v0.7.0 -m "Release v0.7.0 — Windows polish latency optimization"`
5. `git push origin main --tags`

如果實作過程發現本計劃哪一階段不可行,**直接改本計劃 commit 一版說明跳過原因,再進下一階段** — git history 對未來的你(或外部 contributor)是最好的決策紀錄。
