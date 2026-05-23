# Plan — v0.7.0 Windows polish latency optimization

> ⏳ **Disposable planning doc.** 實作完成 + v0.7.0 commit 之後 **刪掉這份檔案**。
> 不要長期留在 repo,不歸 README 管理範圍。

> 📌 **Plan baseline**:本計劃假設你已在 commit `1151809` 之後(8 個 LOW
> code-review fixes 已 ship)的 `text_polisher.py` 上做增量修改,不是從
> v0.6.0 base (ad02681) 重寫。1151809 對 `TorchLocalLlmPolisher` 加了:
> `self._device` literal extraction、`self._pad_token_id` + `_resolve_pad_token_id()`
> helper、polish() 的 `add_special_tokens=False`、`dtype=` 取代 deprecated
> `torch_dtype=`。**這些都要保留**,本 plan 只在它們之上加 attention impl
> 偵測 + 可選 4-bit 量化 + 可選 torch.compile。

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

在現有的 `from_pretrained` 呼叫(已用 `dtype=` 不是 `torch_dtype=`)加一行:

```python
self._model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,                  # 已在 1151809 改為 dtype
    device_map=self._device,               # 已在 1151809 抽成 self._device
    attn_implementation="sdpa",            # ← 階段 1 新增
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

**Pre-flight 必做** — Blackwell (sm_120) prebuilt wheel 可能還沒釋出:

```powershell
# 1. 先確認 flash-attn 有沒有 Blackwell prebuilt wheel
#    去 https://github.com/Dao-AILab/flash-attention/releases 看最新 release
#    或直接試裝 (失敗會立即知道,不會浪費編譯時間)
pip install flash-attn --no-build-isolation --no-deps
```

如果有 prebuilt wheel → 一行裝完。如果沒有 → 要 CUDA toolkit + MSVC + ~30-60 分鐘編譯時間或直接失敗;**建議**遇到 source build 就跳過階段 3,光階段 1 + 2 已有顯著改善。

```python
self._model = AutoModelForCausalLM.from_pretrained(
    ...
    attn_implementation="flash_attention_2",  # 取代 sdpa
)
```

預期再 20-40% 改善(尤其長 context)。階段 1 + 階段 3 加總接近原始 50-100% 加速。

風險:
- **`is_flash_attn_2_available`** 的 import 路徑跨 transformers 版本可能不穩。Plan 改動 1 範例用 try/except 動態 import,別硬寫死路徑
- pip install flash-attn 在 Windows 沒 prebuilt wheel 時要 from source build,需要 CUDA toolkit + MSVC
- 部分 GPU 架構不支援。Ampere / Ada Lovelace / Hopper 都 OK;**RTX 5080 是 Blackwell (sm_120),flash-attn 2.x stable 截 2026 H1 應該已支援,但 prebuilt wheel 釋出時間可能晚於 PyTorch 主流支援**
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
- bitsandbytes Windows native wheel 在 2025 H2 後已較穩定(早期歷史很 flaky,現在 OK)。挑跟你 CUDA 版本對應的 wheel — 我們現在跑 cu128 torch 2.11,bitsandbytes ≥ 0.43 應有支援
- 量化可能讓 polish 品質微降。實測比較 raw 跟 polished 輸出(bench 表的「polish vs baseline 字數差」欄)
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

### 改動 1:`scripts/text_polisher.py` — `TorchLocalLlmPolisher.__init__`(增量)

**重要**:這是在 commit `1151809` 後的 `TorchLocalLlmPolisher` **之上** 加東西,不是全文重寫。
要保留的 1151809 加的東西:`self._device`、`self._pad_token_id` + `_resolve_pad_token_id()`、
polish() 的 `add_special_tokens=False`、`dtype=` (非 `torch_dtype=`)。

**加 3 個 class-level 常數**(放在 class TorchLocalLlmPolisher 開頭,在 docstring 之後):

```python
class TorchLocalLlmPolisher(TextPostProcessor):
    """PyTorch + transformers + NVIDIA CUDA polisher for Windows / Linux.
    (existing docstring stays)"""

    # v0.7.0: attention impl preference order. Auto-falls back if a higher
    # candidate isn't available. flash_attention_2 needs `pip install
    # flash-attn`; sdpa is PyTorch 2.0+ built-in; eager is the last-resort
    # legacy path.
    _PREFERRED_ATTN = ("flash_attention_2", "sdpa", "eager")

    # v0.7.0: 4-bit NF4 quantization via bitsandbytes. Toggle True after
    # `pip install bitsandbytes` is confirmed working. Cuts VRAM ~75% and
    # decode time ~20-40% on small models; may microscopically degrade
    # polish quality (validate via bench table).
    _USE_4BIT_QUANT = False

    # v0.7.0: torch.compile the model after load. Adds ~30-60s startup
    # warmup (graph capture + autotune) but ~20-30% decode improvement
    # once warm. Set False if generate() emits empty/garbage output
    # (known transformers + compile interaction bugs in older PyTorch).
    _USE_TORCH_COMPILE = False
```

**改 `__init__`**:在 `_tokenizer = AutoTokenizer.from_pretrained(...)` **之後** 跟
`self._model = AutoModelForCausalLM.from_pretrained(...)` **之前** 插入 attention
偵測 + load_kwargs 組合。把現有的 `from_pretrained` 一行改成 `**load_kwargs`。

```python
        # ... (preceding lines unchanged: cuda check, self._torch = torch,
        #      self._device = "cuda:0", self._tokenizer = ...)

        # ↓ v0.7.0 新增區段 ↓
        # Resolve best available attention impl. Use try/import not the
        # transformers helper — helper's API path varies across versions.
        attn = "eager"
        for candidate in self._PREFERRED_ATTN:
            if candidate == "flash_attention_2":
                try:
                    import flash_attn  # noqa: F401
                    attn = candidate
                    break
                except ImportError:
                    continue
            elif candidate == "sdpa":
                # SDPA is built-in PyTorch 2.0+ — always available
                attn = candidate
                break
            else:
                attn = candidate
                break

        load_kwargs = dict(
            dtype=torch.bfloat16,         # NOT torch_dtype — deprecated in transformers 4.45+
            device_map=self._device,      # NOT "cuda:0" literal — uses 1151809's extracted attr
            attn_implementation=attn,
        )
        if self._USE_4BIT_QUANT:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            # quantization_config overrides dtype — drop to avoid transformers warning
            load_kwargs.pop("dtype")

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs,
        )
        self._model.eval()

        if self._USE_TORCH_COMPILE:
            # mode="reduce-overhead" optimised for autoregressive decode;
            # falls back to default mode if unsupported by the model.
            self._model = torch.compile(self._model, mode="reduce-overhead")
        # ↑ v0.7.0 新增區段 ↑

        # ... (following lines unchanged: self._system_prompt, self._max_tokens,
        #      self._model_name, self._pad_token_id = self._resolve_pad_token_id(),
        #      device_label building)
```

**改 `device_label` 組裝**(讓 log 反映 attention impl + quant + compile 狀態):

```python
        # Replace the existing self._device_label assignment with:
        quant_label = "INT4-NF4" if self._USE_4BIT_QUANT else "bfloat16"
        compile_label = " + torch.compile" if self._USE_TORCH_COMPILE else ""
        self._device_label = (
            f"{model_name} (PyTorch {quant_label} + {attn}{compile_label} "
            f"@ {gpu_name}, ≤{max_tokens} tok)"
        )
```

**`polish()` 完全不動** — 1151809 加的 `add_special_tokens=False` + `self._pad_token_id` + `self._device` 都保留。

### 改動 2:`__version__` bump

```python
__version__ = "0.6.0"  → "0.7.0"
```

### 改動 3:README updates(雙軌:v0.7.0 新功能 + v0.6.0 doc drift 清理)

**新功能(v0.7.0)**:
1. **Badge** v0.6.0 → v0.7.0
2. **Windows GPU 加速套件表**:加 `flash-attn (optional)` + `bitsandbytes (optional)` 兩個可選項目,標清楚「不必裝,但裝了能加速 polish」
3. **Windows 一鍵安裝**:加一個「v0.7.0 進階加速(選用)」小節,描述三個 toggle 跟 pre-flight check:
   ```powershell
   # 進階 1:Flash Attention 2(先確認 Blackwell prebuilt wheel 有沒有)
   pip install flash-attn --no-build-isolation
   # 進階 2:bitsandbytes 4-bit 量化
   pip install bitsandbytes
   # 進階 3:torch.compile — 不必裝套件,在 text_polisher.py 改 _USE_TORCH_COMPILE = True
   ```
4. **疑難排解**:加一段 "Polish 在 Windows 比 macOS 慢" 的條目,引到 v0.7.0 toggle

**v0.6.0 doc drift 清理**(發在 v0.7.0 順手做,不然 ship 完 README 仍對不上實際 default):
README 仍多處說 Win/Linux polish 預設是 `Qwen3-4B-Instruct-2507` (~8 GB VRAM),
但實際 code(commit `c1f0906` 後)是 `Qwen2.5-1.5B-Instruct` (~3 GB VRAM)。14 處需 sweep:

| 行號 | 現狀 | 改成 |
|------|------|------|
| 7 | 「預設 Qwen3-4B-Instruct-2507」 | 「預設 Qwen2.5-1.5B-Instruct(Win/Linux);Mac 仍 4B MLX 4-bit」 |
| 44 | 「≥ 10 GB VRAM ... + 4B polish (~8 GB)」 | 「≥ 5 GB VRAM ... + 1.5B polish (~3 GB)」 |
| 115 | 「存放空間估算 15-20 GB」 | 「~10 GB(ASR 1.2 + polish 3 + torch wheel + cache)」 |
| 159 | preset 表 Maximum ⭐ = 4B / ~10 GB | ⭐ 改在 Balanced(1.5B / ~5 GB);Maximum 變 opt-in「Quality+」tier |
| 160 | Balanced 列無星號 | 加 ⭐(實際 v0.6.0+ ship default) |
| 233 | Windows step 2「v0.6.0 預設 Maximum tier」 | 「v0.6.0+ 預設 = Balanced(1.5B)。要更高品質可切 Maximum (4B)」 |
| 243 | step 3「下載 Maximum tier ~10 GB」 | 「下載 ~4.5 GB(ASR 1.2 + polish 1.5B 3 GB)」 |
| 255 | log 範例 `polish: Qwen3-4B-Instruct-2507` | `polish: Qwen2.5-1.5B-Instruct` |
| 413 | `POLISH_MODEL` 註解「Win/Linux → 4B」 | 「Win/Linux → Qwen2.5-1.5B-Instruct」 |
| 433 | 「預設模型 Qwen3-4B-Instruct-2507」(雙平台) | 拆雙平台:「Mac → 4B MLX 4-bit;Win/Linux → 1.5B bf16」 |
| 438 | Polish 平台表 Win/Linux row 4B / ~8 GB | 1.5B / ~3 GB |
| 448 | 「Win VRAM ~10 GB」 | 「~5 GB」 |
| 461-464 | 「Win < 10 GB VRAM 想保留 polish 改 1.5B」(本末倒置) | 改「想升級品質改 4B,需 ≥ 12 GB VRAM」 |
| 624 | 跨平台設計「polish ID 都是 Qwen3-4B-Instruct-2507」 | 「Mac:4B MLX 4-bit / Win/Linux:1.5B bf16」 |

**Preset 表結構重組**(行 157-162 整段):⭐(預設)從 Maximum 移到 Balanced 列,
Balanced 列改名為「Default」或保留名稱但加註,Maximum 改名為「Quality(opt-in,需 12 GB+ VRAM)」。

**附帶小修**:
- 行 21:「核心管線(麥克風 → **Whisper** → 文字後處理)」→「→ **ASR** →」
- 行 499-505:Windows troubleshoot「CUDA load failed ...」加註此 troubleshoot 適用於 `faster-whisper` fallback tier(qwen3-asr 預設不會出這個 log)
- 行 269-279 確認狀態:可順手提一下 v0.6.0+ 加的 `[stt] zh raw -> ...` raw-diff log line

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
#    別複製 POLISH_PROMPT 字串(會 drift),直接 importlib 從 stt-daemon.py 讀:
python -c "
import time, sys, importlib.util
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'scripts')
# Import POLISH_PROMPT + POLISH_MODEL from the daemon source — no copy-drift
spec = importlib.util.spec_from_file_location('stt_daemon', 'scripts/stt-daemon.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from text_polisher import build_polisher
p = build_polisher(True, mod.POLISH_MODEL, mod.POLISH_PROMPT)
print('polish:', p.device_label)
samples = {
    '短': '那個我們等等就是說要去吃飯吧',
    '中': '呃我覺得這個 Python function 的設計可以再優化一下，然後就是說標點的部分也想要 polish 一下',
    '長': '...貼一段 ~200 字的長文,可從 v0.6.0 log 撈',
}
for label, s in samples.items():
    times = []
    outs = []
    for _ in range(3):
        t0 = time.time(); out = p.polish(s); times.append(time.time()-t0); outs.append(out)
    avg = sum(times)/len(times)
    # baseline diff metric: when same input is run 3x with greedy decode, output
    # should be identical; if not, model has non-determinism (rare for do_sample=False)
    consistent = len(set(outs)) == 1
    print(f'{label}: avg {avg:.2f}s | len(in)={len(s)} len(out)={len(outs[0])} | consistent={consistent}')
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

「品質 diff」= 跟 baseline (階段 0) 同樣輸入下,輸出**完全相同字串的比率**(0/3, 1/3, 2/3, 3/3)。
greedy decode (do_sample=False) 對同輸入該完全 deterministic,bf16 vs INT4 可能會有微小數值差。
此欄替代主觀「品質感受」,可事後 audit。

| 階段 | 配置 | 短文 polish | 中文 polish | 長文 polish | warmup time | VRAM peak | 品質 diff vs baseline |
|------|------|-----------|------------|------------|------------|-----------|---------|
| 0 | baseline (v0.6.0,eager attention) | ? s | ? s | ? s | ? s | ? GB | 3/3(自己比) |
| 1 | +sdpa | ? | ? | ? | ? | ? | ?/3 |
| 2 | +sdpa +compile | ? | ? | ? | ? | ? | ?/3 |
| 3 | +flash-attn +compile | ? | ? | ? | ? | ? | ?/3 |
| 4 | +flash-attn +compile +4bit | ? | ? | ? | ? | ? | ?/3(預期 < 3/3,可接受需人工判) |

**目標**:長文 polish < 1.5s,理想 < 1.0s;短/中文 polish < 0.5s。

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
