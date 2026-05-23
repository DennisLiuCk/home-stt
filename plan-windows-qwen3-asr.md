# Plan — v0.6.0 Windows complete: Qwen3-ASR + LLM polish (both on CUDA)

> ⏳ **Disposable planning doc.** 實作完成 + v0.6.0 commit 之後 **刪掉這份檔案**。
> 不要長期留在 repo,不歸 README 管理範圍。

## 目標一句話

讓 Windows v0.6.0 跟 macOS v0.5.0 的**完整使用體驗對齊** —
- **ASR**:預設從 `faster-whisper / Whisper-large-v3-turbo` 換成 `qwen3-asr / Qwen3-ASR-0.6B`,跑 NVIDIA CUDA
- **Polish 後處理**:預設啟用 `Qwen3-4B-Instruct-2507` LLM 修飾文字,跑同一張 GPU
- 兩條 pipeline 共用 PyTorch + CUDA + transformers 基礎設施

完整流程: 麥克風 → Qwen3-ASR(CUDA) → OpenCC s2tw → polish LLM(CUDA) → 剪貼簿 → paste。

## 為什麼不是「只改一行模型名」就好

兩件事兩條軌:

**ASR 端** — 現在的 Windows 預設用 `FasterWhisperBackend`,內部 `from faster_whisper import WhisperModel`。`faster-whisper` 只能載入 Whisper 系列的 CTranslate2 權重,Qwen3-ASR 是不同架構(Qwen3 LLM backbone + audio encoder),要換 inference library 用官方 `qwen-asr` 套件(PyTorch + transformers)。

**Polish 端**(v0.5.0 加進來的) — macOS 現在跑 `MlxLocalLlmPolisher`,內部 `from mlx_lm import load, generate`。**`mlx-lm` 只有 Apple Silicon 可用**,Windows 上 import 就炸。要在 Windows 跑同樣的 polish 必須走 `transformers.AutoModelForCausalLM` PyTorch 路徑,跟 ASR 端共用 torch + transformers。

NVIDIA driver、CUDA runtime、GPU 硬體 **完全重用**;要換的只是 Python 層那兩條推論 pipeline 的 library。

## 不會動到的部分(放心)

- NVIDIA driver / GPU / Windows 系統
- Pasteboard 抽象(Windows ctypes SendInput 那一塊)
- pynput keyboard listener
- OpenCC s2tw 後處理
- 提示音 / 觸發鍵 / state machine / `_transcribe_and_emit` pipeline
- `.ps1` 啟動腳本
- **`TextPostProcessor` ABC 跟 `NoopPolisher`** — 一個字都不動
- `MlxLocalLlmPolisher` — macOS 路徑保持原樣
- `faster-whisper` 跟 `FasterWhisperBackend` 繼續存在當 ASR fallback,沒刪

---

## 改動清單

### 改動 1:`scripts/stt-daemon.py` — `Qwen3AsrBackend` 平台感知

#### 現在(macOS-only,跑在 MLX)

```python
class Qwen3AsrBackend(STTBackend):
    name = "qwen3-asr"

    def __init__(self, model_name: str):
        import mlx_qwen3_asr  # lazy import (Apple Silicon only)
        self._mqa = mlx_qwen3_asr
        self._model_name = self._resolve_model_name(model_name)
        self._device_label = "Apple Silicon (Metal, MLX) — Qwen3-ASR"

    def transcribe(self, samples):
        result = self._mqa.transcribe(samples, model=self._model_name, verbose=False)
        text = (getattr(result, "text", "") or "").strip()
        raw_lang = (getattr(result, "language", "") or "").strip().lower()
        language = self._LANG_NORM.get(raw_lang, raw_lang[:2] if raw_lang else "")
        return text, language

    def warmup(self):
        warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._mqa.transcribe(warm_audio, model=self._model_name, verbose=False)
```

#### 改成(平台 dispatch + PyTorch 路徑)

```python
class Qwen3AsrBackend(STTBackend):
    """Qwen3-ASR — Apple Silicon 走 mlx-qwen3-asr (Metal),其餘平台
    (Windows / Linux) 走 qwen-asr (PyTorch + transformers + CUDA)。"""

    name = "qwen3-asr"

    _LANG_NORM = {
        "chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko",
        "french": "fr", "german": "de", "spanish": "es", "portuguese": "pt",
        "russian": "ru", "italian": "it", "arabic": "ar",
    }

    def __init__(self, model_name: str):
        self._model_name = self._resolve_model_name(model_name)
        if sys.platform == "darwin" and _host_platform.machine() == "arm64":
            self._impl = _Qwen3MlxImpl(self._model_name)
        else:
            self._impl = _Qwen3TorchImpl(self._model_name)

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        if "/" in model_name:
            return model_name
        m = model_name.lower()
        if "1.7b" in m or "1_7" in m:
            return "Qwen/Qwen3-ASR-1.7B"
        if "0.6b" in m or "0_6" in m:
            return "Qwen/Qwen3-ASR-0.6B"
        return "Qwen/Qwen3-ASR-0.6B"

    @property
    def device_label(self) -> str:
        return self._impl.device_label

    def transcribe(self, samples):
        result = self._impl.transcribe(samples)
        text = (result.get("text") or "").strip()
        raw_lang = (result.get("language") or "").strip().lower()
        language = self._LANG_NORM.get(raw_lang, raw_lang[:2] if raw_lang else "")
        return text, language

    def warmup(self):
        self._impl.warmup()


class _Qwen3MlxImpl:
    """Apple Silicon path — mlx-qwen3-asr."""

    def __init__(self, model_name: str):
        import mlx_qwen3_asr
        self._mqa = mlx_qwen3_asr
        self._model_name = model_name
        self.device_label = "Apple Silicon (Metal, MLX) — Qwen3-ASR"

    def transcribe(self, samples):
        result = self._mqa.transcribe(samples, model=self._model_name, verbose=False)
        return {
            "text": getattr(result, "text", ""),
            "language": getattr(result, "language", ""),
        }

    def warmup(self):
        warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._mqa.transcribe(warm_audio, model=self._model_name, verbose=False)


class _Qwen3TorchImpl:
    """Windows / Linux path — qwen-asr (PyTorch + transformers + CUDA).

    Auto-detects CUDA at __init__; uses bfloat16 on GPU, float32 on CPU.
    No silent fallback inside transcribe — the device chosen at init is
    the device for the whole daemon lifetime."""

    def __init__(self, model_name: str):
        import torch  # heavy; pulled in by qwen-asr install
        from qwen_asr import Qwen3ASRModel

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            device = "cuda:0"
            dtype = torch.bfloat16
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "CUDA device"
            self.device_label = f"NVIDIA {gpu_name} (bfloat16) — Qwen3-ASR"
        else:
            device = "cpu"
            dtype = torch.float32
            self.device_label = "CPU (float32) — Qwen3-ASR (no CUDA detected)"
            print("[stt] qwen-asr fell back to CPU — install torch with CUDA "
                  "support for GPU acceleration (see plan-windows-qwen3-asr.md).",
                  file=sys.stderr, flush=True)

        self._model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        self._model_name = model_name

    def transcribe(self, samples):
        results = self._model.transcribe(
            audio=(samples, SAMPLE_RATE),
            language=None,
        )
        if not results:
            return {"text": "", "language": ""}
        r = results[0]
        return {
            "text": getattr(r, "text", "") or "",
            "language": getattr(r, "language", "") or "",
        }

    def warmup(self):
        warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._model.transcribe(audio=(warm_audio, SAMPLE_RATE), language=None)
```

> **重點**:`_Qwen3MlxImpl.transcribe` 跟 `_Qwen3TorchImpl.transcribe` 都
> 回 `dict(text=..., language=...)` — 由外層統一 normalization。

---

### 改動 2:`scripts/text_polisher.py` — 加 `TorchLocalLlmPolisher`

#### 現在(macOS-only,跑在 mlx-lm)

```python
class MlxLocalLlmPolisher(TextPostProcessor):
    def __init__(self, model_name, system_prompt, max_tokens=256):
        from mlx_lm import load, generate
        self._generate = generate
        self._model, self._tokenizer = load(model_name)
        ...

def build_polisher(enabled, model_name, system_prompt):
    if not enabled:
        return NoopPolisher()
    try:
        return MlxLocalLlmPolisher(model_name, system_prompt)
    except Exception as e:
        print(f"[stt] polish disabled — could not initialise {model_name}: {e}",
              file=sys.stderr, flush=True)
        return NoopPolisher()
```

#### 改成(加 Torch impl + build_polisher 平台分支)

`MlxLocalLlmPolisher` 完全不動。在後面加 `TorchLocalLlmPolisher`,並把 `build_polisher` 改成依平台分流。

```python
class TorchLocalLlmPolisher(TextPostProcessor):
    """Windows / Linux polish path — transformers + PyTorch + CUDA.

    Uses HuggingFace transformers AutoModelForCausalLM with bfloat16 on
    CUDA. Refuses to run on CPU (a 4B model on CPU is 30-60s/polish,
    way over the daemon's perceptual budget) — falls back to
    NoopPolisher via build_polisher's exception handler.

    VRAM: Qwen3-4B-Instruct-2507 bfloat16 needs ~8 GB on GPU. Combined
    with Qwen3-ASR-0.6B (~1.5 GB), the daemon wants ~10 GB VRAM total.
    Cards with <10 GB VRAM should set POLISH_ENABLED = False, or pick
    a smaller polish model (Qwen2.5-1.5B-Instruct ~3 GB bf16) — the
    smaller-model trade-off is documented in v0.5.0's README polish
    section.
    """

    def __init__(self, model_name: str, system_prompt: str,
                 max_tokens: int = 256) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TorchLocalLlmPolisher requires CUDA. On CPU the 4B "
                "polish model takes 30-60s per polish — way over the "
                "daemon's perceptual budget. Set POLISH_ENABLED = False "
                "to disable, or install torch with CUDA support."
            )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self._model.eval()
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._model_name = model_name
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA device"
        self.device_label = (
            f"{model_name} (PyTorch bfloat16 @ NVIDIA {gpu_name}, "
            f"≤{max_tokens} tok)"
        )

    def polish(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        try:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": text},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda:0")
            with self._torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_tokens,
                    do_sample=False,  # greedy = deterministic + faster
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            response = self._tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            if not response:
                return text
            # Strip a single matched pair of wrapping quotes if the model
            # added them despite the "no quotes" instruction.
            if (response.startswith('"') and response.endswith('"')) or \
               (response.startswith("「") and response.endswith("」")):
                response = response[1:-1].strip() or response
            return response
        except Exception as e:
            print(f"[stt] polish failed, returning raw: {e}",
                  file=sys.stderr, flush=True)
            return text


def build_polisher(
    enabled: bool,
    model_name: str,
    system_prompt: str,
) -> TextPostProcessor:
    """Factory — dispatches on `sys.platform` like build_pasteboard /
    build_backend. Falls back to NoopPolisher on any init failure."""
    if not enabled:
        return NoopPolisher()
    try:
        import sys
        import platform
        if sys.platform == "darwin" and platform.machine() == "arm64":
            return MlxLocalLlmPolisher(model_name, system_prompt)
        return TorchLocalLlmPolisher(model_name, system_prompt)
    except Exception as e:
        print(
            f"[stt] polish disabled — could not initialise "
            f"{model_name}: {e}",
            file=sys.stderr, flush=True,
        )
        return NoopPolisher()
```

> 對應 ASR 那邊 `_Qwen3MlxImpl` / `_Qwen3TorchImpl` 結構 — polish 也分
> 一個 MLX impl + 一個 Torch impl,`build_polisher` 在最外層 dispatch。

---

### 改動 3:Config block — 平台感知預設(ASR + polish 都加)

找到目前的:

```python
if sys.platform == "darwin":
    if _host_platform.machine() != "arm64":
        raise SystemExit(...)
    _DEFAULT_BACKEND = "qwen3-asr"
    _DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
else:
    _DEFAULT_BACKEND = "faster-whisper"
    _DEFAULT_MODEL = "large-v3-turbo"
```

跟下面的:

```python
POLISH_MODEL     = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"
```

改成統一塊:

```python
if sys.platform == "darwin":
    if _host_platform.machine() != "arm64":
        raise SystemExit(...)
    _DEFAULT_BACKEND = "qwen3-asr"
    _DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
    _DEFAULT_POLISH_MODEL = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"
else:
    # Windows / Linux: qwen3-asr (PyTorch + qwen-asr) on CUDA + polish via
    # transformers AutoModelForCausalLM on the same CUDA device. The
    # polish model name is the HF original (not the MLX-quantised variant
    # used on macOS) — transformers loads it directly with bfloat16.
    _DEFAULT_BACKEND = "qwen3-asr"
    _DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
    _DEFAULT_POLISH_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
```

然後 `POLISH_MODEL` 改用這個預設:

```python
POLISH_MODEL = _DEFAULT_POLISH_MODEL
```

`POLISH_ENABLED` / `POLISH_LANGUAGES` / `POLISH_PROMPT` **不變**。

---

### 改動 4:`__version__` bump

```python
__version__ = "0.5.0"  → "0.6.0"
```

---

### 改動 5:README 更新清單

依重要度排序 — 沒做完不要 commit:

1. **Badge** v0.5.0 → v0.6.0(line 3)
2. **平台表** Windows 那列「STT backend / 加速」:
   `faster-whisper + NVIDIA CUDA` →
   `Qwen3-ASR-0.6B via qwen-asr + LLM polish via transformers,皆跑 NVIDIA CUDA(預設 v0.6.0+);也可切 faster-whisper`
3. **Windows 系統需求**:加進「PyTorch+CUDA wheel 要先裝」+「polish 預設 ON,需 ~10GB VRAM,< 10GB VRAM 卡建議設 `POLISH_ENABLED = False`」
4. **依賴 → Windows GPU 加速** + **Windows 一鍵安裝**:重寫成下面「Windows 端安裝步驟」三步格式(transformers 由 qwen-asr 拉,polish 共用)
5. **Backend 狀態表**:
   - `qwen3-asr` 狀態改 `✅ 預設(Apple Silicon v0.3.0+ / Windows + Linux v0.6.0+)`
   - `mlx-lm` 那一列加註 `(僅 Apple Silicon — Windows/Linux v0.6.0+ 改用 transformers + CUDA 走同個 polish model)`
   - `faster-whisper` 狀態改 `✅ 可選(Win/Linux fallback)`
6. **Polish 後處理章節**(README 已有,line ~380):加一句「Windows v0.6.0+ 路徑用 transformers + CUDA,模型 ID `Qwen/Qwen3-4B-Instruct-2507`(非 MLX 變體);需要 ~8 GB VRAM」
7. **Roadmap → 平台支援 → Windows 10/11**:加「v0.6.0 起預設 qwen3-asr + LLM polish,v0.4.x/v0.5.0 的 faster-whisper 仍可切」
8. **Roadmap → STT 模型/後端 → qwen3-asr**:把「macOS Apple Silicon 專用」改成
   「Apple Silicon (MLX) + Windows/Linux (PyTorch CUDA)」
9. **Roadmap → 介面化重構 → TextPostProcessor**:加「v0.6.0 加 Torch impl,Windows/Linux 走 transformers+CUDA」

---

## Windows 端安裝步驟(README 一鍵安裝段要寫這個)

PyTorch 跟 CUDA 互鎖,**順序很重要** — 先裝 PyTorch+CUDA wheel,再裝 qwen-asr。否則 `pip install qwen-asr` 會自動拉 CPU 版 torch,結果 ASR 跟 polish 都跑 CPU(slow)。

```powershell
# 步驟 1:先裝 CUDA 12.x 的 PyTorch (查你 CUDA driver 版本對應的 cuXXX 選對)
# 用 cuda 12.1 的:
pip install --user torch --index-url https://download.pytorch.org/whl/cu121
# 用 cuda 12.4:
# pip install --user torch --index-url https://download.pytorch.org/whl/cu124

# 步驟 2:裝 Qwen3-ASR 官方推論套件(會 reuse 上一步的 torch,順便拉 transformers)
pip install --user qwen-asr

# 步驟 3:其他跨平台依賴 + faster-whisper 保留當 fallback
pip install --user `
    sounddevice `
    pynput `
    opencc-python-reimplemented `
    faster-whisper `
    nvidia-cudnn-cu12 `
    nvidia-cublas-cu12
```

⚠️ **polish 不需要額外裝套件** — `qwen-asr` 已經拉了 `transformers`,`TorchLocalLlmPolisher` 直接用同一個 `transformers.AutoModelForCausalLM`。

⚠️ **已經有舊環境的話**:**不要 uninstall 既有的 torch**,先 `pip show torch` 看是否帶 cuda(`Version: 2.x.x+cu121` 帶 cu = CUDA 版)。如果是 CPU 版要先 `pip uninstall torch` 再走步驟 1。

💡 **存放空間警告**:CUDA torch + qwen-asr + Qwen3-4B-Instruct polish 模型加起來大概 15-20 GB(torch wheel ~2.5 GB,Qwen3-ASR-0.6B 權重 1.2 GB,Qwen3-4B-Instruct-2507 權重 ~8 GB bf16,transformers + cache 等其他)。確認 `C:` 槽有 ≥ 25 GB 空間。

💡 **VRAM 估算**:
- Qwen3-ASR-0.6B bfloat16:~1.5 GB VRAM(常駐)
- Qwen3-4B-Instruct-2507 bfloat16:~8 GB VRAM(常駐,polish 跑時 +activations ~1 GB peak)
- **總計**:~10-11 GB VRAM target(常駐)
- **< 10 GB VRAM 卡**:預期 polish 載入會 CUDA OOM。
  - 對策 1(推薦):`POLISH_ENABLED = False` — 退回 v0.4.x 風格,ASR 仍跑 CUDA
  - 對策 2:改 `POLISH_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"`(~3 GB VRAM,品質中等)
  - 對策 3:用 bitsandbytes 4-bit 量化(要 `pip install bitsandbytes` + 改 `from_pretrained` 加 `load_in_4bit=True`)
- **8 GB 卡(GTX 1080 / RTX 2070 / RTX 4060 / 部分筆電 GPU)**:預設配置會炸,**建議直接走對策 1**

---

## 實作步驟順序(在 Windows 機上跑)

```powershell
# 0. fetch 最新
cd C:\path\to\home-stt
git fetch origin
git pull origin main

# 1. 讀這份計劃
notepad plan-windows-qwen3-asr.md

# 2. 按上面三步裝套件(見「Windows 端安裝步驟」)

# 3. 改 scripts/stt-daemon.py
#    - 找 class Qwen3AsrBackend,整個 class 換成「改動 1」的新版
#    - 在它後面加 _Qwen3MlxImpl + _Qwen3TorchImpl 兩個 class
#    - Config block 套用「改動 3」 — 加 _DEFAULT_POLISH_MODEL,改 POLISH_MODEL
#    - __version__ 改 "0.6.0"

# 4. 改 scripts/text_polisher.py
#    - MlxLocalLlmPolisher 不動
#    - 在它後面加 TorchLocalLlmPolisher class(「改動 2」)
#    - build_polisher 改成有 sys.platform 分支版本

# 5. 改 README.md(對照「改動 5」清單)

# 6. 本地 smoke test:不啟動 daemon,先驗證 import + Torch path
python -c "import sys; sys.path.insert(0, 'scripts'); import importlib.util; spec = importlib.util.spec_from_file_location('stt_daemon', 'scripts/stt-daemon.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('OK; version:', mod.__version__); b = mod.build_backend('qwen3-asr', 'Qwen/Qwen3-ASR-0.6B'); print('asr:', b.__class__.__name__, '|', b.device_label)"
# 預期:
#   OK; version: 0.6.0
#   asr: Qwen3AsrBackend | NVIDIA <GPU 名> (bfloat16) — Qwen3-ASR
# 如果 device_label 變 "CPU (float32)" 表示 step 2 沒裝對 CUDA torch

# 7. polish 也獨立 smoke test:
python -c "import sys; sys.path.insert(0, 'scripts'); from text_polisher import build_polisher; p = build_polisher(True, 'Qwen/Qwen3-4B-Instruct-2507', '你是文字助手,移除冗字輸出乾淨文字'); print('polish:', p.device_label); print(p.polish('呃我覺得這個 Python function 可以再優化'))"
# 預期:
#   polish: Qwen/Qwen3-4B-Instruct-2507 (PyTorch bfloat16 @ NVIDIA <GPU 名>, ≤256 tok)
#   我覺得這個 Python function 可以再優化
# 如果 build_polisher fallback 到 NoopPolisher,看 stderr 知道是 OOM 還是套件問題

# 8. 啟動 daemon
.\scripts\stt-start.ps1

# 9. tail log 確認兩條 model 都在 CUDA
Get-Content "$env:TEMP\stt-daemon.log" -Encoding utf8 -Tail 15
# 預期:
#   [stt] home-stt v0.6.0 starting
#   [stt] platform: win32 (AMD64) | native libs registered: 2
#   [stt] backend: qwen3-asr | model: Qwen/Qwen3-ASR-0.6B
#   [stt] polish: Qwen/Qwen3-4B-Instruct-2507 (PyTorch bfloat16 @ NVIDIA <GPU 名>, ≤256 tok)
#   [stt] warming up on NVIDIA <GPU 名> (bfloat16) — Qwen3-ASR...
#   [stt] warmup X.Xs — hold ...
# 注意 polish 那行:如果跑 NoopPolisher(VRAM 不夠 / 套件問題)會印 'disabled (raw ASR output)'

# 10. 實測 hold-to-talk(中文 / 中英混合 / 純英文)

# 11. 都 OK 後刪掉本檔
git rm plan-windows-qwen3-asr.md

# 12. commit + tag + push
#     git add scripts/stt-daemon.py scripts/text_polisher.py README.md
#     git commit -m "v0.6.0: Qwen3-ASR + LLM polish on Windows / Linux via PyTorch CUDA"
#     git tag -a v0.6.0 -m "Release v0.6.0 — Qwen3-ASR backend + LLM polish on Windows / Linux via PyTorch CUDA"
#     git push origin main --tags
```

---

## 驗證 checklist

### ASR 部分(同 macOS 路徑驗證,跟 Whisper turbo 對比)

| 測試句 | 重點 |
|--------|------|
| 「今天天氣很好,我們等等去吃飯,記得帶傘」 | 中文標點 |
| 「幫我 review 這個 Python function 的 async 部分」 | 中英混合(專有名詞保留) |
| 「在 Windows 用 CUDA 跑 Qwen3-ASR 比 Whisper turbo 強」 | 技術名詞 |
| "Hello world, how are you doing today?" | 純英文(polish bypass) |
| 1-2 秒短句、5-10 秒長句各幾次 | 延遲 + 長句穩定性 |

### Polish 部分

| 測試句(語音輸入) | 重點 |
|--------------------|------|
| 「呃我我我覺得這個 Python function 設計可以再優化一下」 | 移除冗字 + 保留 Python function |
| 「那個 commit 之後我們就是 push 到 remote 然後再開 PR」 | 移除「那個」「就是」「然後」+ 保留所有英文技術名詞 |
| 「然後就是說,那個現在的標點符號還有分詞,我覺得是個問題」 | 重整句結構不過度改寫 |
| "Hello world, this is a test" | 純英文 bypass(log 沒 `+polish` 段) |

### 系統檢查

- [ ] 啟動 log:`backend: qwen3-asr` + `polish: Qwen/Qwen3-4B-Instruct-2507 (PyTorch bfloat16 @ NVIDIA ...)`
- [ ] 啟動 log NO `polish: disabled (raw ASR output)` (除非你刻意 `POLISH_ENABLED = False`)
- [ ] 第一次跑下載 Qwen3-ASR-0.6B(~1.2 GB)+ Qwen3-4B-Instruct(~8 GB)到 `~\.cache\huggingface\`
- [ ] 第二次啟動 warmup < 10 秒(模型已快取)
- [ ] 跑 polish 時 GPU VRAM 用量 ~ 10-11 GB(用 `nvidia-smi` 看)
- [ ] 切回 `STT_BACKEND = "faster-whisper"` 重啟,確認舊路徑仍能跑(不會碰 polish 那段邏輯,polish 仍會載入)
- [ ] 切 `POLISH_ENABLED = False` 重啟,確認 daemon 載入時間少了 ~10 秒、VRAM 用量回 ~1.5 GB
- [ ] daemon 跑幾分鐘後 stderr 沒有新的 warning / OOM error

---

## 風險 + 已知問題

1. **CUDA 版本對應**:Windows 上的 CUDA driver 版本要跟 `--index-url` 的 `cuXXX` 對得上。`nvidia-smi` 看 `CUDA Version: 12.X` 就裝 cu12X 的 wheel。不對的話 PyTorch import 會 throw `Cannot find CUDA driver` 或類似。

2. **VRAM 不夠 polish 載入會 CUDA OOM**:< 10 GB VRAM 的卡(GTX 1080 / RTX 2070 / RTX 4060 / 大多筆電 GPU)在預設配置會炸。`build_polisher` 的 exception handler 抓到後印 stderr + fall back 到 NoopPolisher,所以 daemon 不會 crash,但 polish 永遠不會跑。
   **判斷方式**:啟動 log 看 `[stt] polish:` 那一行,如果是 `disabled (raw ASR output)` 就是失敗了。對策見「VRAM 估算」段。

3. **qwen-asr 套件**:PyPI 上 `qwen-asr` 是 2026 早期版本。可能有 API 微調或新 bug。如果裝完 import 失敗,先 `pip install -U qwen-asr` 看有沒有新版。

4. **transformers 版本相容性**:Qwen3-4B-Instruct-2507 需要較新版 transformers(≥ 4.45 左右)。`pip install qwen-asr` 應該會拉到夠新的版本;若手動 pin 舊版可能載入失敗。

5. **首次模型下載**:第一次 `.from_pretrained()` 從 HF 下載權重(ASR ~1.2 GB + polish ~8 GB)。中國大陸網路可能要設 `HF_ENDPOINT=https://hf-mirror.com` 或先用 `huggingface-cli download` 預下載。

6. **polish 延遲在 Windows 可能不一樣**:macOS MLX 4-bit 的 polish 是 ~0.3 秒,Windows transformers bfloat16 預期 0.5-1 秒(無量化、bigger model 在 GPU 上其實也滿快)。如果超過 1.5 秒可能 GPU 不夠強 / 沒走 CUDA,看 `nvidia-smi` 確認 GPU 確實在用。

7. **macOS 路徑不會被影響**:`_Qwen3MlxImpl` + `MlxLocalLlmPolisher` 是把現有 mlx-* 代碼搬到新 class 結構/維持原樣,行為等價。改完後在 Mac 上 daemon 跑起來應該跟現在 v0.5.0 完全一樣。如果 Mac 端有問題那是 refactor 出錯,不是 PyTorch 路徑的問題。

---

## 完成後清理(別忘了)

1. `git rm plan-windows-qwen3-asr.md`
2. `git add scripts/stt-daemon.py scripts/text_polisher.py README.md`
3. `git commit -m "v0.6.0: Qwen3-ASR + LLM polish on Windows / Linux via PyTorch CUDA"`
4. `git tag -a v0.6.0 -m "Release v0.6.0 — Qwen3-ASR backend + LLM polish on Windows / Linux via PyTorch CUDA"`
5. `git push origin main --tags`

如果實作過程發現本計劃有誤,**直接改本計劃 commit 一版,再進實作** — 之後別人(包括未來的你)看 git history 就能看到「計劃寫錯怎麼修」。
