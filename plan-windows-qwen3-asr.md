# Plan — Windows Qwen3-ASR + CUDA backend

> ⏳ **Disposable planning doc.** 實作完成 + v0.6.0 commit 之後 **刪掉這份檔案**。
> 不要長期留在 repo,不歸 README 管理範圍。

## 目標一句話

讓 Windows 預設模型從 `faster-whisper / Whisper-large-v3-turbo` 換成
`qwen3-asr / Qwen3-ASR-0.6B`,跑在 NVIDIA CUDA 上,跟 macOS Apple
Silicon 的 Qwen3-ASR 預設體驗對齊(中文標點 + 中英混合)。

## 為什麼不是「只改一行模型名」就好

現在的 Windows 預設用 `FasterWhisperBackend`,內部是:

```python
from faster_whisper import WhisperModel
self._model = WhisperModel(model_name, device="cuda", compute_type="float16")
```

`faster-whisper` 只能載入 **Whisper 系列** 的 CTranslate2 權重。Qwen3-ASR
的模型架構不同(Qwen3 LLM backbone + audio encoder),把 model name 換成
`"Qwen/Qwen3-ASR-0.6B"` 餵給 `WhisperModel(...)` 會在載入時 throw。

要跑 Qwen3-ASR 必須換 **另一個 inference library** — 官方提供
`qwen-asr` 套件(PyTorch + transformers + 可選 vLLM 後端)。NVIDIA 硬體
跟 driver / CUDA runtime **完全重用**,要換的只是 Python 層那條推論
pipeline。

## 不會動到的部分(放心)

- NVIDIA driver / GPU / Windows 系統
- Pasteboard 抽象(Windows ctypes SendInput 那一塊)
- pynput keyboard listener
- OpenCC s2tw 後處理
- 提示音 / 觸發鍵 / state machine
- `.ps1` 啟動腳本
- `faster-whisper` 這個 backend **繼續存在當 fallback** — 沒有刪掉,只是
  不再是預設

---

## 改動清單

### 改動 1:`scripts/stt-daemon.py` — `Qwen3AsrBackend` 改成平台感知

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

#### 改成這個(平台 dispatch + 加 PyTorch 路徑)

```python
class Qwen3AsrBackend(STTBackend):
    """Qwen3-ASR — Apple Silicon 走 mlx-qwen3-asr (Metal),其餘平台
    (Windows / Linux) 走 qwen-asr (PyTorch + transformers + CUDA 自動偵測,
    沒 CUDA fallback CPU)。"""

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
    """Windows / Linux path — qwen-asr (PyTorch + transformers).

    Auto-detects CUDA at __init__; uses bfloat16 on GPU, float32 on CPU.
    No silent fallback inside transcribe — the device chosen at init is
    the device for the whole daemon lifetime. Restart the daemon to
    re-probe (e.g. after a driver update)."""

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
        # qwen-asr accepts (numpy array, sample_rate) tuple and handles
        # any resampling internally — we provide 16 kHz mono so nothing
        # to resample.
        results = self._model.transcribe(
            audio=(samples, SAMPLE_RATE),
            language=None,  # auto-detect
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
> 回 `dict(text=..., language=...)` — 由外層 `Qwen3AsrBackend.transcribe`
> 統一做 normalization。這樣 MLX 跟 PyTorch 兩條路返回 shape 對齊。

### 改動 2:Config block — Windows / Linux 預設換 qwen3-asr

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

改成:

```python
if sys.platform == "darwin":
    if _host_platform.machine() != "arm64":
        raise SystemExit(...)
    _DEFAULT_BACKEND = "qwen3-asr"
    _DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
else:
    # Windows / Linux: qwen3-asr (PyTorch + qwen-asr) on CUDA when
    # available, CPU fallback otherwise. faster-whisper remains
    # available via STT_BACKEND="faster-whisper" for users who
    # prefer the v0.4.x default.
    _DEFAULT_BACKEND = "qwen3-asr"
    _DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
```

### 改動 3:`__version__` bump

```python
__version__ = "0.5.0"  → "0.6.0"
```

### 改動 4:README 更新清單

依重要度排序 — 沒做完不要 commit:

1. **Badge** v0.5.0 → v0.6.0(line ~3)
2. **平台表** Windows 那列「STT backend / 加速」:
   `faster-whisper + NVIDIA CUDA` →
   `Qwen3-ASR-0.6B via qwen-asr + NVIDIA CUDA(預設 v0.6.0+);也可切 faster-whisper`
3. **Windows 系統需求**(~line 38):加進「PyTorch+CUDA wheel 要先裝」這一步
4. **依賴 → Windows GPU 加速** + **Windows 一鍵安裝**:重寫成下面的「Windows 端安裝步驟」三步格式
5. **Backend 狀態表**(~line 587):
   - `qwen3-asr` 狀態欄改成 `✅ 預設(Apple Silicon v0.3.0+ / Windows + Linux v0.6.0+)`
   - `faster-whisper` 狀態欄改成 `✅ 可選(Win/Linux fallback)`
6. **Roadmap → STT 模型/後端 → qwen3-asr**:把「macOS Apple Silicon 專用」
   那句改成「Apple Silicon (MLX) + Windows/Linux (PyTorch CUDA)」
7. **Roadmap → 平台支援 → Windows 10/11**:加一句「v0.6.0 起預設 qwen3-asr,
   v0.4.x 預設 faster-whisper 仍可切換」

---

## Windows 端安裝步驟(README 一鍵安裝段要寫這個)

PyTorch 跟 CUDA 互鎖,**順序很重要** — 先裝 PyTorch+CUDA wheel,再裝 qwen-asr。否則 `pip install qwen-asr` 會自動拉 CPU 版 torch,結果模型載入但跑 CPU。

```powershell
# 步驟 1:先裝 CUDA 12.x 的 PyTorch (查你 CUDA driver 版本對應的 cuXXX 選對)
# 用 cuda 12.1 的:
pip install --user torch --index-url https://download.pytorch.org/whl/cu121
# 用 cuda 12.4:
# pip install --user torch --index-url https://download.pytorch.org/whl/cu124

# 步驟 2:裝 Qwen3-ASR 官方推論套件(會 reuse 上一步的 torch)
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

> ⚠️ 已經有舊環境的話:**不要 uninstall 既有的 torch**,先 `pip show torch` 看是否帶 cuda(`Version: 2.x.x+cu121` 帶 cu = CUDA 版)。如果是 CPU 版要先 `pip uninstall torch` 再走步驟 1。

> 💡 **存放空間警告**:CUDA torch + qwen-asr 加起來大概 5-8 GB(torch wheel ~2.5 GB,Qwen3-ASR-0.6B 權重 1.2 GB,transformers + cache 等其他)。確認 `C:` 槽有空間。

---

## 實作步驟順序(在 Windows 機上跑)

```powershell
# 0. fetch 最新
cd C:\path\to\home-stt
git fetch origin
git pull origin main

# 1. 讀這份計劃
notepad plan-windows-qwen3-asr.md   # 或在 VSCode 開

# 2. 按上面三步裝套件
# (見「Windows 端安裝步驟」)

# 3. 改 scripts/stt-daemon.py
#    - 找到 class Qwen3AsrBackend
#    - 整個 class 換成計劃裡「改動 1」的新版
#    - 在它後面加 _Qwen3MlxImpl + _Qwen3TorchImpl 兩個 class
#    - Config block 套用「改動 2」
#    - __version__ 改 "0.6.0"

# 4. 改 README.md(對照「改動 4」清單)

# 5. 本地 smoke test:不啟動 daemon,先驗證 import
python -c "import sys; sys.path.insert(0, 'scripts'); import importlib.util; spec = importlib.util.spec_from_file_location('stt_daemon', 'scripts/stt-daemon.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('OK; version:', mod.__version__); b = mod.build_backend('qwen3-asr', 'Qwen/Qwen3-ASR-0.6B'); print('backend:', b.__class__.__name__, '|', b.device_label)"
# 預期看到:
#   OK; version: 0.6.0
#   backend: Qwen3AsrBackend | NVIDIA <GPU 名> (bfloat16) — Qwen3-ASR
# 如果 device_label 變 "CPU (float32)" 表示 step 1 沒裝對 CUDA torch

# 6. 啟動 daemon
.\scripts\stt-start.ps1

# 7. tail log 確認 backend / device_label
Get-Content "$env:TEMP\stt-daemon.log" -Encoding utf8 -Tail 10
# 預期:
#   [stt] home-stt v0.6.0 starting
#   [stt] platform: win32 (AMD64) | native libs registered: 2
#   [stt] backend: qwen3-asr | model: Qwen/Qwen3-ASR-0.6B
#   [stt] warming up on NVIDIA <GPU 名> (bfloat16) — Qwen3-ASR...
#   [stt] warmup X.Xs — hold ...

# 8. 實測 hold-to-talk(中文 / 中英混合 / 純英文各幾次)
#    重點驗證:中文標點 + 中英混合處理 vs 之前 Whisper turbo

# 9. 都 OK 後刪掉本檔
git rm plan-windows-qwen3-asr.md

# 10. commit + push
#     程式 + README + 刪計劃檔 = 一個 commit:
#     git add scripts/stt-daemon.py README.md
#     git commit -m "v0.6.0: Qwen3-ASR + CUDA on Windows / Linux"
#     git tag -a v0.6.0 -m "Release v0.6.0 — Qwen3-ASR backend on Windows / Linux via qwen-asr"
#     git push origin main --tags
```

---

## 驗證 checklist

實測這幾種情境,印象與舊 Whisper turbo 比對:

| 測試句 | 重點 |
|--------|------|
| 「今天天氣很好,我們等等去吃飯,記得帶傘」 | 中文標點 |
| 「幫我 review 這個 Python function 的 async 部分」 | 中英混合 |
| 「在 Windows 用 CUDA 跑 Qwen3-ASR 比 Whisper turbo 強」 | 技術名詞 |
| "Hello world, how are you doing today?" | 純英文 |
| 1-2 秒短句、5-10 秒長句各幾次 | 延遲 + 長句穩定性 |

也檢查:
- [ ] 啟動 log device_label 顯示 NVIDIA GPU 名稱(非 CPU)
- [ ] 第一次跑會下載 Qwen3-ASR-0.6B(~1.2 GB)到 `~\.cache\huggingface\`
- [ ] 第二次啟動 warmup < 3 秒(模型已快取)
- [ ] 切回 `STT_BACKEND = "faster-whisper"` 重啟,確認舊 Whisper 路徑仍能跑
- [ ] daemon log 沒有新的 warning / error

---

## 風險 + 已知問題

1. **CUDA 版本對應**:你 Windows 上的 CUDA driver 版本要跟 `--index-url` 的
   `cuXXX` 對得上。`nvidia-smi` 看 `CUDA Version: 12.X` 就裝 cu12X 的 wheel。
   不對的話 PyTorch import 會 throw `Cannot find CUDA driver` 或類似。

2. **qwen-asr 套件初次發布還新**:PyPI 上 `qwen-asr` 是 2026 早期版本。可能
   有 API 微調或新 bug。如果裝完 import 失敗,先 `pip install -U qwen-asr`
   看有沒有新版,再來看 GitHub issue。

3. **vLLM 是 optional**:`pip install qwen-asr` 預設 transformers backend 就夠了。
   如果之後想要更高 throughput,`pip install qwen-asr[vllm]` 切 vLLM,
   但 daemon 是 single-clip-at-a-time 場景,transformers 已綽綽有餘。

4. **首次模型下載**:第一次 `.from_pretrained("Qwen/Qwen3-ASR-0.6B")` 會從 HF
   下載權重。在中國大陸網路可能要設 `HF_ENDPOINT=https://hf-mirror.com` 或
   先用 `huggingface-cli download` 預下載。

5. **記憶體**:Qwen3-ASR-0.6B + bfloat16 大概吃 ~2-3 GB VRAM。4 GB VRAM 卡
   可以跑。Qwen3-ASR-1.7B 需要 ~6-7 GB,把 STT_MODEL 改 "1.7B" 試。

6. **macOS 路徑不會被影響**:`_Qwen3MlxImpl` 是把現有 mlx-qwen3-asr 代碼搬到
   一個 class 裡,行為等價。改完後在 Mac 上 daemon 跑起來應該跟現在 v0.5.0
   完全一樣。如果 Mac 端有問題那是 refactor 出錯,不是 PyTorch 路徑的問題。

---

## 完成後清理(別忘了)

1. `git rm plan-windows-qwen3-asr.md`
2. `git commit -am "v0.6.0: Qwen3-ASR + CUDA on Windows / Linux"`(或分兩個 commit)
3. `git tag -a v0.6.0 -m "Release v0.6.0 — Qwen3-ASR backend on Windows / Linux via qwen-asr"`
4. `git push origin main --tags`

如果實作過程發現本計劃有誤,**直接改本計劃 commit 一版,再進實作** — 之後
別人(包括未來的你)看 git history 就能看到「計劃寫錯怎麼修」。
