# Hold-to-Talk STT

按住觸發鍵講話、放開即可把語音轉成文字 **自動貼到當下焦點視窗**。中英文混合直接講沒問題、自動繁體（簡轉繁台灣）、自動在中英之間補空格。

完全離線，無 API key、無流量費。

## 平台支援

| 平台 | 狀態 | 觸發鍵（預設） | GPU 後端 |
|------|------|--------------|--------|
| **Windows 10 / 11** | ✅ 已實作、實測 | Right Alt (AltGr) **或** Right Ctrl | NVIDIA CUDA（CTranslate2 + cuDNN） |
| **macOS（含 Apple Silicon M 系列）** | 🛣️ 規劃中 | 預定 Right Option | Apple MPS / MLX / Metal（換 backend） |
| **Linux（X11 / Wayland）** | 🛣️ 規劃中 | 預定 Right Alt | NVIDIA CUDA（同 Win） |

> 🏗️ **設計上保留跨平台空間**：核心管線（麥克風 → Whisper → 文字後處理）100% 跨平台。需要分平台寫的只有薄薄三層 —— **clipboard 寫入**、**paste 模擬**、**全域熱鍵代號**。詳見 [跨平台設計](#跨平台設計) 段。

## 體驗

```
[使用者] 焦點放在任何輸入框（Terminal / Chrome / Notepad / VSCode / Slack ...）
[使用者] 按住觸發鍵 → 說「幫我 review 這個 Python function 的 async 部分」→ 放開
[daemon] 約 0.2–0.5 秒後，文字自動出現：
         「幫我 review 這個 Python function 的 async 部分」
```

—— 焦點在哪、整個系統的任何輸入框都通用。

---

## 系統需求

### 通用
- **Python**：3.10+（實測 3.12.2）
- **麥克風**：作業系統認得的任何輸入裝置

### Windows（目前唯一已實作）
- **OS**：Windows 10 / 11
- **GPU（強烈建議）**：NVIDIA 顯卡，延遲 0.2 秒級。沒 GPU 也能跑（CPU 約 5–12 秒）

### macOS / Linux（規劃中）
見 [Roadmap](#roadmap)。

---

## 依賴

### 跨平台核心（所有平台都裝）

| 套件 | 用途 |
|------|------|
| `faster-whisper` | Whisper turbo 語音模型本體（基於 CTranslate2） |
| `sounddevice` | 從麥克風讀音訊 |
| `numpy` | 音訊資料運算 |
| `pynput` | 全域 keyboard hook |
| `opencc-python-reimplemented` | 簡轉繁（s2tw，台灣正體） |

### Windows GPU 加速（**只 Windows 需要**）

| 套件 | 用途 |
|------|------|
| `nvidia-cudnn-cu12` | cuDNN 9 動態庫 |
| `nvidia-cublas-cu12` | cuBLAS 動態庫 |

> ⚠️ 這兩個是 NVIDIA 官方 cuDNN/cuBLAS 的 pip wheel，避免去 Developer 網站註冊下載。**沒 GPU 或非 Windows 就不必裝**（daemon 自動 fallback 到 CPU；macOS 之後會走 Metal/MLX 路徑）。

### Windows 一鍵安裝

```powershell
pip install --user `
    faster-whisper `
    sounddevice `
    pynput `
    opencc-python-reimplemented `
    nvidia-cudnn-cu12 `
    nvidia-cublas-cu12
```

`numpy` 通常會被 `faster-whisper` 一起拉進來；如果沒有：
```powershell
pip install --user numpy
```

### macOS / Linux 安裝（規劃）

僅裝跨平台核心即可。GPU 加速套件視平台選擇：

```bash
# macOS / Linux（不裝 nvidia-* 系列）
pip install --user \
    faster-whisper sounddevice pynput opencc-python-reimplemented
```

macOS Apple Silicon 之後可能改用 `mlx-whisper`（原生 Metal 加速）；Linux 有 NVIDIA GPU 時可加裝 `nvidia-cudnn-cu12` 等。詳見 [Roadmap](#roadmap)。

---

## 第一次啟用

### 1. Clone

```powershell
git clone https://github.com/DennisLiuCk/home-stt.git
cd home-stt
```

腳本都在 `scripts/`：

```
scripts/
├── stt-daemon.py    # daemon 主程式
├── stt-start.ps1    # 啟動
└── stt-stop.ps1     # 停止
```

### 2. 啟動 daemon

```powershell
.\scripts\stt-start.ps1
```

第一次跑會自動下載 `large-v3-turbo` 模型（約 1.5 GB）到 `~/.cache/huggingface/`，需要等 1–5 分鐘。後續啟動只要約 15 秒（model load + GPU warmup）。

成功會看到：
```
STT daemon started (PID 8232).
Log: C:\Users\<name>\AppData\Local\Temp\stt-daemon.log
Allow ~15s for model load + GPU warmup before first trigger key.
```

### 3. 使用

1. 把焦點放在任何想輸入文字的視窗
2. 按住 **Right Alt** 或 **Right Ctrl**（兩者皆可）
3. 對麥克風講話（中英文都可）
4. 放開觸發鍵
5. 約 0.2–1 秒後，文字自動出現

> 為什麼兩個鍵都綁？Right Alt 在某些應用（例如 Chrome）會跟既有快捷鍵衝突，Right Ctrl 是備援。**按下哪個就放開哪個**；同時按只有先按下的那個有效。

### 4. 確認狀態

看 log：
```powershell
Get-Content "$env:TEMP\stt-daemon.log" -Encoding utf8 -Tail 20
```

每次說話 daemon 會 log 一行：
```
[stt] zh 0.21s -> 幫我 review 這個 Python function
```

---

## 停止

```powershell
.\scripts\stt-stop.ps1
```

停止後 Right Alt / Right Ctrl 就回到原本的鍵位功能。

---

## 開機自動啟動（選用）

把 `stt-start.ps1` 的捷徑放進 Windows 啟動資料夾：

1. `Win+R` → 輸入 `shell:startup` → Enter
2. 建立 `stt-start.lnk`，目標（把路徑換成你 clone 的位置）：
   ```
   powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\path\to\home-stt\scripts\stt-start.ps1"
   ```

---

## 自訂

直接編輯 `stt-daemon.py` 頂部的 `Config` 區塊：

```python
SAMPLE_RATE   = 16000           # 麥克風取樣率
MIN_AUDIO_SEC = 0.3             # 太短的按鍵自動忽略
MODEL_NAME    = "large-v3-turbo"  # 可改為 medium / small（速度 vs 品質）
TRIGGER_KEYS  = {Key.alt_gr, Key.ctrl_r}  # set of pynput.keyboard.Key — 想換鍵 / 加鍵就改這個 set
```

改完用 `stt-stop.ps1` + `stt-start.ps1` 重啟。

### 模型大小參考

| 模型 | 大小 | GPU 延遲 | CPU 延遲 | 品質 |
|------|------|---------|----------|------|
| `small` | 460 MB | ~0.1s | ~2s | 一般 |
| `medium` | 1.5 GB | ~0.15s | ~5s | 好 |
| `large-v3-turbo`（預設） | 1.5 GB | ~0.2s | ~12s | 最好 |

---

## 疑難排解

**Daemon 啟動但按觸發鍵沒反應**
- 確認 daemon 還活著：`Get-Content "$env:TEMP\stt-daemon.log" -Encoding utf8 -Tail 5` 應該有 `warmup ... — hold Key.ctrl_r, Key.alt_gr to record.`
- 國際鍵盤布局上 Right Alt = AltGr，是預期行為。其他鍵盤可能要把 `Key.alt_gr` 改成 `Key.alt_r`

**麥克風權限**
- 第一次跑可能要去 `Windows 設定 → 隱私權與安全性 → 麥克風`，確認「允許桌面應用程式存取麥克風」是開的

**GPU 啟動失敗 fallback 到 CPU**
- log 出現 `CUDA load failed (...); falling back to CPU int8.`
- 通常是 cuDNN/cuBLAS 路徑沒掛上，或 GPU driver 太舊
- 確認 `nvidia-cudnn-cu12` 和 `nvidia-cublas-cu12` 都裝了：
  ```powershell
  pip show nvidia-cudnn-cu12 nvidia-cublas-cu12
  ```

**中文是簡體**
- 確認 `opencc-python-reimplemented` 已安裝
- 看 log 是否有 OpenCC 相關錯誤

**文字貼到不對的視窗**
- 焦點問題 — 你放開觸發鍵前必須先把焦點放到目標輸入框
- 不要按完馬上切視窗

**Daemon 死掉**
```powershell
Get-Content "$env:TEMP\stt-daemon.err.log" -Encoding utf8 -Tail 30
```

---

## 檔案結構

```
home-stt/
├── README.md
├── .gitignore
└── scripts/
    ├── stt-daemon.py    # 主程式：keyboard hook + audio capture + Whisper + clipboard/paste
    ├── stt-start.ps1    # 啟動：背景 spawn python，寫 PID file（Windows）
    ├── stt-stop.ps1     # 停止：讀 PID file 或掃 process tree 殺（Windows）
    └── stt-daemon.pid   # daemon PID（runtime 產生，已 gitignore）

%TEMP%\
├── stt-daemon.log       # 主 log（包含 transcription）
└── stt-daemon.err.log   # 錯誤 log
```

---

## 設計重點

| 問題 | 解法 |
|------|------|
| `pynput` Right Alt 在台灣鍵盤被識別為 AltGr | 監聽 `Key.alt_gr` 而非 `Key.alt_r` |
| 注音 IME 攔截英文 ASCII keypress（逐字 type 時） | 早期試 `SendInput + KEYEVENTF_UNICODE` 直送 unicode；後來放棄 |
| `KEYEVENTF_UNICODE` 在中文頓號（、）之後 IME 仍會吞掉後續字元 | **改用 clipboard + Ctrl+V 一次貼整段**：Ctrl 是 system modifier，IME 不攔；系統 paste 是原子操作不會中途斷掉 |
| PowerShell 5.1 stdin/stdout cp950 編碼簡體字爆炸 | 強制 `[Console]::InputEncoding = UTF8` / `sys.stdout.reconfigure(utf-8)` |
| Whisper 預設輸出簡體中文 | OpenCC `s2tw` 自動簡轉繁台灣正體 |
| 中文跟英文連在一起沒空格 | regex 自動在 CJK ↔ ASCII letter/digit 邊界補空格 |
| CTranslate2 找不到 cuDNN DLL | `pip install nvidia-cudnn-cu12` + 啟動時 `add_dll_directory` 和 prepend PATH |
| Cold start CUDA JIT compile 約 10 秒 | daemon 啟動時跑一次 dummy transcribe 預熱 |

---

## 跨平台設計

目前 daemon **核心管線 100% 跨平台**：

```
[mic] sounddevice
  → faster-whisper（CPU 各平台都跑；GPU backend 因平台而異）
  → OpenCC s2tw  +  regex CJK/ASCII spacing
  → [insert]      ← ★ 這層是唯一綁平台的薄層
```

需要分平台寫的只有三件事：

| 抽象層 | Windows（已實作） | macOS（規劃） | Linux（規劃） |
|--------|------------------|--------------|--------------|
| **Clipboard 寫入** | PowerShell `Set-Clipboard` | `pbcopy` | `xclip` (X11) / `wl-copy` (Wayland) |
| **Paste 模擬** | ctypes `SendInput` Ctrl+V | `osascript` Cmd+V 或 pyobjc NSEvent | `xdotool key ctrl+v` (X11) / `ydotool` 或 `wtype` (Wayland) |
| **觸發鍵代號** | `{Key.alt_gr, Key.ctrl_r}` | 預定 Right Option | 預定 Right Alt / Right Ctrl |

未來重構建議：抽象成 `Pasteboard` 介面，三平台各自 implementation，主流程不動：

```python
class Pasteboard:
    def set(self, text: str) -> None: ...
    def paste(self) -> None: ...

if sys.platform == "win32":   pasteboard = WindowsPasteboard()
elif sys.platform == "darwin": pasteboard = MacOSPasteboard()
else:                          pasteboard = LinuxPasteboard()
```

GPU 後端類似結構：Windows / Linux 走 CUDA + CTranslate2；macOS Apple Silicon 改用 `mlx-whisper`（原生 Metal）或 `whisper.cpp` + Metal。

---

## STT 模型抽象

語音模型本身也走介面化設計，方便日後 A/B 測試不同引擎。`stt-daemon.py` 中：

```python
class STTBackend(ABC):
    def transcribe(self, samples: np.ndarray) -> tuple[str, str]: ...
    def warmup(self) -> None: ...
    @property
    def device_label(self) -> str: ...

class FasterWhisperBackend(STTBackend):
    # 現有實作（Whisper large-v3-turbo via CTranslate2）

def build_backend(name: str, model: str) -> STTBackend:
    if name == "faster-whisper":
        return FasterWhisperBackend(model)
    # elif name == "sense-voice":    ← 規劃
    # elif name == "paraformer":     ← 規劃
    # elif name == "mlx-whisper":    ← 規劃
```

切換只需改頂部兩個常數：

```python
STT_BACKEND = "faster-whisper"   # or "sense-voice", "paraformer", "mlx-whisper"
STT_MODEL   = "large-v3-turbo"   # backend-specific model identifier
```

主流程 (`_transcribe_and_emit`) 只跟介面對話，**換引擎不影響 mic 收音、post-processing、clipboard/paste 任何邏輯**。

### 已實作 vs 規劃中的後端

| 後端 | 引擎 | 主要強項 | 狀態 |
|------|------|---------|-----|
| `faster-whisper` | Whisper large-v3-turbo via CTranslate2 | **中英混合 SOTA**、99 語、CUDA float16 | ✅ 預設 |
| `sense-voice` | 阿里 FunASR SenseVoice-Small | 體積 234 MB、速度極快、含情感/事件偵測、5 語 | 🛣️ 規劃 |
| `paraformer` | 阿里 FunASR Paraformer-zh | **純中文 SOTA**（非自回歸） | 🛣️ 規劃 |
| `mlx-whisper` | Apple MLX 原生 Metal | macOS Apple Silicon 上最快 Whisper backend | 🛣️ 規劃（搭配 macOS 平台支援） |

---

## Roadmap

### 平台支援

- [x] **Windows 10/11**
  - NVIDIA CUDA 加速、注音 IME 共存、繁中混英文、自動 spacing、雙 trigger key
- [ ] **macOS（含 Apple Silicon M 系列）**
  - Clipboard：`subprocess pbcopy`
  - Paste：`osascript -e 'tell application "System Events" to keystroke "v" using command down'`
  - 觸發鍵：Right Option
  - GPU 加速：搭配 `mlx-whisper` backend（原生 Metal）
  - 系統權限：「系統設定 → 隱私與安全性 → 輔助使用 / 麥克風」需手動授權 Python
- [ ] **Linux X11**
  - Clipboard：`xclip -selection clipboard`
  - Paste：`xdotool key ctrl+v`
  - GPU：與 Windows 同 NVIDIA CUDA 路徑（`nvidia-cudnn-cu12` 等 wheel 也有 Linux 版）
- [ ] **Linux Wayland**
  - Clipboard：`wl-copy`
  - Paste：`ydotool key ctrl+v`（需 `uinput` 權限）或 `wtype`
  - 注意：`pynput` 在 Wayland 受限，可能要改用 `evdev` + `uinput` 走更底層

### STT 模型 / 後端

- [x] **faster-whisper**（CTranslate2 + Whisper large-v3-turbo）
  - 中英混合 SOTA、CUDA float16 + CPU int8 fallback
- [ ] **sense-voice**（阿里 FunASR SenseVoice-Small）
  - 234 MB 體積、含情感 / 事件偵測、多語但同句內單一語言
  - 套件：`pip install funasr modelscope`
  - 待評估：中英混合在同句的表現是否能接受
- [ ] **paraformer**（阿里 FunASR Paraformer-zh）
  - 純中文場景的精度標竿、非自回歸架構速度極快
  - 同上 FunASR 套件
  - 適用：純中文場景比 Whisper 強，但英文/中英混合會崩
- [ ] **mlx-whisper**（Apple MLX 原生 Metal Whisper）
  - macOS Apple Silicon 專用
  - 同 `large-v3-turbo` 模型但跑在 Metal 上
  - 跟 [macOS 平台支援] 一起做

### 介面化重構（已部分完成）

- [x] **`STTBackend` 抽象** — 已抽出，換引擎只改 `STT_BACKEND` 常數 + 加一個 class
- [ ] **`Pasteboard` 抽象** — Clipboard 寫入 / Paste 模擬 / 觸發鍵代號要分平台，隨第二個平台支援時一起做
