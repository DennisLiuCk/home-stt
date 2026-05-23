# Hold-to-Talk STT

![version](https://img.shields.io/badge/version-0.6.0-blue) ![platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20(Apple%20Silicon)-lightgrey) ![python](https://img.shields.io/badge/python-3.10%2B-green)

按住觸發鍵講話、放開即可把語音轉成文字 **自動貼到當下焦點視窗**。中英文混合直接講沒問題、自動繁體（簡轉繁台灣）、自動在中英之間補空格、按下與處理完成都有提示音。

v0.5.0+ 起 **可選 LLM polish 後處理層**:走小型本地 LLM (預設 Qwen3-4B-Instruct-2507) 移除口語贅字(呃、嗯、就是、那個、然後)、修口誤,讓貼出去的文字更乾淨。可關。v0.6.0 起 Windows / Linux 也跑 polish(走 PyTorch + transformers + CUDA bfloat16)。

完全離線，無 API key、無流量費。

## 平台支援

| 平台 | 狀態 | 觸發鍵（預設） | STT backend / 加速 |
|------|------|--------------|------------------|
| **Windows 10 / 11** | ✅ 已實作、實測 | Right Alt (AltGr) **或** Right Ctrl | **Qwen3-ASR-0.6B via qwen-asr + LLM polish via transformers**（皆跑 NVIDIA CUDA bfloat16,預設 v0.6.0+）/ 也可切 `faster-whisper`(無 polish) |
| **macOS（Apple Silicon M 系列）** | ✅ 已實作 | Right Option | **Qwen3-ASR-0.6B via mlx-qwen3-asr**（Metal 原生,預設 v0.3.0+）/ 也可切 mlx-whisper |
| **Linux（X11 / Wayland）** | 🛣️ 規劃中 | 預定 Right Alt | 同 Windows 路徑(qwen-asr + transformers + CUDA);pasteboard 層尚未實作 |

> ⛔ **Intel Mac (darwin x86_64) 不再支援(v0.4.0+)** — daemon 啟動會 SystemExit 並提示降版到 v0.3.0。Intel 機已罕見,維護 + 文件成本不划算。要 Intel Mac 請 `git checkout v0.3.0`。

> 🏗️ **核心管線（麥克風 → Whisper → 文字後處理）100% 跨平台**。平台特定的薄薄三層 —— **clipboard 寫入**、**paste 模擬**、**全域熱鍵代號** —— 從 v0.2.0 起抽到 `Pasteboard` 介面實作。詳見 [跨平台設計](#跨平台設計) 段。

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

### Windows
- **OS**：Windows 10 / 11
- **硬體（v0.6.0+ 預設配置）**：**NVIDIA GPU ≥ 10 GB VRAM**(RTX 4070 / 4080 / 4090 等)。預設同時跑 Qwen3-ASR-0.6B(~1.5 GB)+ Qwen3-4B-Instruct polish(~8 GB)在 CUDA bfloat16
- **PyTorch CUDA wheel 必裝**:`qwen-asr` 跟 polish 都靠 PyTorch + transformers + CUDA。順序很重要,**先裝 torch+CUDA wheel,再 `pip install qwen-asr`**,否則 pip 會拉 CPU 版 torch,ASR + polish 都會跑 CPU(實質不可用)。詳見 [Windows 一鍵安裝](#windows-一鍵安裝)
- **依硬體選 preset**:見 [依硬體選擇 Preset](#依硬體選擇-preset)。低 VRAM 卡(< 10 GB)請看 Balanced / Light tier,polish 改小模型或關閉
- **不想裝 PyTorch CUDA 也可以**:切到 Mini tier(`STT_BACKEND = "faster-whisper"` + `POLISH_ENABLED = False`),回到 v0.4.x 風格,無 polish 但安裝簡單

### macOS（v0.2.0 起支援）
- **OS**：macOS 12 Monterey 以上（實測 26.1 Tahoe / Apple Silicon）
- **硬體**：**Apple Silicon（M1 以上)專用**(v0.4.0 起);走 Apple MLX 在 Metal 上原生跑 **Qwen3-ASR-0.6B**(預設),延遲 ~0.3-0.5 秒,中文標點 + 中英混合表現比 Whisper turbo 強。Intel Mac 不再支援。
- **權限**：第一次跑要去「系統設定 → 隱私權與安全性」開三個權限（Input Monitoring / Accessibility / 麥克風），詳見 [macOS 第一次啟用](#macos-第一次啟用)

### Linux（規劃中）
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

### Windows GPU 加速套件(v0.6.0+ 預設配置)

| 套件 | 用途 |
|------|------|
| `torch`(CUDA wheel) | PyTorch + CUDA runtime;`qwen-asr` 跟 polish 都建基於此 |
| `qwen-asr` | Alibaba 官方 Qwen3-ASR 推論套件(會拉 transformers) |
| `transformers` | LLM polish 用(`AutoModelForCausalLM` 載入 Qwen3-4B-Instruct);`qwen-asr` 已會一起拉進來,通常不必獨立裝 |
| `nvidia-cudnn-cu12` | cuDNN 9 動態庫(只 `faster-whisper` fallback 路徑需要) |
| `nvidia-cublas-cu12` | cuBLAS 動態庫(同上) |

> ⚠️ **不裝 PyTorch CUDA 就跑不動 v0.6.0 預設**。回到 v0.4.x 路徑(只裝 `faster-whisper` + cuDNN)請切 Mini tier — 見 [依硬體選擇 Preset](#依硬體選擇-preset)。macOS Apple Silicon 走 Metal/MLX 路徑,完全不需要這些。

### Windows 一鍵安裝

順序很重要 —— 先裝 PyTorch+CUDA wheel,**再**裝 `qwen-asr`,否則 pip 會自動拉 CPU 版 torch,結果 ASR 跟 polish 都跑 CPU(慢到不可用)。

```powershell
# 步驟 1:先裝 CUDA 12.x 的 PyTorch(查你 nvidia-smi 看到的 CUDA Version 對應的 cuXXX)
#   CUDA 12.1:
pip install --user torch --index-url https://download.pytorch.org/whl/cu121
#   CUDA 12.4:
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

> 💡 **polish 不需要額外裝套件** — 步驟 2 的 `qwen-asr` 會拉 `transformers`,polish 直接共用同一個 `transformers.AutoModelForCausalLM`。

> 💡 **舊環境已有 CPU 版 torch**:`pip show torch` 看 `Version:` 那一行;如果結尾沒 `+cu121` / `+cu124` 就是 CPU 版,要先 `pip uninstall torch` 再走步驟 1。**不要 uninstall 已經有 CUDA 標記的 torch**。

> 💡 **HF cache 路徑長**:HuggingFace 模型 cache 在 `~\.cache\huggingface\hub\models--Qwen--...\snapshots\<hash>\`,Windows 10/11 預設 260 字元路徑限制可能踩到(特別是用戶名長 / 安裝在 OneDrive 同步目錄下)。建議設環境變數 `setx HF_HOME D:\hf_cache` 改到較短的路徑(要重開終端機才生效)。

> 💡 **企業 Windows / 防毒攔截大檔下載**:第一次跑會從 HF CDN 下 ~10 GB(ASR 1.2 + polish 8)。防毒可能擋掉或拖慢。如果中途 timeout,可先用 `huggingface-cli download Qwen/Qwen3-ASR-0.6B Qwen/Qwen3-4B-Instruct-2507` 預下載完整模型,再啟動 daemon。

> 💡 **存放空間估算**:CUDA torch wheel(~2.5 GB)+ Qwen3-ASR-0.6B(~1.2 GB)+ Qwen3-4B-Instruct-2507(~8 GB bf16)+ transformers cache 等大約合計 15-20 GB。確認 `C:`(或 HF_HOME 指到的槽)有 ≥ 25 GB 空間。

`numpy` 通常會被 `torch` 一起拉進來;如果沒有:
```powershell
pip install --user numpy
```

### macOS Apple Silicon 加速（**只 Apple Silicon 需要**）

| 套件 | 用途 |
|------|------|
| `mlx-qwen3-asr` | **預設 ASR backend（v0.3.0+）** — Qwen3-ASR via Apple MLX（Metal 加速）。中文標點 + 中英混合表現比 Whisper turbo 強 |
| `mlx-whisper` | 可選 ASR — Whisper large-v3-turbo via Apple MLX，v0.2.x 的舊預設，仍可切回去 |
| `mlx-lm` | **預設 polish 後處理 backend（v0.5.0+）** — 跑小 LLM 移除口語贅字 + 修口誤。可關 (`POLISH_ENABLED = False`) |

> ⚠️ 這兩個都只在 Apple Silicon（M1 以上）可用。Intel Mac 從 v0.4.0 起不支援(daemon 啟動會拒絕)。

### macOS 一鍵安裝(Apple Silicon)

```bash
pip install \
    faster-whisper \
    sounddevice \
    pynput \
    opencc-python-reimplemented \
    mlx-qwen3-asr \
    mlx-whisper \
    mlx-lm
```

> `mlx-whisper` 不是必裝，只是想留個切換選項才裝。最小集合可拿掉它，daemon 預設只用 `mlx-qwen3-asr`。`faster-whisper` 也不是預設需要的(macOS 預設 backend 是 qwen3-asr),但留著當 CPU fallback / debug 工具。`mlx-lm` 是 v0.5.0+ 的 polish 後處理用的;不裝的話 daemon 自動 fall back 到 NoopPolisher,直接貼原始 ASR 輸出,行為等同 v0.4.x。

### Linux 安裝（規劃中）

僅裝跨平台核心即可,有 NVIDIA GPU 時可加裝 `nvidia-cudnn-cu12` 等（同 Windows 路徑）。詳見 [Roadmap](#roadmap)。

---

## 依硬體選擇 Preset

v0.6.0+ 起預設跑 **Qwen3-ASR + LLM polish** 雙模型在 GPU/Metal 上(macOS + Windows 一致);沒這麼多 VRAM 或不想裝 PyTorch CUDA 的人,有三個降階 tier 可選:

| Preset | `STT_BACKEND` | `STT_MODEL` | `POLISH_MODEL` | Disk(累計) | VRAM(常駐) | GPU 延遲 | 品質 | 適用硬體 |
|--------|---------------|-------------|-----------------|------------|------------|---------|------|---------|
| **Maximum** ⭐(v0.6.0+ 預設) | `qwen3-asr` | `Qwen/Qwen3-ASR-0.6B` | `Qwen3-4B-Instruct-2507`(Win:HF bf16 / Mac:MLX 4-bit)✅ | 1.2 + 8 GB | **~10 GB** | ~0.3-0.6 s + polish ~0.5-1 s | **最高** — 中文標點 + 中英 SOTA、文字最乾淨 | NVIDIA ≥ 12 GB(RTX 4070 / 4080 / 4090)**或** Apple Silicon(16 GB+ unified) |
| **Balanced** | `qwen3-asr` | `Qwen/Qwen3-ASR-0.6B` | `Qwen/Qwen2.5-1.5B-Instruct` ✅ | 1.2 + 3 GB | ~5 GB | ~0.3-0.6 s + polish ~0.2-0.4 s | 高 — polish 模型較小,贅字清得不夠乾淨但仍可用 | NVIDIA 6-10 GB(RTX 3060 / 4060) |
| **Light** | `qwen3-asr` | `Qwen/Qwen3-ASR-0.6B` | ❌ `POLISH_ENABLED = False` | 1.2 GB | ~2 GB | ~0.3-0.5 s(無 polish) | 中等 — raw ASR、口語贅字會直接貼出 | NVIDIA 4-6 GB(RTX 3050 / 2060) |
| **Mini**(v0.4.x 風格 fallback) | `faster-whisper` | `large-v3-turbo` / `medium` / `small` / `base` | ❌ | 75 MB – 1.5 GB | 250 MB – 2 GB(GPU)/ 180 MB – 1.5 GB(CPU) | < 0.1 s – 0.5 s(GPU)/ 0.5 – 12 s(CPU) | 中等 — Whisper turbo,英文細節稍弱於 Qwen3-ASR | 無獨顯 / GPU < 4 GB / 不想裝 PyTorch CUDA;CPU 延遲依模型大小 |

### 注意

- **數字是估算範圍**,實際依模型版本、CUDA / Metal、量化精度、GPU 世代有變動
- **「品質」針對中英混合場景**:Maximum 在中文標點 + 中英 code-switching 上表現最強(LLM-backbone 加 polish);Balanced polish 模型較小,贅字清得不像 4B 那麼乾淨;Light = raw ASR、口語感保留;Mini Whisper turbo 在英文細節有時會犯錯
- **首次啟動會下載模型**到 `~/.cache/huggingface/`(或 `HF_HOME`),下載完就一直放著;切 preset 也不會刪舊模型,所以**換來換去都很快**
- **VRAM 不夠會 OOM**:Maximum tier 載入時如果 VRAM 不夠,`build_polisher` 會 catch OOM、退回 NoopPolisher(daemon 不會 crash),log 會印「polish disabled — CUDA OOM」+ 建議改成 Balanced tier 或 `POLISH_ENABLED = False`
- **Mini tier 路徑**:用 v0.5.0 以前的 `faster-whisper`,不裝 PyTorch CUDA 也能跑(只需 `nvidia-cudnn-cu12` / `nvidia-cublas-cu12`)。Mini tier 的 `STT_MODEL` 可在 `large-v3-turbo`(最強、~2 GB VRAM)/ `medium`(~1.5 GB)/ `small`(~600 MB)/ `base`(~250 MB)之間挑

### 怎麼切換

編輯 `scripts/stt-daemon.py`,找到 Config 區的 `STT_BACKEND` / `STT_MODEL` / `POLISH_ENABLED` / `POLISH_MODEL` 四行(line ~108–138),按照上表挑你硬體對應的 tier 改:

```python
# 例如:Balanced tier(中階卡 RTX 3060 / 4060)
STT_BACKEND      = "qwen3-asr"
STT_MODEL        = "Qwen/Qwen3-ASR-0.6B"
POLISH_ENABLED   = True
POLISH_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"

# 例如:Light tier(入門卡 RTX 3050 / 2060)
STT_BACKEND      = "qwen3-asr"
STT_MODEL        = "Qwen/Qwen3-ASR-0.6B"
POLISH_ENABLED   = False

# 例如:Mini tier(沒裝 PyTorch CUDA / 想極省資源)
STT_BACKEND      = "faster-whisper"
STT_MODEL        = "large-v3-turbo"   # 或 medium / small / base
POLISH_ENABLED   = False
```

然後 stop + start:

Windows：
```powershell
.\scripts\stt-stop.ps1
.\scripts\stt-start.ps1
```

macOS：
```bash
bash scripts/stt-stop.sh
bash scripts/stt-start.sh
```

啟動 log 會印出 `model: <你選的>`，第一次跑會下載新模型（幾百 MB 到 1.5 GB 不等）。

---

## Windows 第一次啟用

### 1. Clone

```powershell
git clone https://github.com/DennisLiuCk/home-stt.git
cd home-stt
```

腳本都在 `scripts/`：

```
scripts/
├── stt-daemon.py         # daemon 主程式
├── stt_platform*.py      # 平台抽象（Pasteboard）
├── stt-start.ps1         # Windows 啟動
└── stt-stop.ps1          # Windows 停止
```

### 2. （可選）依硬體挑 Preset

v0.6.0 預設 **Maximum tier** — `Qwen3-ASR-0.6B` + `Qwen3-4B-Instruct-2507` polish,雙模型常駐 ~10 GB VRAM。**RTX 4070 / 4080 / 4090 跳過這步直接 (3)**。

如果你的卡 < 10 GB VRAM(RTX 3060 / 4060 / 筆電 GPU)、或不想裝 PyTorch CUDA,先去 [依硬體選擇 Preset](#依硬體選擇-preset) 挑 Balanced / Light / Mini tier,編輯 `scripts/stt-daemon.py` 把 `STT_BACKEND` / `STT_MODEL` / `POLISH_ENABLED` / `POLISH_MODEL` 改掉再啟動。

### 3. 啟動 daemon

```powershell
.\scripts\stt-start.ps1
```

第一次跑會自動下載模型(Maximum tier ~10 GB:Qwen3-ASR-0.6B 1.2 GB + Qwen3-4B-Instruct-2507 ~8 GB)到 `~\.cache\huggingface\`(或 `HF_HOME` 指定的位置),要等 1-5 分鐘看網速。後續啟動 model load + CUDA warmup 約 15-30 秒(Maximum tier 兩個模型都要載)。

成功會看到:
```
STT daemon started (PID 8232).
Log: C:\Users\<name>\AppData\Local\Temp\stt-daemon.log
Allow ~30s for model load + GPU warmup before first trigger key.
```

確認 log 開頭兩個關鍵字段:
```
[stt] backend: qwen3-asr | model: Qwen/Qwen3-ASR-0.6B
[stt] polish: Qwen/Qwen3-4B-Instruct-2507 (PyTorch bfloat16 @ NVIDIA <GPU 名>, ≤256 tok)
```
**polish 那行不應該是** `disabled (raw ASR output)` — 如果是,看 stderr log 的 `[stt] polish disabled — ...` 訊息(會告訴你是 import 失敗、CUDA OOM、還是其他原因)。

### 4. 使用

1. 把焦點放在任何想輸入文字的視窗
2. 按住 **Right Alt** 或 **Right Ctrl**（兩者皆可）→ 聽到「叮」表示開始錄
3. 對麥克風講話（中英文都可）
4. 放開觸發鍵
5. 約 0.2–1 秒後文字自動出現，並聽到「咚」表示完成

> 為什麼兩個鍵都綁？Right Alt 在某些應用（例如 Chrome）會跟既有快捷鍵衝突，Right Ctrl 是備援。**按下哪個就放開哪個**；同時按只有先按下的那個有效。

### 5. 確認狀態

看 log：
```powershell
Get-Content "$env:TEMP\stt-daemon.log" -Encoding utf8 -Tail 20
```

每次說話 daemon 會 log 一行：
```
[stt] zh 0.21s -> 幫我 review 這個 Python function
```

---

## macOS 第一次啟用

### 1. Clone

```bash
git clone https://github.com/DennisLiuCk/home-stt.git
cd home-stt
```

### 2. 安裝套件

見 [macOS 一鍵安裝](#macos-一鍵安裝)。Apple Silicon 用戶會多裝 `mlx-qwen3-asr`(預設,v0.3.0+),選裝 `mlx-whisper`(舊預設,要切換才用得到)。

### 3. 授權三個權限（最容易忽略的一步）

macOS 對「全域聽鍵 + 發鍵 + 錄音」這三件事要分別授權,而且**要授權給你的 Python binary（不是 pyenv shim）**。先找出真實路徑:

```bash
python3 -c "import sys; print(sys.executable)"
# 例如: /Users/<you>/.pyenv/versions/3.11.11/bin/python3
# (這是 symlink → python3.11,授給任一個都行,系統會解析)
```

> 為什麼不用 `which python3` 或 `readlink -f`? pyenv shim 是個 Python 包裝腳本(不是 symlink),那些指令會回 shim 路徑(`~/.pyenv/shims/python3`),但 macOS 權限要綁的是實際執行 binary。`sys.executable` 才會給你對的答案。

打開「系統設定 → 隱私權與安全性」,把上面那個路徑加進這三個項目：

| 項目 | 授給誰 | 為什麼 |
|------|--------|--------|
| **輸入裝置監控**（Input Monitoring） | Python binary | pynput 全域 listener 才能聽到 Right Option 被按下 |
| **輔助使用**（Accessibility） | Python binary | 走 Quartz CGEvent 模擬 Cmd+V — **繞過 IME**,中文輸入法開著也能貼 |
| **輔助使用**（Accessibility） | System Events / osascript | Python 沒授權時的 fallback paste 路徑(IME 開啟時可能被攔截) |
| **麥克風**（Microphone） | Python binary | sounddevice 才能讀麥克風 |

> 💡 **Python 加進輔助使用 = 中文輸入法相容**:v0.2.1 起 macOS paste 走兩條路徑 — Python 有 Accessibility 走 Quartz CGEvent(post-IME tap,中文/日文/韓文 IME 都吞不到);沒授權則 fallback 到 osascript,日常文字能貼但中文 IME 開著時 Cmd+V 可能被攔。**強烈建議把 Python binary 也加進輔助使用**,雙保險。daemon 啟動 log 會印出當前 paste path:`[stt] paste path: Quartz CGEvent ...` 或 `osascript ...`。

每加完一個項目,系統可能要求 daemon 重啟才會吃到新權限。

> 💡 第一次跑 daemon 時,macOS 可能會跳出對話框問你要不要授權麥克風 —— 按「允許」即可。Input Monitoring / Accessibility 通常需要你主動去設定加。

### 4. （可選）依硬體挑 Preset

Apple Silicon 預設跑 **Qwen3-ASR-0.6B** on MLX,延遲約 0.3-0.5 秒,中文標點 + 中英混合表現比 Whisper turbo 強。想試 1.7B 把 `STT_MODEL` 改成 `"1.7B"`;想切回 Whisper 把 `STT_BACKEND` 改成 `"mlx-whisper"`,然後去 [依硬體選擇 Preset](#依硬體選擇-preset) 改 `STT_MODEL`。

### 5. 啟動 daemon

```bash
bash scripts/stt-start.sh
```

第一次跑會下載預設模型(Qwen3-ASR-0.6B 約 1.2 GB,或切到 mlx-whisper 的 large-v3-turbo 約 1.5 GB)到 `~/.cache/huggingface/`,要等 30 秒到幾分鐘。後續啟動約 1-10 秒(model load + Metal warmup)。

成功會看到：
```
STT daemon started (PID 12345).
Log: /var/folders/.../T/stt-daemon.log
Allow ~10-30s for model load + Metal warmup before first trigger key.
```

### 6. 使用

1. 把焦點放在任何想輸入文字的視窗（Notes / TextEdit / iTerm / VSCode / Slack...）
2. 按住 **Right Option** → 聽到「叮」表示開始錄
3. 對麥克風講話（中英文都可）
4. 放開 Right Option
5. 約 0.3–1 秒後文字自動出現,並聽到「咚」表示完成

### 7. 確認狀態

看 log（`$TMPDIR` 通常是 `/var/folders/.../T/`）：
```bash
tail -n 20 "$TMPDIR/stt-daemon.log"
```

每次說話 daemon 會 log 一行（跟 Windows 相同格式）：
```
[stt] zh 0.34s -> 幫我 review 這個 Python function
```

---

## 停止

Windows：
```powershell
.\scripts\stt-stop.ps1
```

macOS：
```bash
bash scripts/stt-stop.sh
```

停止後觸發鍵就回到原本的功能（Windows：Right Alt / Right Ctrl；macOS：Right Option）。

---

## 開機自動啟動（選用）

### Windows

把 `stt-start.ps1` 的捷徑放進 Windows 啟動資料夾：

1. `Win+R` → 輸入 `shell:startup` → Enter
2. 建立 `stt-start.lnk`，目標（把路徑換成你 clone 的位置）：
   ```
   powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\path\to\home-stt\scripts\stt-start.ps1"
   ```

### macOS

可以做 LaunchAgent,但要考量到三個權限只能授給特定的 binary,如果之後切 pyenv 版本路徑會變。

最簡單的做法：把 `bash /path/to/home-stt/scripts/stt-start.sh` 加進「系統設定 → 一般 → 登入項目與擴充功能 → 開啟登入」(用 `Automator` 包成 Application,或寫個 `.command` 拖進去)。日後如果需要正式 LaunchAgent,可以放在 `~/Library/LaunchAgents/com.homestt.daemon.plist`。

---

## 自訂

直接編輯 `stt-daemon.py` 頂部的 `Config` 區塊：

```python
SAMPLE_RATE      = 16000                   # 麥克風取樣率
MIN_AUDIO_SEC    = 0.3                     # 太短的按鍵自動忽略
STT_BACKEND      = _DEFAULT_BACKEND        # 自動:Apple Silicon → qwen3-asr (v0.3.0+);Win/Linux → qwen3-asr (v0.6.0+)
STT_MODEL        = _DEFAULT_MODEL          # 自動:皆為 "Qwen/Qwen3-ASR-0.6B"
TRIGGER_KEYS     = None                    # None = 平台預設(Win: alt_gr+ctrl_r,Mac: alt_r);改 set 可覆蓋

# Polish 後處理(v0.5.0+;Windows v0.6.0+;用小 LLM 修飾 ASR 輸出,去口語贅字)
POLISH_ENABLED   = True                    # False = 跳過 polish,直接貼原始 ASR(行為等同 v0.4.x)
POLISH_MODEL     = _DEFAULT_POLISH_MODEL   # 自動:Mac → MLX 4-bit 變體 / Win/Linux → "Qwen/Qwen3-4B-Instruct-2507"
POLISH_LANGUAGES = {"zh", "ja", "ko"}      # 只 polish CJK;純英文 bypass(小 LLM 容易誤翻英文)
POLISH_PROMPT    = "..."                   # 移除「呃、嗯、就是、那個、然後」+ 修口誤的 system prompt

# 提示音
BEEPS_ENABLED    = True                    # 想完全靜音設 False
BEEP_START_HZ    = 880                     # 按下觸發鍵時的「叮」
BEEP_END_HZ      = 660                     # 處理完貼上後的「咚」
BEEP_DURATION_MS = 80
BEEP_VOLUME      = 0.15                    # 0.0–1.0;太大聲會干擾 mic
```

改完用 stop + start 腳本重啟（Windows: `.ps1`,macOS: `.sh`）。

> 切換模型大小（依硬體）見 [依硬體選擇 Preset](#依硬體選擇-preset)；切換到不同 STT 引擎見 [STT 模型抽象](#stt-模型抽象)。

### Polish 後處理 (v0.5.0+;Windows + macOS v0.6.0+)

ASR 跑完後可選一段 polish 後處理:走小型本地 LLM 修飾文字,去除口語贅字(呃、嗯、就是、那個、然後)、修正立即重複(「我我我覺得」→「我覺得」),保留說話原意 + 中英專有名詞。

**預設行為(雙平台)**:`POLISH_ENABLED = True`,模型 `Qwen3-4B-Instruct-2507`,只 polish CJK 語句(`zh`、`ja`、`ko`),純英文輸入 bypass。實作路徑依平台分流:

| 平台 | Polish backend | 預設模型 ID | 加速 | 模型大小 |
|------|---------------|------------|------|---------|
| **macOS Apple Silicon** | `MlxLocalLlmPolisher` (via `mlx-lm`) | `lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit` | Metal (4-bit 量化) | ~2.5 GB disk / ~4-5 GB RSS |
| **Windows / Linux** (v0.6.0+) | `TorchLocalLlmPolisher` (via `transformers`) | `Qwen/Qwen3-4B-Instruct-2507` | NVIDIA CUDA (bfloat16,未量化) | ~8 GB disk / ~8 GB VRAM |

**典型效果**:
```
[stt] zh 0.40s+polish 0.36s -> 我覺得這個 Python function 的設計可以再優化一下
                              ↑ 原始 ASR:「呃我我我覺得這個 Python function 的設計,嗯,可以再優化一下」
```

**記憶體 / VRAM 估算**:
- macOS:daemon 總 RSS peak ~4.5 GB(Qwen3-ASR 0.6B + Qwen3-4B-Instruct 4bit + Python overhead)。16 GB Mac 舒服;**8 GB Mac 建議 `POLISH_ENABLED = False`**(回到 ~1.5-2 GB)
- Windows:VRAM ~10 GB 常駐(Qwen3-ASR 1.5 + polish 8 GB);RTX 4070 / 4080 / 4090 舒服。**< 10 GB VRAM 的卡**(RTX 3060 / 4060 / 筆電 GPU)請看 [依硬體選擇 Preset](#依硬體選擇-preset) Balanced / Light tier

**Polish 失敗 fallback**:套件沒裝(`mlx-lm` 或 `transformers`)、模型載入 OOM、generate exception,daemon 印一行 warning 後自動退到 `NoopPolisher`(原樣輸出),不會 crash。v0.6.0+ 起 `build_polisher` 會依錯誤類別印不同提示:
- **ImportError** → 「請按 README Windows 安裝步驟裝 torch+CUDA 跟 qwen-asr,或設 POLISH_ENABLED = False」
- **CUDA OOM** → 「建議改用 Qwen/Qwen2.5-1.5B-Instruct(~3 GB VRAM),或設 POLISH_ENABLED = False」

**換更小的 polish 模型**:

```python
# macOS 8 GB Mac:
# ~1.8 GB on disk,~2.8 GB RSS peak,品質中等(偶爾誤翻英文邊角 case)
POLISH_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Windows < 10 GB VRAM:
# ~3 GB VRAM bf16,品質略弱但仍可移除主要贅字
POLISH_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

> ⚠️ Qwen3.5 系列(0.8B / 2B / 4B)目前**不適合**做 polish — 它們預設 thinking 模式會吐 chain-of-thought trace,把 max_tokens 用光也沒寫到 polish 結果。Qwen3-Instruct-2507 才是純指令跟隨變體。

### 提示音說明

預設兩個音：
- **按下觸發鍵** → 880 Hz / 80 ms 「叮」（清亮上揚），代表「開始聽你說」
- **處理完成貼上** → 660 Hz / 120 ms 「咚」（較柔和），代表「文字已貼入」

提示音用 `sounddevice` 動態生成正弦波（不依賴音檔），跨平台 zero-dependency。音量預設低（0.15）避免被麥克風收進去影響 transcription。完全不想要設 `BEEPS_ENABLED = False`。

---

## 疑難排解

### 通用

**中文是簡體**
- 確認 `opencc-python-reimplemented` 已安裝
- 看 log 是否有 OpenCC 相關錯誤

**文字貼到不對的視窗**
- 焦點問題 — 你放開觸發鍵前必須先把焦點放到目標輸入框
- 不要按完馬上切視窗

### Windows

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

**Daemon 死掉**
```powershell
Get-Content "$env:TEMP\stt-daemon.err.log" -Encoding utf8 -Tail 30
```

### macOS

**按 Right Option 沒任何反應(沒「叮」聲、log 沒 REC 行)**
- 大概率是「輸入裝置監控(Input Monitoring)」沒給 — 沒給的話 pynput listener 完全收不到鍵
- 去「系統設定 → 隱私權與安全性 → 輸入裝置監控」確認你的 Python binary(`python3 -c "import sys; print(sys.executable)"` 取得的真實路徑)在列表裡且開關打開
- 改完設定後重啟 daemon:`bash scripts/stt-stop.sh && bash scripts/stt-start.sh`

**有「叮」聲但放開後沒文字貼出來**
- 「輔助使用(Accessibility)」沒給 — daemon 收得到鍵但發不出 Cmd+V
- log 通常看得到 `[stt] zh ...s -> ...`(代表 STT 跑完了),但畫面沒文字 = 卡在 paste
- 同樣去「系統設定 → 隱私權與安全性 → 輔助使用」加 Python binary

**麥克風完全收不到聲音 / log 出現 silence**
- 第一次跑時 macOS 應該會跳對話框問麥克風授權,如果沒跳或不小心按了拒絕,去「系統設定 → 隱私權與安全性 → 麥克風」開
- 確認系統麥克風輸入裝置有選到對的(內建/外接)

**mlx-whisper 載入失敗 / 沒 Metal 加速**
- log 出現 `Unknown STT backend: 'mlx-whisper'` → 沒裝 `mlx-whisper`,跑 `pip install mlx-whisper`
- v0.4.0 起 Intel Mac 不再支援,daemon 啟動會 SystemExit。要 Intel Mac 請降版到 v0.3.0

**Daemon 死掉**
```bash
tail -n 30 "$TMPDIR/stt-daemon.err.log"
```

**pyenv 切換 Python 版本後權限失效**
- macOS 權限是綁特定 binary 路徑,你切到別的 pyenv version 等於換 binary
- 解法:把新版本的 binary 重新加進三個權限項目 — 或者固定一個版本就不要切

---

## 檔案結構

```
home-stt/
├── README.md
├── .gitignore
└── scripts/
    ├── stt-daemon.py        # 主程式:keyboard hook + audio + STT backends + state machine
    ├── stt_platform.py      # Pasteboard ABC + build_pasteboard() factory dispatch
    ├── stt_platform_win.py  # WindowsPasteboard:ctypes SendInput + PowerShell clipboard + NVIDIA DLL
    ├── stt_platform_mac.py  # MacOSPasteboard:pbcopy + pynput Cmd+V
    ├── stt-start.ps1        # Windows 啟動:背景 spawn python,寫 PID file
    ├── stt-stop.ps1         # Windows 停止:讀 PID file 或掃 process tree 殺
    ├── stt-start.sh         # macOS 啟動:nohup 背景跑,寫 PID file
    ├── stt-stop.sh          # macOS 停止:讀 PID file + pgrep fallback
    └── stt-daemon.pid       # daemon PID(runtime 產生,已 gitignore)

Windows %TEMP%\ 或 macOS $TMPDIR (/var/folders/.../T/)
├── stt-daemon.log       # 主 log(包含 transcription)
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
| Windows-only `ctypes.WinDLL("user32")` 在 macOS import 直接炸 | 抽到 `stt_platform_win.py`,只在 `sys.platform == "win32"` 時 lazy import |
| macOS 全域聽鍵需要 Input Monitoring,模擬發鍵需要 Accessibility | daemon 啟動時就需要這兩個權限授給 Python binary;pyenv shim 路徑會綁錯,必須授實際 binary(`readlink -f $(which python3)`) |
| macOS mlx-whisper 第一次跑要從 HF 下載 ~1.5GB | 跟 faster-whisper 一樣會快取在 `~/.cache/huggingface/`,後續啟動秒級 |
| macOS pynput Controller 模擬 Cmd+V 在某些 pyenv 設定下被 Accessibility silent drop(文字停在剪貼簿但沒貼出) | 改走 `osascript -e 'tell ... keystroke "v" using command down'` — 權限綁系統 binary 而非 Python,跨 pyenv 版本穩定 |
| macOS beep 用 16kHz 取樣率送 sd.play 在 48kHz 輸出裝置上 resample 產生「叮叮」破碎感 | 啟動時用 `sd.query_devices(kind='output')['default_samplerate']` 取得原生取樣率、改用 raised-cosine fade、前面墊 5ms 靜音吸收開 stream 的 click |

---

## 跨平台設計

daemon **核心管線 100% 跨平台**：

```
[mic] sounddevice
  → STTBackend (qwen3-asr default on Apple Silicon v0.3.0+ AND Windows/Linux v0.6.0+; faster-whisper / mlx-whisper switchable)
  → OpenCC s2tw  +  regex CJK/ASCII spacing
  → TextPostProcessor.polish() (optional, v0.5.0+ macOS / v0.6.0+ Win/Linux; gated on detected language)
  → Pasteboard.set_text() + Pasteboard.paste()   ← ★ 唯一綁平台的薄層
```

平台特定的三件事從 v0.2.0 起抽到 `Pasteboard` 介面(`scripts/stt_platform.py`),`build_pasteboard()` 依 `sys.platform` lazy-import 對應實作模組:

| 抽象層 | Windows(`stt_platform_win.py`) | macOS(`stt_platform_mac.py`) | Linux(規劃) |
|--------|------------------------------|----------------------------|----------|
| **Clipboard 寫入** | PowerShell `Set-Clipboard`(UTF-8 強制) | `pbcopy` 子程序 | `xclip` (X11) / `wl-copy` (Wayland) |
| **Paste 模擬** | ctypes `SendInput` Ctrl+V(IME-proof) | `osascript` 透過 System Events 發 Cmd+V(Accessibility 綁系統 binary,不會因 pyenv 切版漂移) | `xdotool key ctrl+v` (X11) / `ydotool` 或 `wtype` (Wayland) |
| **預設觸發鍵** | `{Key.alt_gr, Key.ctrl_r}` | `{Key.alt_r}` (Right Option) | 預定 `{Key.alt_r}` |
| **Native lib 註冊** | NVIDIA cuDNN/cuBLAS DLL 路徑 | n/a(回 0) | NVIDIA(同 Win) |

加新平台:在 `scripts/` 開一個 `stt_platform_<os>.py`、實作 `Pasteboard` 子類、在 `stt_platform.py:build_pasteboard()` 加一個 `if sys.platform == "..."` 分支即可,`stt-daemon.py` 不用動。

```python
# scripts/stt_platform.py
def build_pasteboard() -> Pasteboard:
    if sys.platform == "win32":
        from stt_platform_win import WindowsPasteboard
        return WindowsPasteboard()
    if sys.platform == "darwin":
        from stt_platform_mac import MacOSPasteboard
        return MacOSPasteboard()
    raise NotImplementedError(...)
```

> 重點:**lazy import**。`stt_platform_win.py` 在 module-level 用了 `ctypes.WinDLL("user32")`,如果在 macOS 上直接 import 會炸 — 所以 `build_pasteboard()` 只在跑到對應 branch 時才 import,跨平台 import 安全。

STT 後端走完全相同結構(見 [STT 模型抽象](#stt-模型抽象))。**v0.6.0 起兩平台預設皆是 Qwen3-ASR-0.6B**:macOS Apple Silicon 走 `mlx-qwen3-asr`(MLX/Metal);Windows / Linux 走 `qwen-asr`(PyTorch + transformers + CUDA bfloat16)。v0.5.0 以前的 Windows 路徑(`faster-whisper` + CTranslate2 + cuDNN)保留為可選 fallback。Polish 後處理同一個模式:macOS 走 `mlx-lm`、Win/Linux 走 `transformers`,模型 ID 都是 Qwen3-4B-Instruct-2507(僅量化精度不同)。Intel Mac 從 v0.4.0 起不再支援。

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
    # Whisper large-v3-turbo via CTranslate2 — Win/Linux CUDA + CPU fallback

class MlxWhisperBackend(STTBackend):
    # Whisper large-v3-turbo via Apple MLX — Apple Silicon Metal 原生

class Qwen3AsrBackend(STTBackend):
    # Alibaba Qwen3-ASR (0.6B / 1.7B) via mlx-qwen3-asr — Apple Silicon Metal
    # 原生,中文標點 + 中英混合表現比 Whisper turbo 強(v0.3.0+ 預設)

def build_backend(name: str, model: str) -> STTBackend:
    if name == "faster-whisper":
        return FasterWhisperBackend(model)
    if name == "mlx-whisper":
        return MlxWhisperBackend(model)
    if name == "qwen3-asr":
        return Qwen3AsrBackend(model)
    # elif name == "sense-voice":    ← 規劃
    # elif name == "paraformer":     ← 規劃
    raise ValueError(...)
```

切換只需改頂部兩個常數(或讓平台自動選):

```python
# 自動:Apple Silicon → qwen3-asr (v0.3.0+),其餘 → faster-whisper
STT_BACKEND = _DEFAULT_BACKEND      # 或硬編碼 "faster-whisper" / "mlx-whisper" / "qwen3-asr"
STT_MODEL   = _DEFAULT_MODEL        # 對應 platform 的 default model 字串
```

主流程 (`_transcribe_and_emit`) 只跟介面對話，**換引擎不影響 mic 收音、post-processing、clipboard/paste 任何邏輯**。

### 已實作 vs 規劃中的後端

| 後端 | 引擎 | 主要強項 | 狀態 |
|------|------|---------|-----|
| `qwen3-asr` | Alibaba Qwen3-ASR 0.6B / 1.7B — Mac via Apple MLX,Win/Linux via `qwen-asr`(PyTorch + transformers + CUDA bfloat16) | **中文標點 + 中英 code-switching SOTA**,52 語 + 22 中文方言,中文場景比 Whisper 強 | ✅ **預設(Apple Silicon v0.3.0+ / Windows + Linux v0.6.0+)** |
| `faster-whisper` | Whisper large-v3-turbo via CTranslate2 | 中英混合強,99 語、CUDA float16 / CPU int8;v0.6.0 起在 Win/Linux 降為 fallback(不裝 PyTorch CUDA 時使用) | ✅ 可選(Win/Linux fallback) |
| `mlx-whisper` | Whisper large-v3-turbo via Apple MLX | Apple Silicon Whisper backend,中英混合穩、v0.2.x 預設 | ✅ 可選(Apple Silicon) |
| `sense-voice` | 阿里 FunASR SenseVoice-Small | 體積 234 MB、速度極快、含情感/事件偵測、5 語 | 🛣️ 規劃 |
| `paraformer` | 阿里 FunASR Paraformer-zh | **純中文 SOTA**(非自回歸) | 🛣️ 規劃 |

---

## Roadmap

### 平台支援

- [x] **Windows 10/11**
  - NVIDIA CUDA 加速、注音 IME 共存、繁中混英文、自動 spacing、雙 trigger key
- [x] **macOS(Apple Silicon M 系列)** — v0.2.0 起,v0.3.0 升 Qwen3-ASR 預設
  - Clipboard:`subprocess pbcopy`
  - Paste(v0.2.1+):Quartz CGEvent @ AnnotatedSessionEventTap(IME-safe;Python 有 Accessibility 才走這條),否則 fallback `osascript -e 'tell application "System Events" to keystroke "v" using command down'`
  - 觸發鍵:Right Option
  - 加速:`qwen3-asr` (Qwen3-ASR-0.6B via mlx-qwen3-asr) 預設(v0.3.0+);可切回 `mlx-whisper` (Whisper large-v3-turbo)
  - 系統權限:Python 要授 Input Monitoring + Microphone + Accessibility(IME-safe paste);System Events 也建議授 Accessibility(fallback 用)
  - Intel Mac:**v0.4.0 起不再支援**(daemon 啟動會 SystemExit,要用請 pin v0.3.0)
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
  - 中英混合強、CUDA float16 + CPU int8 fallback
- [x] **mlx-whisper**(Apple MLX 原生 Metal Whisper) — v0.2.0
  - macOS Apple Silicon 專用,跟 macOS 平台支援一起做
  - 同 `large-v3-turbo` 模型但跑在 Metal 上,turbo 延遲 ~0.3-0.5s
- [x] **qwen3-asr**(Alibaba Qwen3-ASR) — Apple MLX v0.3.0 / PyTorch CUDA v0.6.0
  - macOS Apple Silicon 預設(v0.3.0,取代 mlx-whisper 當預設,後者仍可切換)
  - Windows / Linux 預設(v0.6.0,取代 faster-whisper 當預設,後者仍可切回)
  - 模型 Qwen3-ASR-0.6B(~1.2GB,fp16)或 Qwen3-ASR-1.7B(~3.4GB,更高精度)
  - **中文標點 + 中英 code-switching 比 Whisper turbo 強**,52 語 + 22 中文方言
  - 套件:`pip install mlx-qwen3-asr`(Mac)/ `pip install qwen-asr`(Win/Linux,需先裝 PyTorch CUDA wheel),皆 Apache-2.0 開源,離線跑
- [ ] **sense-voice**（阿里 FunASR SenseVoice-Small）
  - 234 MB 體積、含情感 / 事件偵測、多語但同句內單一語言
  - 套件：`pip install funasr modelscope`
  - 待評估：中英混合在同句的表現是否能接受
- [ ] **paraformer**（阿里 FunASR Paraformer-zh）
  - 純中文場景的精度標竿、非自回歸架構速度極快
  - 同上 FunASR 套件
  - 適用：純中文場景比 Whisper 強，但英文/中英混合會崩

### 介面化重構

- [x] **`STTBackend` 抽象** — 已抽出,換引擎只改 `STT_BACKEND` 常數 + 加一個 class
- [x] **`Pasteboard` 抽象** — v0.2.0 隨 macOS 支援一起完成。Clipboard 寫入 / Paste 模擬 / 觸發鍵代號全部抽到 `scripts/stt_platform*.py`,加新平台只要實作 `Pasteboard` 子類即可
- [x] **`TextPostProcessor` 抽象** — v0.5.0 加進來,跑在 ASR 跟 paste 之間。三個 impl:`NoopPolisher`(disabled fallback)、`MlxLocalLlmPolisher`(Apple Silicon via mlx-lm,v0.5.0)、`TorchLocalLlmPolisher`(Win/Linux via transformers + CUDA bfloat16,v0.6.0)。`build_polisher` 依 `sys.platform` 自動 dispatch。介面 `polish(text) -> str`,未來要試 multi-modal polishing(例如 Qwen3-Omni)可擴展加 audio kwarg
