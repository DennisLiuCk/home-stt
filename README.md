# Hold-to-Talk STT

![version](https://img.shields.io/badge/version-0.7.5-blue) ![platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20(Apple%20Silicon)-lightgrey) ![python](https://img.shields.io/badge/python-3.10%2B-green)

**按住熱鍵講話、放開自動貼到當下焦點視窗,完全離線。** 兩個模式:語音輸入(Dictate)+ 語音編輯(Voice-Edit),詳見 [核心功能](#核心功能)。

> 📌 **macOS 用戶必讀**:第一次跑前要在「系統設定 → 隱私權與安全性」授權 3 個權限給 Python(輸入裝置監控 / 輔助使用 / 麥克風)。沒授權 daemon 不會 crash 但**按鍵完全沒反應** — 這是 #1 onboarding 掉坑點。先讀完 [macOS 第一次啟用 Step 3](#3-授權三個權限最容易忽略的一步) 再裝。

## 目錄

**🚀 新用戶從這開始**

- [核心功能](#核心功能) — Dictate + Voice-Edit 兩個模式
- [平台支援](#平台支援)
- [系統需求](#系統需求)
- [macOS 第一次啟用](#macos-第一次啟用) ⭐ Mac 新用戶這條
- [Windows 第一次啟用](#windows-第一次啟用)
- [停止](#停止)
- [疑難排解](#疑難排解)

**⚙️ 進階配置**

- [依硬體選擇 Preset](#依硬體選擇-preset) — 低 VRAM / 低 RAM 機降階
- [自訂](#自訂) — Config 變數說明
- [Polish 後處理](#polish-後處理-v050windows-macos-v060) — LLM 修飾文字機制
- [開機自動啟動（選用）](#開機自動啟動選用)

**📚 架構 / Reference**

- [檔案結構](#檔案結構)
- [跨平台設計](#跨平台設計)
- [STT 模型抽象](#stt-模型抽象)
- [測試](#測試)
- [Roadmap](#roadmap)

**📜 歷史紀錄(maintainer 取向)**

- [v0.7.x 效能與品質投資紀錄](#v07x-效能與品質投資紀錄)
- [已測試但放棄的優化方向](#已測試但放棄的優化方向別重複嘗試)
- [設計重點](#設計重點)

## 核心功能

兩個語音模式,**獨立 hotkey、互不干擾**:

### 🎤 1. 語音輸入 (Dictate)

> 按住觸發鍵 → 講話 → 放開 → 文字自動貼到當下焦點視窗

按住 **Right Option**(macOS)/ **Right Alt** 或 **Right Ctrl**(Windows)→ 對麥克風講話 → 放開 → 文字 ~0.3-2 秒後自動出現在當下焦點視窗。

```
[使用者] 按住 Right Option → 說「幫我 review 這個 Python function 的 async 部分」→ 放開
[daemon] 約 0.5 秒後,Notes / VSCode / iTerm 任何輸入框自動出現:
         「幫我 review 這個 Python function 的 async 部分」
```

中英文混合直接講、自動繁體、自動 CJK↔ASCII 補空格、口語贅字自動清掉。

### ✏️ 2. 語音編輯 (Voice-Edit, v0.7.5+)

> 選中文字 → 按住觸發鍵 → 講編輯指令 → 放開 → LLM 改寫選取並貼回去

在任何 app 選取一段文字,按住 **Right Command**(macOS)/ **F13**(Windows)→ 講編輯指令(「翻譯成英文」「改成正式語氣」「縮短一半」「整理成條列式」「改成過去式」)→ 放開 → 選取被改寫後的版本取代。

```
[使用者] 在 Notes 選一段中文段落 → 按住 Right Command
         → 說「翻譯成英文,保留技術名詞」→ 放開
[daemon] 約 1-2 秒後,選取的段落被英文版取代
```

完全離線,LLM 在本地跑(macOS:MLX 4-bit;Windows:PyTorch CUDA bfloat16)。

---

## 平台支援

| 平台 | 狀態 | 觸發鍵（預設） | STT backend / 加速 |
|------|------|--------------|------------------|
| **Windows 10 / 11** | ✅ 已實作、實測 | Right Alt (AltGr) **或** Right Ctrl | **Qwen3-ASR-0.6B via qwen-asr + LLM polish via transformers**（皆跑 NVIDIA CUDA bfloat16,預設 v0.6.0+）/ 也可切 `faster-whisper`(無 polish) |
| **macOS（Apple Silicon M 系列）** | ✅ 已實作 | Right Option | **Qwen3-ASR-0.6B via mlx-qwen3-asr**（Metal 原生,預設 v0.3.0+）/ 也可切 mlx-whisper |
| **Linux（X11 / Wayland）** | 🛣️ 規劃中 | 預定 Right Alt | 同 Windows 路徑(qwen-asr + transformers + CUDA);pasteboard 層尚未實作 |

> ⛔ **Intel Mac (darwin x86_64) 不再支援(v0.4.0+)** — daemon 啟動會 SystemExit 並提示降版到 v0.3.0。Intel 機已罕見,維護 + 文件成本不划算。要 Intel Mac 請 `git checkout v0.3.0`。

> 🏗️ **核心管線（麥克風 → Whisper → 文字後處理）100% 跨平台**。平台特定的薄薄三層 —— **clipboard 寫入**、**paste 模擬**、**全域熱鍵代號** —— 從 v0.2.0 起抽到 `Pasteboard` 介面實作。詳見 [跨平台設計](#跨平台設計) 段。

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
| `opencc-python-reimplemented` | 簡轉繁(s2twp,台灣正體 + 詞彙慣用) |

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

### macOS Apple Silicon 安裝

完整指令 + 三權限授權步驟一條 vertical scroll 走完,見 [macOS 第一次啟用](#macos-第一次啟用)。

### Linux 安裝（規劃中）

僅裝跨平台核心即可,有 NVIDIA GPU 時可加裝 `nvidia-cudnn-cu12` 等（同 Windows 路徑）。詳見 [Roadmap](#roadmap)。

---

## 依硬體選擇 Preset

v0.6.0+ 起預設跑 **Qwen3-ASR + LLM polish** 雙模型在 GPU/Metal 上(macOS + Windows 一致);沒這麼多 VRAM 或不想裝 PyTorch CUDA 的人,有三個降階 tier 可選:

| Preset | `STT_BACKEND` | `STT_MODEL` | `POLISH_MODEL` | Disk(累計) | VRAM(常駐) | GPU 延遲 | 品質 | 適用硬體 |
|--------|---------------|-------------|-----------------|------------|------------|---------|------|---------|
| **Maximum** ⭐(v0.7.0+ 預設) | `qwen3-asr` | `Qwen/Qwen3-ASR-0.6B` | `Qwen3-4B-Instruct-2507`(Win:HF bf16 / Mac:MLX 4-bit)✅ | 1.2 + 8 GB | **~10 GB** | ~0.3-0.6 s + polish ~0.7-2 s(v0.7.1 後)| **最高** — 中文標點 + 中英 SOTA、英文 keyword/identifier 保留率高 | NVIDIA ≥ 12 GB(RTX 4070 / 4080 / 4090)**或** Apple Silicon(16 GB+ unified) |
| **Balanced** | `qwen3-asr` | `Qwen/Qwen3-ASR-0.6B` | `Qwen/Qwen2.5-1.5B-Instruct` ⚠️ | 1.2 + 3 GB | ~5 GB | ~0.3-0.6 s + polish ~0.5-1 s | 中 — VRAM 妥協選項;**Qwen2.5-1.5B 對英文 keyword 跟事實 fidelity 較弱**,詳見 [v0.7.0 投資紀錄](#v07x-效能與品質投資紀錄) | NVIDIA 6-10 GB(RTX 3060 / 4060) |
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

然後 restart(雙平台同指令):

```bash
home-stt restart
```

或底層腳本(power user):

```powershell
# Windows
.\scripts\stt-stop.ps1
.\scripts\stt-start.ps1
```
```bash
# macOS
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
├── home_stt.py           # 統一 CLI（v0.7.6+,pip install -e . 安裝後變 `home-stt` 指令）
├── stt-start.ps1         # Windows 啟動
└── stt-stop.ps1          # Windows 停止
```

### 2. （可選）依硬體挑 Preset

v0.7.0+ 預設 **Maximum tier** — `Qwen3-ASR-0.6B` + `Qwen3-4B-Instruct-2507` polish,雙模型常駐 ~10 GB VRAM。**RTX 4070 / 4080 / 4090 跳過這步直接 (3)**。

如果你的卡 < 10 GB VRAM(RTX 3060 / 4060 / 筆電 GPU)、或不想裝 PyTorch CUDA,先去 [依硬體選擇 Preset](#依硬體選擇-preset) 挑 Balanced / Light / Mini tier,編輯 `scripts/stt-daemon.py` 把 `STT_BACKEND` / `STT_MODEL` / `POLISH_ENABLED` / `POLISH_MODEL` 改掉再啟動。

### 3. 啟動 daemon

**建議**先一次性安裝 `home-stt` CLI(跨目錄、跨平台統一指令):

```powershell
pip install -e .       # 在 home-stt repo 根目錄執行一次
home-stt start
```

之後 `home-stt {start,stop,restart,status,log,config}` 在任何目錄都能跑,雙平台語法一致。

> ⚠️ **Windows 安裝兩個小坑(踩過留紀念,別重複)**:
> 1. **`pip install -e .` 寫不進 `C:\Python<X>\Scripts`**:Python 裝在 all-users 路徑(`C:\Python312\` 等)時,Scripts 目錄預設沒寫權限,pip 會炸 `WARNING: Failed to write executable ... WinError 2` 然後 `ERROR: Could not install`(訊息誤導,實際是權限不足、不是檔案不見)。加 `--user`:`pip install -e . --user`;或用 venv 也行。
> 2. **`--user` 裝完 entry point 不在 PATH 上**:`home-stt.exe` 會放到 `%APPDATA%\Python\Python<X>\Scripts\`,該路徑**預設不在 PATH**,直接打 `home-stt` 找不到。永久加進去:`setx PATH "%PATH%;%APPDATA%\Python\Python312\Scripts"`(Python 版本對齊你裝的)後**重開終端機**才生效;或當下用 `& "$env:APPDATA\Python\Python312\Scripts\home-stt.exe" status` 全路徑跑。
>
> 不想 pip install 也可以,直接 `python scripts\home_stt.py {start,stop,status,...}` 跑同一個 CLI(下方 `.\scripts\stt-start.ps1` 是更底層、只負責 spawn daemon 的版本)。

也可以直接執行底層腳本(不想 pip install / 偏好顯式呼叫):

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

各套件用途:

| 套件 | 用途 | 必裝? |
|------|------|------|
| `mlx-qwen3-asr` | **預設 ASR backend(v0.3.0+)** — Qwen3-ASR via Apple MLX(Metal 加速),中文標點 + 中英混合 SOTA | ✅ 必裝 |
| `mlx-lm` | **預設 polish 後處理 backend(v0.5.0+)** — 跑小 LLM 移除口語贅字 + 修口誤 | ✅ 必裝(不裝則 polish 退到 NoopPolisher,等於 v0.4.x 行為) |
| `sounddevice` / `pynput` / `numpy` | 麥克風 / 全域 key hook / 音訊運算 | ✅ 必裝 |
| `opencc-python-reimplemented` | 簡轉繁(s2twp,台灣正體 + 詞彙慣用) | ✅ 必裝 |
| `mlx-whisper` | 可選 ASR — Whisper large-v3-turbo via Apple MLX,v0.2.x 舊預設,切回去才用得到 | ⚠️ 選裝 |
| `faster-whisper` | CPU fallback / debug 工具,macOS 預設 backend 不用 | ⚠️ 選裝 |

> ⚠️ 這些都只在 Apple Silicon(M1 以上)可用。Intel Mac 從 v0.4.0 起不支援(daemon 啟動會拒絕)。

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

**先看你 Mac 的記憶體**:

| Mac 配置 | 建議設定 | RAM 用量 (peak) |
|---------|---------|----------------|
| **M1/M2/M3/M4 + 16GB+** | **預設不動**(Maximum tier:ASR + LLM polish 全開) | ~4.5 GB |
| **M-Air / 任何 8GB 機** | `POLISH_ENABLED = False`(Light tier:只 ASR、無 polish) | ~1.5-2 GB |
| **M-Pro / M-Max / 32GB+** | 預設不動,可選升 `STT_MODEL = "1.7B"` 換更高 ASR 精度 | ~4.5-7 GB |

要改的話編輯 `scripts/stt-daemon.py` 頂部 Config 區。完整 4-tier 比較表(含 Windows / Linux 路徑)見 [依硬體選擇 Preset](#依硬體選擇-preset)。

### 5. 啟動 daemon

**建議**先一次性安裝 `home-stt` CLI(跨目錄、跨平台統一指令):

```bash
pip install -e .       # 在 home-stt repo 根目錄執行一次
home-stt start
```

之後 `home-stt {start,stop,restart,status,log,config}` 在任何目錄都能跑,雙平台語法一致。

也可以直接執行底層腳本(不想 pip install / 偏好顯式呼叫):

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

### 6. 使用 — 兩個模式

兩個 hotkey 互不干擾,**Right Option = 講新文字**,**Right Command = 改現有選取**。

#### 6a. 語音輸入(Dictate)— 按住 **Right Option**

1. 把焦點放在任何想輸入文字的視窗（Notes / TextEdit / iTerm / VSCode / Slack...）
2. 按住 **Right Option** → 聽到「叮」表示開始錄
3. 對麥克風講話（中英文都可）
4. 放開 Right Option
5. 約 0.3–1 秒後文字自動出現,並聽到「咚」表示完成

#### 6b. 語音編輯(Voice-Edit, v0.7.5+)— 按住 **Right Command**

1. 在任何 app **選取一段現有文字**(Notes / Mail / VSCode / Safari 編輯框 / iTerm 都可以)
2. 按住 **Right Command** → 聽到「叮」表示開始錄
3. 對麥克風講編輯指令,例如:
   - 「翻譯成英文」/ 「翻譯成中文」
   - 「改成正式語氣」/「改得口語一點」
   - 「縮短一半」/「展開成兩段」
   - 「整理成條列式」/「合併成一段」
   - 「改成過去式」/「改成命令句」
4. 放開 Right Command
5. 約 1-2 秒後選取被改寫後的版本取代,聽到「咚」表示完成

> 沒選取文字就按了 Right Command 會聽到 220 Hz 的「dull」失敗 beep — 表示 daemon 抓不到 selection,沒事發生。常見原因:該 app 不支援 Cmd+C 抓選取(影像檢視器 / 終端機輸出區)、或你忘了選取就按下去。

### 7. 確認狀態

裝了 `home-stt` CLI 後一個指令看完(PID / uptime / RSS / backend / polish / paste path / triggers / 最近 3 筆 transcribe):

```bash
home-stt status
```

範例輸出:
```
home-stt v0.7.5 -- running
  PID:      67289 (uptime 2h 5m)
  RSS:      2.77 GB
  log:      /var/folders/.../T/stt-daemon.log (last write 30m ago)
  err.log:  (clean)

  backend:  qwen3-asr (Qwen/Qwen3-ASR-0.6B)
  polish:   Qwen3-4B-Instruct-2507-MLX-4bit (MLX, <=512 tok)
  paste:    Quartz CGEvent @ AnnotatedSessionEventTap (IME-safe)
  triggers: hold Key.alt_r to dictate, hold Key.cmd_r to voice-edit
```

或直接看 log:
```bash
home-stt log               # 最後 30 行
home-stt log --tail 100    # 最後 100 行
home-stt log -f            # follow 模式(Ctrl+C 結束)
home-stt log --err         # err.log
```

每次說話 daemon 會 log 一行（跟 Windows 相同格式）：
```
[stt] zh 0.34s -> 幫我 review 這個 Python function
```

---

## 停止

```bash
home-stt stop
```

或底層腳本:
```powershell
# Windows
.\scripts\stt-stop.ps1
```
```bash
# macOS
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

**換更小的 polish 模型**(**只在 VRAM / RSS 真的不夠跑 4B 時用**):

```python
# macOS 8 GB Mac:
# ~1.8 GB on disk,~2.8 GB RSS peak,品質中等(偶爾誤翻英文邊角 case)
POLISH_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Windows < 10 GB VRAM:
# ~3 GB VRAM bf16,品質略弱但仍可移除主要贅字
POLISH_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

> ⚠️ **品質警告(v0.7.0 投資發現)**:Qwen2.5 系列在「最小編輯」任務上比 Qwen3 系列差很多 — 會把英文 keyword 亂改(`commit` → `push`)、改變事實(數字、語義反向)、翻譯英文片語、過度書面化。**不要當作「快速版」**,只在硬體裝不下 4B 時才用。詳見 [v0.7.x 效能與品質投資紀錄](#v07x-效能與品質投資紀錄)。

> ⚠️ Qwen3.5 系列(0.8B / 2B / 4B)目前**不適合**做 polish — 它們預設 thinking 模式會吐 chain-of-thought trace,把 max_tokens 用光也沒寫到 polish 結果。Qwen3-Instruct-2507 才是純指令跟隨變體。

### v0.7.x 效能與品質投資紀錄

完整工程紀錄抽到 [`CHANGELOG.md`](CHANGELOG.md)。每版做了什麼、為什麼、bench 結果、未來重啟條件全部寫在那。這裡只留摘要 + 延遲基準。

**實測延遲(Windows + RTX 5080,v0.7.1+)**:

| 場景 | 音訊長度 | ASR | Polish | 總等待 |
|------|---------|-----|--------|--------|
| 短(「好」「對啊就是這樣」)| 1-3s | ~0.5-1s | ~0.25s | **< 2s** |
| 中(技術討論一兩句)| 5-15s | ~2-3s | 0.7-1.0s | **3-5s** |
| 長(完整段落)| 20-30s | ~7-8s | ~2-3s | **10-13s** |
| 超長(連續講 40s)| 40s+ | ~10s | ~5s | **~15s** |

Mac M-series 跑同 stack 短中段約 1-2s(MLX 4-bit + Metal native,Python overhead 低於 PyTorch 路徑)。

**版本主題 highlights**:

| 版本 | 主題 | 影響 / 結果 |
|------|------|------------|
| **v0.7.0** | Polish model 升 Qwen3-4B-Instruct-2507 | 修 Qwen2.5-1.5B 翻譯英文 keyword、改主詞、語義反向、過度書面化等 4 類品質問題 |
| **v0.7.1** | Polish decode lossless 加速 -55% | PLD + 預 cache POLISH_PROMPT KV + cuDNN benchmark,長文 4.26s → 1.90s,quality byte-identical |
| **v0.7.2** | Correctness sweep + clipboard direct API | 修 3 個 correctness bug + ctypes/PyObjC 取代 subprocess(每次 paste 省 250-450 ms)+ dynamic max_tokens + silence trim + tests/CI |
| **v0.7.3** | Encoder pipelining framework(shipped DISABLED) | **Bench-first save**:plan 估省 50%,實測 3%,framework 留著但預設關 |
| **v0.7.4** | Polish prompt 標點保留修正 | Live-log discovered;三軸並進(正向約束 + 負向約束 + few-shot)修好多句中間「。」被刪 |
| **v0.7.5** | Voice-Edit 模式上線 | 第二個 trigger 熱鍵(Mac Right Command / Win F13),選取 + 講指令 → LLM 改寫選取 |

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

**長文 polish / 總等待感覺久(>5 秒)**
- 看 daemon log 的 `[stt] zh X.XXs+polish Y.YYs` 拆解 — ASR 通常才是大頭(40s 音訊 ASR 約 10s,polish 階段 v0.7.1 已優化到 ~2-5s for 100-300 字輸出)
- Polish 階段沒被浪費。長文等待的本質瓶頸 v0.7.3 bench 證實是 **decoder** (~95% post-release time、~6s for 200 token output)、不是 encoder (~0.2s for 40s audio)、也不是「按完才開始」(framework 已 ship、但實測收益 ~3% 不值 Lev=21 drift,見 [v0.7.3 投資紀錄](#v07x-效能與品質投資紀錄))。下一輪要在 decoder 動手 (llama.cpp + GGUF Q8_0 / FP8 / 等)
- 短期建議:長段 hold-to-talk 用較短語句的習慣,daily usage 體感「按完幾秒就出」

**啟動 log 跳 `expandable_segments not supported on this platform`**
- 預期警告 — Windows CUDA 不支援這個 allocator hint。設了沒效果但無害,留著為 cross-platform 一致(Linux 上會生效)

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

## 已測試但放棄的優化方向(別重複嘗試)

v0.7.x 系列做了大量加速 / 品質研究,以下路徑都 **bench-validated 失效** 或 **證實不適用 Windows + RTX 5080**。未來想繼續優化的人不要再走。詳見 [v0.7.x 效能與品質投資紀錄](#v07x-效能與品質投資紀錄) 各版本子段、`review-v0.7.1.md`、+ commit `8bb6157` (v0.7.1)。

### Polish 模型 / 量化

| 路徑 | 為何失效 |
|------|---------|
| **bitsandbytes NF4 INT4** | RTX 5080 + 小模型(<7B)上**慢 67%**(實測 long polish 4.3s → 9.1s)。per-layer dequant overhead 超過省下的 memory bandwidth。bnb 的 Blackwell sm_120 kernels 未 tune(bitsandbytes#1851 open 無 maintainer 回應)|
| **Qwen2.5-{0.5B, 1.5B}-Instruct 當主用 polish** | 品質不可靠:翻譯英文 keyword(`commit` → `push`)、改主詞(「幫我」→「幫你」)、刪 underscore identifier(`_USE_TORCH_COMPILE` → `USE_TORCH_COMPILE`)、語義反向(「更慢」→「更快」)。只在 VRAM 不夠時當 fallback |
| **Speculative decoding(Qwen3-0.6B 當 4B 的 draft)**| 對 4B 小目標,small draft 的 acceptance rate 太低,net negative(AWS benchmark 確認)。PLD 在這個任務上 strictly dominates |

### CUDA / PyTorch 加速路徑

| 路徑 | 為何失效 |
|------|---------|
| **torch.compile on Windows** | Inductor backend 需 `triton`,Windows 沒官方 wheel(`triton-windows` 社群 fork 已 archived 2026-02)。實測 0% 改善,只多 ~2s load |
| **flash-attn pre-built Windows wheel** | 沒對齊 `PyTorch 2.11 + cu128 + py 3.12` 的社群 wheel。要用必須 compile from source(留 v0.8.0 候選)|
| **GPU mel spectrogram patch (v0.7.2 嘗試)** | bench 實測只省 3 ms(研究 agent 預估 1-3s,**估錯 300x**)。WhisperFeatureExtractor 在 transformers 4.57+ 已是高效 CPU 實作。**bench-first 救了一命,差點 ship placebo** |

### ASR 替代 backend

| 路徑 | 為何失效 |
|------|---------|
| **faster-whisper INT8 on Blackwell** | sm_120 `CUBLAS_STATUS_NOT_SUPPORTED` 直接 crash。必須強制 `compute_type="float16"`(速度退一半,丟失 INT8 主要優勢)|
| **distil-whisper / NVIDIA Canary / Parakeet** | 英文 only / 無中文支援。我們是 80% 中文場景,不符合 |
| **TensorRT-LLM Windows native** | RTX 50 系列 Whisper 0.17 直接 crash,Blackwell 完整支援未到 |
| **CTranslate2 for Qwen3-ASR** | CTranslate2 沒 Qwen3-ASR arch support(只支援 Whisper / T5 / BERT)|
| **vLLM streaming ASR on Windows native** | vLLM sm_120 wheel issue,只能走 WSL2 + daemon 跨 boundary IPC(v0.8.0 候選但工程量大)|
| **官方 FP8 Qwen3-ASR** | Alibaba 沒 ship FP8 ASR checkpoint(polish 用的 Qwen3-4B-Instruct-2507-FP8 有,ASR 沒)|

### 架構 / UX

| 路徑 | 為何失效 |
|------|---------|
| **PLD / prefix-cache 套到 ASR** | ASR input 是 audio 沒文字可 lookup;每段 audio context 不同沒 stable prefix。Polish 兩招完全不能 port 過去 |
| **Hybrid 先預覽後替換(faster-whisper 預覽 + Qwen3-ASR 修正)** | Windows clipboard 一旦 user 開始編輯就無法 retroactively replace。需要 dedicated overlay UI 才能做,2-3 週工程,不是 hack |

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
| Whisper / Qwen3-ASR 預設輸出簡體中文 | OpenCC `s2twp` 自動簡轉繁台灣正體(含詞彙慣用,例如 异步→非同步、代码→程式碼) |
| 中文跟英文連在一起沒空格 | regex 自動在 CJK ↔ ASCII letter/digit 邊界補空格 |
| CTranslate2 找不到 cuDNN DLL | `pip install nvidia-cudnn-cu12` + 啟動時 `add_dll_directory` 和 prepend PATH |
| Cold start CUDA JIT compile 約 10 秒 | daemon 啟動時跑一次 dummy transcribe 預熱 |
| Windows-only `ctypes.WinDLL("user32")` 在 macOS import 直接炸 | 抽到 `stt_platform_win.py`,只在 `sys.platform == "win32"` 時 lazy import |
| macOS 全域聽鍵需要 Input Monitoring,模擬發鍵需要 Accessibility | daemon 啟動時就需要這兩個權限授給 Python binary;pyenv shim 路徑會綁錯,必須授實際 binary(`readlink -f $(which python3)`) |
| macOS mlx-whisper 第一次跑要從 HF 下載 ~1.5GB | 跟 faster-whisper 一樣會快取在 `~/.cache/huggingface/`,後續啟動秒級 |
| macOS pynput Controller 模擬 Cmd+V 在某些 pyenv 設定下被 Accessibility silent drop(文字停在剪貼簿但沒貼出) | 改走 `osascript -e 'tell ... keystroke "v" using command down'` — 權限綁系統 binary 而非 Python,跨 pyenv 版本穩定 |
| macOS beep 用 16kHz 取樣率送 sd.play 在 48kHz 輸出裝置上 resample 產生「叮叮」破碎感 | 啟動時用 `sd.query_devices(kind='output')['default_samplerate']` 取得原生取樣率、改用 raised-cosine fade、前面墊 5ms 靜音吸收開 stream 的 click |
| 小 LLM(Qwen2.5-1.5B)不老實聽 system prompt 規則 — 翻譯英文、改主詞、改事實 | 對「最小編輯」這種約束式任務,**換大但更聽話的模型**(Qwen3-4B-Instruct-2507)比量化的小模型可靠。Qwen3 generation 的 instruction-following 大幅優於 Qwen2.5(v0.7.0 18-case bench 驗證)|
| Polish decode 不是運算量大,是 **per-token Python + kernel launch overhead bound**(實測 22 ms/step,理論下限 8 ms,9x gap)| v0.7.1 用 **PLD + 預 cache POLISH_PROMPT KV** 攤平 — 一次 forward 平行驗證多 token、靜態 prompt 預 prefill。lossless 因為 verifier 仍是完整模型,長文 -55% |
| Windows 上很多 PyTorch CUDA 加速選項 silently no-op 或反而更慢 | torch.compile 需 triton(無 Windows wheel)、bnb-NF4 在 Blackwell + 小模型反而慢 67%、flash-attn 沒對齊 stack 的 wheel。**bench-first 否則 ship placebo**(早期一次「GPU mel spectrogram」Tier 1A 嘗試就是被 bench 攔住、研究 agent 預估錯 300x 的例子,revert 後 v0.7.2 改做 correctness sweep)|
| Plan / agent 估「press-time encoder pipelining 省 50% release-to-text latency」聽起來合理(GPU 在用戶 hold 期間閒著) | v0.7.3 bench 實測 RTX 5080 + Qwen3-ASR-0.6B 只省 ~3% (0.18s on 40s)、不是 50%。Root cause:plan 估 encoder forward 3-5s,實測 ~0.2s (off 15-25x);decoder 才是 ~95% 瓶頸。**第二次 bench-first save** — framework 留著 commit 進去 (`scripts/qwen3_asr_streaming.py` + worker + 11 tests) 但 `ENCODER_PIPELINING=False` 預設。未來換 FP8 decoder 或 llama.cpp 後再重評。詳見 [v0.7.3 投資紀錄](#v07x-效能與品質投資紀錄) |
| 用戶鬆手時 PortAudio 還在處理 in-flight 50 ms audio block,callback 的 `if _recording:` gate 拒絕後尾音 phoneme 被切掉 | v0.7.2 在 `_on_release` 加 80 ms drain delay 後才 flip `_recording = False`。Windows `time.sleep` 預設 ~15.6 ms 解析度 → 80 ms padding 保證 ≥1 個 PortAudio 50 ms 週期 elapse。Drain 期間 re-press 會 abort 該 release |
| 用戶在 transcribe 處理中再按一次,第二段 audio 沉默留在 buffer,等下次按鍵被 prepend 到第三段 transcript | v0.7.2 busy path 明確 `[stt] busy — dropped X.XXs of captured audio` log + clear buffer + reset sample counter,不再 silent merge |
| 卡住的 trigger key 沒有 buffer 上限,RDP 斷線 / kernel hang 會 memory bomb | v0.7.2 加 `MAX_AUDIO_SEC = 120`,callback 內檢查 `_recording_samples`,超過就 force-release + spawn transcribe |
| Qwen3-ASR LLM-backbone 對長靜音會 hallucinate(HF model card 列為已知 edge case,decoder 生成 training data「fit silence」的習得片語,如「好好好好」)| v0.7.2 在 ASR 前加 RMS silence trim(30 ms frame、-50 dBFS threshold、100 ms margin),純 numpy 微秒成本,順帶縮短 encoder forward |
| 每次 paste 開 powershell.exe ~100-300 ms 冷啟動 + 150 ms async 發佈 settle sleep,累計 ~250-450 ms 浪費 | v0.7.2 Win 換 ctypes `OpenClipboard` + `SetClipboardData(CF_UNICODETEXT, ...)`(~1-5 ms 同步寫,sleep 降到 20 ms);Mac 換 PyObjC `NSPasteboard.setString_forType_()`,pbcopy 保留 fallback |
| polish `max_tokens = 256` 對 280 字輸入會默默截斷句尾,paste 出去斷句 | v0.7.2 dynamic budget `max(64, min(input_tokens × 1.2, 512))`,ceiling 從 256 升到 512(memory 成本可忽略);加 truncation detection — 撞 budget 且 last token != `<|im_end|>` 就 fallback 到原 ASR 文字 |
| `POLISH_PROMPT` 全程中文只 anchor 中文行為,但 `POLISH_LANGUAGES` 包含 ja/ko 等於對日韓 transcript 0 規則約束 | v0.7.2 收斂到 `POLISH_LANGUAGES = {"zh"}`,等 per-language prompt dispatch 路徑就緒再開 ja/ko |
| 中文「好」「對」「是」典型發音 ~0.25 s,被 `MIN_AUDIO_SEC = 0.3` silent reject | v0.7.2 降到 0.15 s(仍高於 ~20 ms key-bounce + ~80 ms 最短意圖按鍵,但低於最短單音節回應)|

---

## 跨平台設計

daemon **核心管線 100% 跨平台**：

```
[mic] sounddevice
  → STTBackend (qwen3-asr default on Apple Silicon v0.3.0+ AND Windows/Linux v0.6.0+; faster-whisper / mlx-whisper switchable)
  → OpenCC s2twp  +  regex CJK/ASCII spacing (called twice: pre + post polish)
  → TextPostProcessor.polish() (optional, v0.5.0+ macOS / v0.6.0+ Win/Linux; gated on detected language)
  → Pasteboard.set_text() + Pasteboard.paste()   ← ★ 唯一綁平台的薄層
```

平台特定的三件事從 v0.2.0 起抽到 `Pasteboard` 介面(`scripts/stt_platform.py`),`build_pasteboard()` 依 `sys.platform` lazy-import 對應實作模組:

| 抽象層 | Windows(`stt_platform_win.py`) | macOS(`stt_platform_mac.py`) | Linux(規劃) |
|--------|------------------------------|----------------------------|----------|
| **Clipboard 寫入** | ctypes `OpenClipboard` + `SetClipboardData(CF_UNICODETEXT, GMEM_MOVEABLE handle)`(v0.7.2+,同步寫 ~1-5 ms;之前是 PowerShell `Set-Clipboard` 100-300 ms 冷啟動 + 150 ms async settle sleep)| PyObjC `NSPasteboard.setString_forType_()`(v0.7.2+,~1 ms 同步);AppKit 沒裝時 fallback 到 `pbcopy` 子程序 | `xclip` (X11) / `wl-copy` (Wayland) |
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

## 測試

v0.7.2 起 `tests/` 內有 pytest 回歸 test,主要防護:

1. **設定回歸**:`__version__` / `MIN_AUDIO_SEC` / `MAX_AUDIO_SEC` / `POLISH_LANGUAGES` 被改回舊值會立刻失敗 — 避免 v0.7.2 修好的 correctness fix(C1-C5)被 silent rollback
2. **狀態機**:`_on_press` / `_on_release` / `_audio_callback` / `_transcribe_and_emit` 的狀態轉移、drain delay、busy log + buffer clear、MAX_AUDIO_SEC auto-stop 全部有 test;backend / polisher / pasteboard 走 `unittest.mock.MagicMock`,**不需 GPU 也不載模型**
3. **`_trim_silence`**:tight clip 不變、靜音 pad 被剝掉、純靜音回空、太短 clip pass-through
4. **Polish quality 回歸 bench**:18 個來自 README v0.7.0 投資紀錄的 failure case(`commit` → `push`、「更慢」→「更快」、`_USE_TORCH_COMPILE` underscore 被吞、「幫我」→「幫你」、`prebuilt wheel` → 「預建的輪子」等)固化為 fixture(`tests/fixtures/polish_cases.json`),每 case 有 `must_contain` / `must_not_contain` / `max_edit_ratio` 規則。**Skip-by-default** 因為要載 8 GB 模型;本地驗證 polish model 升版或 transformers bump 沒回歸時跑

### 跑測試

```bash
pip install pytest  # 唯一額外依賴(numpy + sounddevice + pynput + opencc 是 runtime 依賴,已裝)

# 狀態機 + 設定 + silence trim(18 個 test,~3 秒,不載模型)
python -m pytest tests/ -v

# 加跑 polish 回歸 bench(載 Qwen3-4B,~30 秒 warmup + 每 case ~1-3 秒)
python -m pytest tests/ -v --run-polish-bench
# 或環境變數:HOME_STT_RUN_POLISH_BENCH=1 python -m pytest tests/ -v
```

### CI

`.github/workflows/tests.yml` 在 push / PR 到 main 時跑:
- **Matrix**:Windows + macOS × Python 3.10 + 3.12
- **跑什麼**:狀態機 + 設定 + silence trim(polish bench 預設 skip,避免 CI runner 載 8 GB 模型)
- **時間**:每 OS × py 組合 ~5 秒

### 手動 smoke 測試

`tests/smoke_clipboard.py` 是 ctypes Windows clipboard 的 round-trip 測試(zh-TW、識別符號、emoji surrogate pair、1000 字壓力),會短暫覆蓋系統 clipboard 然後還原。不是 pytest 因為動到真實系統狀態。跑法:

```powershell
python tests\smoke_clipboard.py
```

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

### 效能 / 品質

- [x] **v0.7.0 polish quality 升級** — Win/Linux 預設改 `Qwen3-4B-Instruct-2507`(對齊 Mac),修掉 Qwen2.5-1.5B 的英文翻譯 / 主詞改變 / 事實反向問題。詳見 [v0.7.x 投資紀錄](#v07x-效能與品質投資紀錄)
- [x] **v0.7.1 polish decode lossless 加速** — PLD + 預 cache POLISH_PROMPT KV + cuDNN benchmark。長文 polish -55%(4.26s → 1.90s),quality byte-identical 對照 v0.7.0
- [x] **v0.7.2 correctness sweep + clipboard direct API + dynamic budgets** — multi-agent review (`review-v0.7.1.md`) 後的工程交付。3 個 correctness fix (尾音 drain / busy log + buffer clear / `MAX_AUDIO_SEC=120`) + RMS silence trim (防 Qwen3-ASR LLM-backbone silence hallucination) + ctypes Win clipboard + PyObjC NSPasteboard 取代 subprocess(**每次 paste 省 ~250-450 ms**) + polish dynamic `max_tokens` + truncation detection + `POLISH_LANGUAGES` 收斂到 zh-only + `MIN_AUDIO_SEC` 0.3→0.15。同步加 `tests/`(18 個狀態機 test + 18 個 polish 回歸 case + GH Actions Win+Mac matrix)防回歸。詳見 [v0.7.2 投資紀錄](#v07x-效能與品質投資紀錄)
- [x] **v0.7.3 press-time encoder pipelining framework (shipped DISABLED, null result)** — 原訂為 v0.8.0、目標 50% release-to-text latency reduction。完整 framework 落地 (`scripts/qwen3_asr_streaming.py` `StreamingQwen3ASRModel` 子類 + `_encoder_worker` thread + STTBackend ABC 加 5 個 streaming method + Option C silence-detect fallback + 11 個新 state-machine test)。Day 13-14 daemon-driven bench 證明 RTX 5080 + Qwen3-ASR-0.6B 上只省 ~3% (0.18s on 40s audio)、不是預估的 50%。Root cause:plan 假設 encoder forward 3-5s、實測 0.2s,off by 15-25x;decoder 才是 ~95% 瓶頸。Ship framework with `ENCODER_PIPELINING = False` 預設、daemon 運行行為 byte-identical to v0.7.2。詳見 [v0.7.3 投資紀錄](#v07x-效能與品質投資紀錄)
- [ ] **v0.8.0 decoder-side acceleration** — v0.7.3 bench data 指明 decoder 是真瓶頸,不是 encoder。候選:**llama.cpp + GGUF Q8_0 backend swap** (原 v0.8.0 plan candidate B、估 decoder 2-3x、4-8 hr 工程量 + A/B quality 驗證)、**Qwen3-ASR-FP8 checkpoint** (等 Alibaba ship、~1.5-2x 但 lossy 需驗)、**speculative decoding 對小模型可行性 re-eval** (原 ruled out 但只測過 0.6B、未測更大 ASR 的 draft acceptance)
- [ ] **Polish KV 量化 / FP8 model swap** — Qwen3-4B-Instruct-2507-FP8 官方 ship,RTX 5080 有 native FP8 tensor cores,預估再 1.5-1.9x polish decode。風險:transformers FP8 path 較新,要驗證沒退回 dequant 路徑。留 v0.7.x 探索
