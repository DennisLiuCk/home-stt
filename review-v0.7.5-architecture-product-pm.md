# home-stt 跨平台 AI 輸入法專案 — 架構師 / 產品設計師 / PM 綜合 Review

## Context

home-stt 是一個完全離線的跨平台 hold-to-talk 語音轉文字 daemon（v0.7.5），支援 Windows 10/11 + macOS Apple Silicon，提供兩個模式：語音輸入（Dictate）+ 語音編輯（Voice-Edit）。核心管線為 麥克風 → 全域鍵盤 hook → 音訊擷取 → STT 後端（Qwen3-ASR 預設）→ 後處理（OpenCC + CJK 間距）→ 可選 LLM 修飾 → 剪貼簿 + 貼上。全部 ~5500 行 Python，架構乾淨、工程決策有 bench 數據支撐、文件齊全。

本 review 從架構師、產品設計師、PM 三個角度提出可落地的改進建議，按 impact × effort 排優先序。

---

## 一、架構 Review

### 優勢（值得保留的設計）

1. **三層抽象乾淨** — `Pasteboard`（平台 IO）、`STTBackend`（語音引擎）、`TextPostProcessor`（LLM 修飾）三個 ABC，lazy import 避免跨平台 import 失敗。新增 Linux 只需 1 個檔案 + 1 個 branch。
2. **Bench-first 文化** — GPU mel patch 發現 300x 估算誤差後 revert、encoder pipelining 實測 3% vs 預估 50% 後 ship disabled。不 ship 安慰劑優化。
3. **優雅降級** — polish 失敗 → NoopPolisher、CUDA 失敗 → CPU fallback、flash-attn 不在 → sdpa → eager。daemon 永遠能工作。
4. **IME 免疫設計** — clipboard + atomic paste (Ctrl+V / Cmd+V) 取代逐字輸入，避開注音/拼音/倉頡攔截問題。macOS 雙路徑（Quartz CGEvent vs osascript）是聰明的 fallback。

### 改進建議

#### A1. 拆分 stt-daemon.py 單體（1885 行 → 4-5 個模組）

**現狀**：stt-daemon.py 包含 config、3 個 STT backend class、state machine、audio callback、transcription pipeline、keyboard hooks、main()。
**問題**：單一檔案改動影響面過大、code review 困難、測試 fixture 需要 import 整個模組才能 mock。
**建議**：
- `stt_config.py` — 所有 config 常數 + `post_process()` + `_trim_silence()`
- `stt_backends.py` — `STTBackend` ABC + `FasterWhisperBackend` + `MlxWhisperBackend` + `Qwen3AsrBackend` + 兩個 impl
- `stt_audio.py` — `_play_beep()` + beep config + `_detect_output_samplerate()`
- `stt_daemon.py` — state machine + keyboard hooks + main()（核心 orchestration）

好處：每個模組可獨立測試、backend 開發不必翻完整 daemon、conftest.py 不再需要 `importlib.util.spec_from_file_location` hack。

#### A2. 消除全域 mutable state（~20 個 module-level global）

**現狀**：`_recording`, `_processing`, `_buffer`, `_active_trigger`, `_encoder_*`, `_edit_*` 等 ~20 個 global 靠單一 `_state_lock` 保護。
**問題**：狀態管理隱式、thread safety 脆弱、新功能（如多 trigger 同時按）很難安全加入。
**建議**：封裝成 `DaemonState` dataclass + context manager：
```python
@dataclasses.dataclass
class DaemonState:
    recording: bool = False
    processing: bool = False
    active_trigger: Key | None = None
    buffer: list[np.ndarray] = field(default_factory=list)
    recording_samples: int = 0
    edit_mode: bool = False
    # ... encoder state ...
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
```
好處：state 有型別、可 snapshot、可序列化 debug、新功能只加 field 不加 global。

#### A3. 引入 config file（取代 hardcoded 常數）

**現狀**：所有 config（STT_BACKEND、POLISH_MODEL、TRIGGER_KEYS、BEEP 設定等）要改 source code。
**問題**：用戶需要 git 知識才能自訂；升版時 merge conflict；不懂 Python 的用戶無法自訂。
**建議**：
- 支援 `~/.config/home-stt/config.toml`（或 `%APPDATA%\home-stt\config.toml`）
- 現有 module-level 常數作為 default，config file 覆蓋
- `home-stt config --init` 產生帶註解的預設 config
- `home-stt config --edit` 用 $EDITOR 開啟（home_stt.py 已有部分實作）
- 環境變數 `HOME_STT_*` 作為第三層 override（CI / 自動化場景）

優先序：config file > env var > code default

#### A4. 清理 encoder pipelining dead code

**現狀**：~400 行 disabled framework（`ENCODER_PIPELINING = False`）+ 11 個測試，佔 stt-daemon.py 超過 20%。
**建議**：將 `_encoder_worker`、`_encoder_*` state、`_audio_callback` 的 dual-write 邏輯、streaming dispatch 抽到 `stt_streaming.py`。daemon 只保留 `if ENCODER_PIPELINING: from stt_streaming import ...`。框架保留但不佔主模組認知負擔。

#### A5. 版本號統一管理

**現狀**：version 重複存在於 `stt-daemon.py:__version__` 和 `pyproject.toml:version`，`home_stt.py` 用 regex 讀取。
**建議**：single source of truth 放 `pyproject.toml`，runtime 用 `importlib.metadata.version("home-stt")` 讀取。或用 `__version__.py` pattern。

---

## 二、產品 / UX Review

### 優勢

1. **核心互動簡潔** — 按住說話、放開自動貼上。零學習成本的核心迴路。
2. **音訊回饋** — press beep / end beep / fail beep 三種音高區分狀態，raised-cosine envelope 做得細緻。
3. **Voice-Edit 創新** — 選取 + 語音指令改寫是有差異化的 feature，對 power user 有吸引力。
4. **完全離線** — 對隱私敏感的用戶（醫療、法律、企業內部）是強賣點。

### 改進建議

#### P1. System Tray / Menu Bar 狀態指示器（獨立 PR）

**現狀**：daemon 在背景跑，用戶只能透過 terminal 看 `home-stt status` 或聽 beep。
**問題**：
- 用戶不知道 daemon 是否在跑
- 不知道當前是否在錄音
- 不知道 transcription 進度
- 第一次用的人不知道有沒有成功

**建議**：
- macOS: Menu Bar icon（可用 `rumps` 輕量框架，~100 行）— 錄音時變紅、處理中轉圈、idle 灰色
- Windows: System Tray icon（`pystray` 套件）— 同上
- 右鍵選單：Start/Stop/Status/Recent Transcripts/Settings
- 這是提升「用戶感知產品在工作」的最高 ROI 改動

#### P2. 首次啟動引導（Onboarding Wizard + Doctor）

**現狀**：macOS 需要手動授 3 個權限（Input Monitoring / Accessibility / Microphone），Windows 需要特定 pip install 順序。README 雖然寫得很詳細，但仍然是 #1 掉坑點。
**建議**：
- `home-stt doctor` 子命令（快速健檢）：
  - 自動檢查 Python 版本、必要套件是否已裝、CUDA 是否可用
  - macOS: 偵測缺少的權限並提示精確的系統設定路徑
  - Windows: auto-detect torch CUDA 版本 mismatch
  - 麥克風是否可用
  - 輸出 pass/fail checklist
- `home-stt setup` 子命令（完整引導）：
  - 包含 doctor 的所有檢查
  - 引導安裝缺少的依賴
  - 跑 warmup 並確認 STT + polish 正常
  - 最後跑一個 "say something" 互動測試

#### P3. 視覺化錄音回饋（搭配 P1）

**現狀**：僅有 beep 音。
**建議**：
- Menu Bar / Tray icon 旁顯示音量 meter（即時 RMS）
- 錄音時 icon 動畫
- Transcription 完成時 notification（macOS notification center / Windows toast）顯示轉錄結果
- 讓用戶「看到」系統在工作，減少「我剛剛有沒有按到」的疑惑

#### P4. 可自訂 Trigger Key（不需改 source）

**現狀**：改 trigger key 必須編輯 stt-daemon.py。TKL / 筆電用戶沒有 F13。
**建議**：
- Config file 中支援 `trigger_keys = ["alt_r"]` 和 `edit_trigger_keys = ["f13"]`
- `home-stt config --set-trigger` 互動式按鍵偵測
- 搭配 A3（config file）一起做

#### P5. 麥克風裝置選擇

**現狀**：永遠使用系統預設麥克風。
**建議**：
- `home-stt devices` 列出可用麥克風
- Config file 中 `mic_device = "MacBook Pro Microphone"` 或 device index
- 用例：外接 USB mic、藍牙耳機、多裝置工作站

#### P6. Smart VRAM Tier Auto-Detection

**現狀**：預設 config 需要 ~10 GB VRAM。README 提供降階 Preset（Balanced / Light / Mini），但 daemon 不會自動偵測可用 VRAM 選擇 preset。CUDA OOM 時 fallback 到 NoopPolisher（直接關閉 polish），中間沒有嘗試小模型。
**建議**：
- 啟動時 print 可用 VRAM（`torch.cuda.get_device_properties(0).total_memory`）讓用戶知道自己在哪個 tier
- `build_polisher()` 在 CUDA OOM 時自動嘗試降階：4B → 1.5B → disabled（目前是 4B → disabled）
- Config file 支援 `polish_model = "auto"`，daemon 根據可用 VRAM 自動選擇最佳模型

#### P7. Undo 支援（Voice-Edit）

**現狀**：voice-edit 改寫後若 LLM 結果不好，用戶只能手動 Cmd+Z。
**建議**：
- Voice-edit 結果貼回後，自動 set clipboard 為 original selection
- 加一個 beep pattern 或 notification 提示「按 Cmd+Z 可復原」
- 考慮短時間內（如 5 秒）按 edit trigger 兩次 = undo

---

## 三、PM / 策略 Review

### 優勢

1. **明確的技術定位** — 100% 離線、hold-to-talk、中英混合最佳化，不是什麼都想做的 Swiss Army knife。
2. **工程紀律** — bench-first 決策、每個 version 的投資紀錄詳實、已測試但放棄的方向有文件記錄避免重複嘗試。
3. **平台抽象就緒** — Linux 支援的架構已 ready，只差實作。

### 改進建議

#### S1. Linux 支援（暫緩）

架構已就緒（只需 `stt_platform_linux.py` + 1 branch），但目前暫無支援計劃。Wayland 生態碎片化（wlroots vs GNOME vs KDE 的鍵盤 hook 機制不同）是主要風險。列入長期 backlog，待 Windows + macOS 的體驗打磨完成後再評估。

#### S2. 依賴管理正規化

**現狀**：pyproject.toml 不列 runtime dependencies，README 是 source of truth。
**問題**：
- 新用戶容易裝錯版本或漏裝
- `pip install home-stt` 不會拉依賴
- 無法用 `pip install home-stt[cuda]` 或 `home-stt[mlx]` 做 platform extras

**建議**：
```toml
[project]
dependencies = [
    "numpy>=1.24",
    "sounddevice>=0.4",
    "pynput>=1.7",
    "opencc-python-reimplemented>=0.1.7",
]

[project.optional-dependencies]
cuda = ["torch>=2.0", "transformers>=4.40", "qwen-asr>=0.1"]
mlx = ["mlx>=0.10", "mlx-lm>=0.10", "mlx-qwen3-asr>=0.1"]
whisper = ["faster-whisper>=1.0"]
dev = ["pytest>=7.0"]
```
README 改為 `pip install home-stt[cuda]` 或 `pip install home-stt[mlx]`。

#### S3. 引入結構化 logging

**現狀**：所有 log 用 `print(f"[stt] ...", flush=True)` 和 `print(..., file=sys.stderr)`。
**問題**：
- 無 log level 區分（debug / info / warning / error）
- 無法 programmatic 過濾
- 無時間戳
- 無法 JSON 輸出供 monitoring 工具消費

**建議**：
- 用 Python `logging` 模組替換所有 `print`
- 預設 StreamHandler → `[stt] {timestamp} {level} {message}`
- `home-stt log --level debug` 可調
- 為未來的 monitoring / alerting 預留 structured JSON handler

#### S4. 建立 plugin / 擴展機制

**現狀**：新增 STT backend 或 polish behavior 需要改 source code。
**建議**：
- 利用 Python entry points 機制：`[project.entry-points."home_stt.backends"]`
- 第三方可以 `pip install home-stt-sensevoice` 自動被 `build_backend()` 發現
- 長期降低 core maintainer 負擔、讓社群貢獻 backend

#### S5. 競品差異化定位文件

**現狀**：README 沒有跟 macOS Dictation、Windows Speech Recognition、Whisper.cpp、Talon Voice 的比較。
**建議**：加一個簡短的比較表：

| Feature | home-stt | macOS Dictation | Talon Voice | Whisper.cpp |
|---------|----------|-----------------|-------------|-------------|
| 完全離線 | Yes | No (cloud) | Yes | Yes |
| LLM 修飾 | Yes | No | No | No |
| Voice-Edit | Yes | No | Yes (advanced) | No |
| 中英混合 | 最佳化 | 一般 | 英文為主 | 一般 |

讓用戶快速理解為什麼選這個。

---

## 四、優先排序矩陣（Impact × Effort）

### 🔴 High Impact / Low Effort（先做）

| # | 建議 | Impact | Effort | 理由 |
|---|------|--------|--------|------|
| A3 | Config file | ★★★★★ | 2-3 天 | 解鎖 P4 (trigger key 自訂)、降低用戶門檻、升版不 conflict |
| P4 | 可自訂 trigger key | ★★★★ | 1 天 | 搭 A3，筆電/TKL 用戶當前完全無法自訂 |
| A5 | 版本號統一 | ★★★ | 0.5 天 | 消除重複，簡單 |
| S2 | 依賴管理正規化 | ★★★★ | 1-2 天 | `pip install home-stt[cuda]` 大幅降低安裝摩擦 |

### 🟡 High Impact / Medium Effort（接著做）

| # | 建議 | Impact | Effort | 理由 |
|---|------|--------|--------|------|
| A1 | 拆分 monolith | ★★★★ | 3-5 天 | 降低認知負擔、改善可測試性 |
| P1 | System Tray icon（獨立 PR） | ★★★★★ | 3-5 天 | 用戶體驗質變 — 從「不知道有沒有在跑」到「看得見」 |
| S3 | 結構化 logging | ★★★ | 2 天 | 為 monitoring / debug / plugin 打基礎 |
| P2 | Onboarding wizard | ★★★★ | 3-4 天 | 降低 #1 掉坑點（macOS 權限 + Windows pip 順序）|

### 🟢 Medium Impact / Varied Effort（排入 backlog）

| # | 建議 | Impact | Effort | 理由 |
|---|------|--------|--------|------|
| A2 | DaemonState dataclass | ★★★ | 3-4 天 | 重構風險需要完整 state machine test 保護 |
| A4 | 抽出 streaming code | ★★ | 1-2 天 | 清理，非功能性 |
| P3 | 視覺化錄音回饋 | ★★★ | 2-3 天 | 搭 P1 做 |
| P5 | 麥克風選擇 | ★★ | 1 天 | 少數用戶需求 |
| P6 | Smart VRAM auto-detection | ★★★ | 0.5 天 | 自動降階 4B→1.5B→disabled |
| P7 | Voice-Edit undo | ★★ | 1 天 | nice-to-have |
| S4 | Plugin 機制 | ★★★ | 5-7 天 | 長期 ROI，但目前 backend 數量少 |
| S5 | 競品比較 | ★★ | 0.5 天 | 文件改善 |
| S1 | Linux 支援 | ★★★ | 3-5 天 | 架構就緒但暫無計劃，排最低 |

---

## 五、建議實施路線圖

### Phase 1: Foundation（1-2 週）
- A3 Config file + P4 Trigger key 自訂
- A5 版本號統一
- S2 依賴管理正規化（extras）
- S5 競品比較文件

### Phase 2: 可見性 + 體驗（2-3 週）
- P1 System Tray / Menu Bar icon（獨立 PR）
- S3 結構化 logging
- P2 Onboarding wizard（`home-stt setup` / `home-stt doctor`）
- P5 麥克風裝置選擇

### Phase 3: 架構打磨（3-4 週）
- A1 拆分 monolith
- A2 DaemonState dataclass
- P3 視覺化錄音回饋
- A4 Streaming code 抽離

### Phase 4: 長期 Backlog
- S4 Plugin 機制
- P6 Voice-Edit undo
- S1 Linux 支援（暫緩，待 Win+Mac 體驗成熟後再評估）
- Smart VRAM tier auto-detection
- Per-language polish prompt (ja/en)
- Per-app preset（競品 parity with Sumi）

---

## 六、Verification

- 每個改動都應有對應的 test（state machine tests 已有好基礎）
- Config file: 測試 default → file override → env override 優先序
- System Tray: 在 macOS + Windows 實測 icon state 切換
- 依賴管理: 在 clean venv 測 `pip install home-stt[cuda]` / `home-stt[mlx]`
- 拆分: 確保所有現有 18 state machine tests + 18 polish bench cases 繼續 pass
- Doctor/Setup: 在 clean macOS + Windows 環境實測完整 onboarding flow

---

## 總結

home-stt 是一個工程品質高、架構思考清晰的專案。三層抽象設計、bench-first 優化文化、IME 免疫策略都是值得保留的設計。主要改進方向是：

1. **降低使用門檻**（config file、依賴管理、onboarding wizard）
2. **提升可見性**（System Tray icon、結構化 logging、視覺回饋）
3. **擴展平台覆蓋**（Linux 支援已就緒，ROI 最高）
4. **改善可維護性**（拆分 monolith、統一版本號、封裝 state）

這些改進不是「推倒重來」而是「在已有的好基礎上做增量優化」，符合專案一貫的 bench-first、漸進式工程風格。
