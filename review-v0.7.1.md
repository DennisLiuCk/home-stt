# Multi-Agent Code Review — home-stt v0.7.1

> 📅 Snapshot: 2026-05-24
> 🤖 Method: 4 specialist Claude agents reviewing in parallel — ASR architecture, polish LLM, systems engineering, GitHub competitive survey
> 📦 Scope: v0.7.1 codebase (`scripts/stt-daemon.py`, `scripts/text_polisher.py`, `scripts/stt_platform*.py`, `README.md`, `plan-v0.8.0-asr-latency.md`)

---

## 🎯 整體結論（TL;DR）

1. **這個專案的工程水準很高**（README 詳實、PLD+KV cache 設計到位、bench-first 紀錄、跨平台抽象清楚），但**最大的延遲瓶頸是架構性的，不是 kernel 級的** —— v0.8.0 plan 押 Flash Attention 2 / llama.cpp swap 都是攻錯地方。
2. **有 3 個 correctness bug 隱藏在優化工作底下**，**兩個 agent 獨立發現同一個**：尾音被丟、`_processing` 沉默吞 audio、stuck key 無上限。修這些比再追 1.3x 推論加速更有價值。
3. **沒有 committed 的回歸 bench**，但 README 列了 ~10 個 bench-validated 的拒絕路徑。下次升 transformers / model 版本，v0.7.0 修好的 quality regression（commit→push 等）可能無聲復發。
4. **直接競品 [Sumi](https://github.com/alan890104/sumi)** 在三個月前剛冒出來：Rust、Mac-only，但**用了完全相同的 stack**（Qwen3-ASR + LLM polish + 繁體中文 + OpenCC），而且還有 voice-edit mode (⌥+E) + 18 個 per-app preset —— home-stt 的差異化窗口在縮小。
5. **landscape 的普世弱點 = home-stt 的天然優勢**：<10% 專案有 LLM polish 層，~2% 做 zh-CN→zh-TW 正規化，**只有 home-stt 同時有 Win+Mac+Qwen3-ASR+LLM polish**。
6. **有 ~4 個 1-2 週工程可做的差異化功能**（per-app preset、glossary、voice-edit mode、簽章安裝包）—— landscape 都驗證了用戶價值。

---

## 🔴 立即修復 — Correctness bugs

### C1. 尾音被丟（兩個 agent 各自發現）
**位置：** `stt-daemon.py:869-879` + `:738-744`

`_on_release` 先 `_recording = False`（line 877），然後才 spawn transcribe thread。PortAudio callback 每 50 ms 觸發一次，**這個 50ms 之間的 in-flight audio block 會被 line 742 的 `if _recording:` gate 丟掉**。實際使用症狀：講到尾巴一鬆手「...這個 function」可能變「...這個 functio」、最後一個 phoneme clip 掉。

**修法：** 在 `_on_release` 內加 ~60ms 的 drain delay，讓 callback 把最後一個 audio block 落下後再 flip `_recording = False`。處理 user 在 drain window 內再按鍵的 race。

### C2. `_processing` 沉默吞掉第二段語音
**位置：** `stt-daemon.py:754-761`, `:855-879`

User 在 transcribe 跑的時候再按一次鍵：第二段 transcribe thread 進入 `_transcribe_and_emit` 發現 `_processing=True` → **silent early-return，第二段 audio 留在 `_buffer` 裡，等第三次按鍵時被 prepend 到第三段**。

**Bug 行為：** 講第二句的時候系統沒任何回應（沒 log 任何 dropped 訊息），講第三句的時候會被前面留下來的音訊污染。

**修法：** 在 `_on_release` 內，spawn 之前檢查 `_processing`；若 True，明確 log `[stt] busy, dropped previous utterance` 並 `_buffer.clear()`。

### C3. 沒有 `MAX_AUDIO_SEC` 上限
**位置：** `stt-daemon.py:88-89`

Stuck key、RDP 斷線、kernel hang、key-repeat 異常都會讓 `_buffer` 無限長大。

**修法：** `MAX_AUDIO_SEC = 120`；在 callback 檢查長度超過就強制 `_recording = False` + log。

### C4. Polish 對 ja/ko 沒有 prompt-level 保護
**位置：** `stt-daemon.py:187, 196-205`

`POLISH_LANGUAGES = {"zh", "ja", "ko"}` 但 `POLISH_PROMPT` 全程用中文，只 anchor 中文行為（「中文一律繁體」「禁翻譯英文」）。日韓 transcript 進 polish 時，4B 模型完全沒有 prompt 規則約束。

**修法：** 短期 `POLISH_LANGUAGES = {"zh"}` 排除 ja/ko；長期針對 ja/ko 各寫一份 minimum-edit prompt。

### C5. Polish silence hallucination 風險（未驗證但機制存在）
**位置：** `stt-daemon.py:531-539`

Qwen3-ASR 是 LLM-backbone，HF model card 明確列出「silence hallucination, mispronunciations, long-form drift」為 known edge cases。home-stt 對 `faster-whisper` 有 `vad_filter=True`，但 Qwen3AsrBackend 沒有任何 silence trim 或 VAD。

**修法：** 加個輕量 RMS-based silence trim（~20 行 numpy，0 成本）；或用 `webrtcvad`（200KB MIT）做 frame-level VAD。

---

## 🟡 高 ROI 優化

### O1. 真正的延遲瓶頸：架構性，不是 kernel 級
**所有 4 個 agent 收斂的結論。**

v0.8.0 plan 候選 A (Flash Attention 2, 1.3-1.6x) 跟候選 B (llama.cpp + GGUF, 2-3x) 都是攻 backend inference time，**但對 hold-to-talk 模式而言，user 鬆手前 GPU 都閒著**。40s 音訊：等 40s 講完 + 10s 推論 = 50s wall-clock。即使 backend 加速 2x → 40s + 5s = 45s。

**真正的勝場是 press-time encoder pipelining**：
- Encoder 是 deterministic feed-forward over audio chunks，沒有 inter-chunk dependency
- User 講話時，背景 thread 每 5s 跑一次 encoder
- 鬆手時只需算最後 0-5s chunk 的 encoder + 一次 decoder over 全部 hidden states
- **體感延遲縮減 ~50%**

**前提：** `qwen-asr` / `mlx-qwen3-asr` 是否分開 expose `model.encoder(audio_chunk)`？沒有的話 vendor-fork 加 20 行。**這個方向不在 v0.8.0 plan 的拒絕列表，值得先 spike 1-2 天**。

**較簡單的替代方案：** 鬆手時用 silence 切分 + 多 chunk 平行 decode（CUDA streams）。

### O2. Windows clipboard 替換 PowerShell：直接省 ~250ms / 次
**位置：** `stt_platform_win.py:143-159` + `stt-daemon.py:814`

每次 paste 開 powershell.exe = 冷啟動 100-300ms，然後 `time.sleep(0.15)` 再等 150ms。**累積每次 paste 浪費 ~250-450ms**。

**修法：** ctypes 直接呼叫 `OpenClipboard` / `EmptyClipboard` / `SetClipboardData(CF_UNICODETEXT, ...)` / `CloseClipboard`。Mac 端 `pbcopy` 換成 `NSPasteboard.generalPasteboard().setString_forType_()`。

### O3. `copy.deepcopy(self._prefix_cache)` 是性能 leak
**位置：** `text_polisher.py:390-391`

每次 polish call 都對 ~200 token × 36 layer × bf16 的 DynamicCache 做 deepcopy。預估 ~5-20ms / call。

**修法（待驗證 API）：** transformers ≥4.46 的 `DynamicCache` 有 `.crop(prefix_len)` —— polish 結束後 crop 回 prefix 長度而非每次 deepcopy。

### O4. `max_tokens=256` 對長輸入會默默截斷
**位置：** `stt-daemon.py:188`, `text_polisher.py:101, 195`

Qwen3 tokenizer ~1 token/中文字。User 講 280 字（README 中的「超長」場景）→ polish output 截在 256 token → 句子斷尾。

**修法：** Dynamic sizing：`max_new_tokens = min(int(len(input_tokens) * 1.2), 512)`。並加 truncation detection：output 長度 == max_new_tokens 時 log warning 並 fallback 到原 ASR 文字。

### O5. `temperature=0.0` 拿掉了 Whisper 的 loop-breaker
**位置：** `stt-daemon.py:389-398, 433-448`

當 greedy decode 卡進 repetition loop 時，會貼出「好好好好好好...」200 字。

**修法（最小）：** 加 post-decode regex 偵測 ≥3 token 重複 ≥3 次 → 若觸發、retry 一次 with `temperature=0.4`。Qwen3-ASR 路徑可加 `repetition_penalty=1.1` generation kwarg。

### O6. `condition_on_previous_text=False` 對 >25s 長句缺乏 coherence
**位置：** `stt-daemon.py:443`

Whisper 內部對 >30s audio 自動 chunk 成 30s 段，True 時段間傳 context。**關掉時段 2 不知道段 1 講什麼**，proper noun 跟 punctuation style 兩段不一致。

**修法：** Per-call 化、長錄音 (>25s) 自動切 True。

### O7. `MIN_AUDIO_SEC = 0.3` 拒絕了「好」「對」「是」
**位置：** `stt-daemon.py:88-89`

中文「好」「對」「是」典型發音 ~0.25s。**現在這些快速回應全部被 silent reject**。

**修法：** 降到 0.15s。

---

## 🟢 已經設計很好的部分

| 設計 | 為什麼好 |
|------|---------|
| `Pasteboard` ABC + lazy import | 跨平台抽象乾淨；Windows ctypes 不會在 Mac import 時炸 |
| `STTBackend` ABC + factory | 換引擎不影響 mic / post / paste 流程 |
| `_PREFERRED_ATTN` ladder (FA2 > sdpa > eager) | 自動降級，無 wheel 也 graceful |
| `_format_polish_user_msg` 包成 data prefix | 防止 4B model 把 ASR 文字當成 user 對話請求回答 |
| 雙呼叫 `post_process()`（pre-polish + post-polish） | 對「polish 可能滲簡體」的 deterministic backstop |
| macOS Quartz @ `kCGAnnotatedSessionEventTap` | IME-safe（注音/拼音不會吞掉 Cmd+V），符合 Apple docs |
| `build_polisher` 的三類錯誤分流訊息（ImportError / CUDA OOM / DLL） | 用戶看到 log 就知道下一步 |
| Polish 模型選 4B 而非 1.5B（v0.7.0 18-case bench） | 對 minimum-edit 任務「更大更聽話」優於「更小更快但會亂改」 |
| v0.7.1 PLD + 預 cache KV + cuDNN benchmark | 三招 lossless 疊加 -55% decode，bench-validated |

---

## 🌐 競品調研重點

### Sumi 是當前唯一直接競爭者
[`alan890104/sumi`](https://github.com/alan890104/sumi) — 2026-02-25 才出來、22 stars、Mac only、Rust（candle backend）。**功能集跟 home-stt 完全平行**：Qwen3-ASR + Whisper、Qwen3-8B 或 Llama-3-Taiwan polish、zh-CN→zh-TW、18 個 per-app preset，還多了個 `⌥+E` voice-edit mode。

**威脅評估：** Mac-only 但成長很快；對 Mac 用戶來說功能集已經超過 home-stt。home-stt 的窗口是「Windows + Mac 一致」這個 Sumi 短期填不到的洞。

### 高星專案總覽（前 8）

| 專案 | Stars | Lang | 模式 | ASR | LLM polish |
|------|-------|------|------|-----|-----------|
| whisper.cpp | 50.0k | C++ | library | Whisper | — |
| faster-whisper | 23.1k | Py | library | Whisper/CT2 | — |
| **Buzz** | 19.4k | Py | batch+live | Whisper | — |
| vosk-api | 14.8k | multi | streaming lib | Vosk/Kaldi | — |
| WhisperLiveKit | 10.3k | Py | streaming server | Whisper | — |
| **RealtimeSTT** | 9.8k | Py | streaming lib | multi | — |
| mlx-examples | 8.6k | Py | library | MLX Whisper | — |
| **CapsWriter-Offline** | 5.5k | Py | **PTT (CapsLock)** | Paraformer | **yes** |
| **VoiceInk** | 5.1k | Swift (Mac) | PTT+toggle | Whisper+Parakeet | dictionary only |
| **Epicenter/Whispering** | 4.6k | TS (Tauri) | PTT | Whisper | — |
| **openwhispr** | 3.3k | TS (Electron) | PTT+toggle | Whisper+Parakeet | yes (BYO) |

### Landscape 的普世弱點（= home-stt 護城河）
- **<10% 有 built-in LLM polish 層** → home-stt 在少數派
- **~2% 做 zh-CN→zh-TW 正規化** → 只有 Sumi 跟 home-stt
- **PyTorch+CUDA Qwen3-ASR on Windows** → home-stt 是 essentially unique

### Landscape 已經 ship 但 home-stt 還沒的功能

| 功能 | 來源 | 對 home-stt 的價值 |
|------|------|------------------|
| **Per-app preset / Power Mode** | VoiceInk, Sumi (18 個), TypeWhisper | 自動切換 polish prompt（Slack vs code editor）|
| **Personal glossary / hotwords** | VoiceInk, CapsWriter, voxtype | bias decoding 對 proper noun → CJK 巨大精度提升 |
| **Voice-edit mode（重寫選取文字）** | Sumi `⌥+E`, FreeFlow | 全新 use case，把 home-stt 從口述變成寫作 assistant |
| **Screen-aware context** | xuiltul/voice-input | 截圖前景 app → VLM 抽 vocab → polish 用 |
| **Signed installer (MSI / DMG)** | VoiceInk, FastWord, voicetypr | 去掉 `pip install` 門檻 → 10x 可觸及用戶 |
| **Local OpenAI-compat HTTP endpoint** | openwhispr, TypeWhisper, speaches | 讓 Claude Code / Cursor 把 home-stt 當 tool 呼叫 |

---

## 📋 建議 roadmap

### v0.7.2 — Correctness + Quick Wins（1 週工作量）
1. C1 修尾音丟失
2. C2 修 `_processing` 沉默吞 audio
3. C3 加 `MAX_AUDIO_SEC = 120`
4. C4 暫時 `POLISH_LANGUAGES = {"zh"}`
5. C5 加 RMS-based silence trim
6. O2 換 ctypes clipboard / NSPasteboard
7. O4 dynamic `max_new_tokens` + truncation detection
8. O7 `MIN_AUDIO_SEC` 0.3 → 0.15

### v0.7.3 — Tests & 回歸防護（半週）
9. tests/ + pytest + state-machine tests
10. Commit 18-case polish bench fixture
11. GitHub Actions matrix

### v0.8.0 — Architectural latency（2-3 週）
12. O1 spike: press-time encoder pipelining
13. **不要先做 Flash Attention 2** — 對長音訊體感改善有限

### v0.9.0 — 差異化功能（每個 1-2 週，獨立可 ship）
14. Per-app polish preset
15. User glossary YAML
16. Voice-edit mode
17. Signed installer (MSI + DMG)

---

## ⚠️ Risk register

| Risk | 機制 | 緩解 |
|------|------|------|
| Transformers / qwen-asr 升版打破 PLD + DynamicCache lossless 保證 | API 未公開保證 | 上 commit bench fixture（v0.7.3）|
| Qwen3-4B-2507 model 升版破壞 v0.7.0 fix (commit→push 等) | model card 變更 | 同上 |
| Blackwell sm_120 對齊持續落後 | 上游進度問題 | 持續 watch flash-attn / bnb / triton-windows |
| Sumi 加 Windows port | 對手強化 | 加速 v0.9.0 差異化功能 |
| Log 漏密碼 / 機敏資料 | print 完全 transcript 到 `%TEMP%` | 加 `LOG_TRANSCRIPTS = False` toggle |

---

**最重要的一句話：** 不要為了下一個 1.5x 推論加速，繼續把工程預算花在 backend swap 上。先修 C1-C5 correctness bug、加 bench 防回歸、然後試 press-time encoder pipelining。**這三件事的累計體感改善 > 任何一次 backend 換 stack**。
