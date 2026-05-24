# Changelog — home-stt

詳細工程紀錄(由 README v0.7.4 之後從主文件抽出)。每個版本記錄**做了什麼**、**為什麼**、**bench 結果**,目的是讓未來重啟某個方向時不用重新繞同樣的彎路。

主 README 只放 summary table + 一句話 highlights;這裡放完整投資推演。

---

## 延遲基準

實測延遲(Windows + RTX 5080,v0.7.1+):

| 場景 | 音訊長度 | ASR | Polish | 總等待 |
|------|---------|-----|--------|--------|
| 短(「好」、「對啊就是這樣」)| 1-3s | ~0.5-1s | ~0.25s | < 2s |
| 中(技術討論一兩句)| 5-15s | ~2-3s | 0.7-1.0s | ~3-5s |
| 長(完整段落)| 20-30s | ~7-8s | ~2-3s | ~10-13s |
| 超長(連續講 40s)| 40s+ | ~10s | ~5s | ~15s |

Mac M-series 跑同 stack 短中段約 1-2s(MLX 4-bit + Metal native,Python overhead 比 PyTorch 路徑低)。

---

## v0.7.0 — Polish model 換回 4B 修品質

v0.6.0 為了省 VRAM 暫用 Qwen2.5-1.5B-Instruct,後續 18-case bench 對照 `Qwen2.5-{0.5B, 1.5B}` 跟 `Qwen3-4B-Instruct-2507` 發現 1.5B 在我們這個「最小編輯」任務上會:

- **改 English keyword**:`commit` → `push`(完全不同的動詞,嚴重事實錯誤)
- **改變事實 / 數字 / 語義**:「INT4 反而**更慢**」→「INT4 反而**更快**」(語義反向)
- **翻譯英文片語**:`prebuilt wheel` → 「預建的輪子」(違反 prompt「禁翻譯英文」規則)
- **過度書面化**:「想要解決」→「目的是為了解決」、「容易實作」→「容易實現」

換回 Qwen3-4B-Instruct-2507 後上述全部消失。**Qwen3 generation 的 instruction-following 顯著優於 Qwen2.5**(Qwen3.5-4B IFEval 89.8 vs Qwen2.5 大幅進步)。對約束式任務,**更大但更老實的模型,比量化的小模型可靠** — bigger ≠ better in raw size, but bigger generation 通常 = more obedient。

---

## v0.7.1 — Polish decode -55%,quality byte-identical

長文 polish:v0.7.0 ≈ 4.26s,v0.7.1 ≈ **1.90s**。三個 **lossless** 工程優化疊加(都在 `TorchLocalLlmPolisher` 內,verifier 仍是完整 Qwen3-4B):

1. **Prompt Lookup Decoding (PLD)** — `generate(..., prompt_lookup_num_tokens=10)`。Polish 輸出 ≈ 輸入(只刪贅字、補標點),PLD 從 input 抓 token 序列做**平行驗證**,「一步猜一 token」變「一步驗證多 token」。lossless 因為驗證仍由完整模型做
2. **預先 cache POLISH_PROMPT 的 KV** — 系統 prompt 每次都一樣,daemon 啟動時跑一次 prefill 把 `DynamicCache` 存起來,每次 polish `deepcopy` 一份從「已讀完 prompt」狀態接續,省 ~30-50 ms prefill / call
3. **cuDNN benchmark mode** — `torch.backends.cudnn.benchmark = True`,第一次跑 autotune 挑最快 kernel(+~24s daemon startup 一次性成本),之後每步省一點

**根因觀察**:Polish 慢的不是模型運算量(8 GB weights / 960 GB/s = 8 ms 理論下限),而是 **per-token 固定 overhead**(Python loop + CUDA kernel launch + transformers internals 累計 ~22 ms/step,9x 高於理論)。v0.7.1 三招都是攤平這個 fixed overhead,**不碰模型也不量化** → quality 完全保留。

---

## v0.7.2 — Correctness sweep + clipboard direct API + dynamic budgets

v0.7.2 是 multi-agent review (見 `review-v0.7.1.md`) 後的工程交付。三類修正:correctness bug、clipboard 開 subprocess 浪費的延遲、polish 對長輸入的 silent truncation。

### 3 個 correctness bug

1. **尾音被丟** (`stt-daemon.py:_on_release`) — 鬆手後 PortAudio 還在處理 in-flight 50 ms audio block,但 `_recording = False` 已經被 flip,callback 的 `if _recording:` gate 拒絕該 block。實際症狀:講「...這個 function」會被切成「...這個 functio」(尾音 phoneme 缺一截)。修法:鬆手後 `time.sleep(0.08)` drain delay 再 flip。Windows `time.sleep` 預設解析度 ~15.6 ms (一個 multimedia timer tick),`sleep(0.05)` 可能 ~47 ms 就返回 → padding 到 80 ms 保證 ≥1 個 PortAudio 50 ms 週期 elapse。Drain 期間 re-press 會 abort 該 release 並讓新 press 繼續 capture。
2. **`_processing` 沉默吞第二段語音** (`_transcribe_and_emit`) — 用戶在 transcribe 跑時再按一次,第二段 audio 留在 buffer 等下次按鍵時被 prepend 到第三段 transcript。用戶完全不知道發生什麼。修法:busy path 明確 `[stt] busy — dropped Xs of captured audio (previous transcribe still running)` log + clear buffer + reset `_recording_samples`。
3. **無 `MAX_AUDIO_SEC` 上限** (`_audio_callback`) — stuck key / RDP 斷線 / kernel hang 會讓 buffer 無限長大。修法:120 s 硬上限,callback 內檢查 `_recording_samples`,超過就 force-release + spawn transcribe。

### Polish silence hallucination 防護

Qwen3-ASR 是 LLM-backbone,HF model card 明確列出「silence hallucination」為已知 edge case (decoder 在大段沉默上會生成 training data 中「fit silence」的習得片語,如中文 voice-blog 結尾「好好好好」)。新增 RMS-based silence trim 在 ASR 之前:30 ms frame、-50 dBFS threshold、100 ms margin。純 numpy 微秒成本,順帶縮短 encoder forward。

### Polish 範圍收斂到 zh-only

`POLISH_LANGUAGES` 從 `{zh, ja, ko}` 改 `{zh}`。`POLISH_PROMPT` 全程中文、只 anchor 中文行為(「中文一律繁體」「禁翻譯英文」),ja/ko transcript 透過同一個 prompt 等於 **0 規則約束**,4B 模型可任意改寫。等 per-language prompt dispatch 路徑就緒再開回 ja/ko。

### Clipboard 從 subprocess 改成直接 API

- **Windows**:`powershell.exe -Command "Set-Clipboard"` (冷啟動 100-300 ms + async 發佈需要 150 ms settle sleep) 換成 ctypes 直接呼叫 `OpenClipboard` + `EmptyClipboard` + `SetClipboardData(CF_UNICODETEXT, GMEM_MOVEABLE handle)` + `CloseClipboard`。**~1-5 ms 同步寫**,SetClipboardData 返回就代表 OS-side clipboard daemon 已發佈。settle sleep 從 150 ms 降到 20 ms (只留 keystroke timing margin)。每次 paste 累計省 **~250-450 ms**。OpenClipboard 含 5×10 ms retry 處理 clipboard manager 短暫鎖住。
- **macOS**:`pbcopy` subprocess (~20-50 ms) 換成 PyObjC `NSPasteboard.generalPasteboard().setString_forType_(text, NSPasteboardTypeString)`,~1 ms 同步。AppKit 沒裝時自動 fallback 到 `pbcopy`(不會 crash)。

### Polish `max_tokens` dynamic budget + truncation detection

固定 256 token cap 對 README 的「超長」測試(~280 字 zh,~280 token)會默默截斷句尾,paste 出去是斷句。改成 `max(64, min(input_tokens * 1.2, ceiling))`,ceiling 從 256 提到 **512**(memory 成本 ~25 MB 額外 KV cache,可忽略對 8 GB 模型)。Truncation detection:若 output token 數撞到 budget **且** last token != `<|im_end|>` (chat terminator),log warning 並 fallback 到原 ASR 文字 — 「truncated polish」嚴格 worse than 「raw ASR」。

### `MIN_AUDIO_SEC` 0.3 → 0.15

中文「好」「對」「是」典型發音 ~0.25 s,原本被 silent reject (`[stt] too short`)。0.15 s 仍在 key-bounce (~20 ms) 跟最短意圖按鍵 (~80 ms) 之上,但低於最短單音節回應。

### Tests + CI (v0.7.3 同步交付)

新增 `tests/` (state machine + silence trim + 設定回歸 18 個 test,純 mock 不需 GPU) + 18 個 polish quality 回歸 case (v0.7.0 投資發現的 failure mode 全部變成 fixture) + `.github/workflows/tests.yml` Win + Mac × py3.10/3.12 matrix。Polish bench skip-by-default,本地跑 `pytest tests/ --run-polish-bench`。

---

## v0.7.3 — Press-time encoder pipelining framework + bench-first null result

v0.7.3 是**第二次 bench-first save**(第一次是 v0.7.x 的 GPU mel patch revert)。原訂為 v0.8.0、目標 50% release-to-text latency reduction,實際 bench 數據證明假設錯了,改 ship framework 但 `ENCODER_PIPELINING = False` 預設、記錄 null 結果。

**原始 plan 假設**:hold-to-talk 模式下 GPU 在用戶講話時閒著、encoder forward 對 40s 音訊估 3-5s。把 encoder 移到背景 worker thread 跑、用戶鬆手時只算 decoder + tail encoder → 釋放後等待時間從 ~7s 降到 ~3-4s (≈50% 改善)。

### 已做的完整工程

(15 天 spec、~600 行 code、11 個新 test、全部 commit 進 repo 留作未來重啟)

- `scripts/qwen3_asr_streaming.py` — `StreamingQwen3ASRModel(Qwen3ASRModel)` 子類,透過 monkey-patch `thinker.get_audio_features` 注入 pre-computed encoder hidden states。Spike (`tmp/spike_torch_encoder.py`) 已驗證 chunked encode + concat dim=0 在真實中英混合 20s 上 Levenshtein=0、40s continuous 上 Lev=21 (字詞 drift 但語意保留)、30s 中段靜音上 Lev=21 (silence 在 chunk 中央破壞 hidden states)
- STTBackend ABC 加 5 個 streaming method (`supports_streaming` / `start_encoder` / `push_chunk` / `finalize` / `abort`);default `False` + `NotImplementedError` 讓 batch-only backend 零改動
- `_encoder_worker` thread + `_encoder_*` state machine (queue.Queue / threading.Event)、`_audio_callback` 加 lazy spawn + dual-write + RMS silence-detect、`_on_press` 重置、`_on_release` 設 stop event、`_transcribe_and_emit` 兩路 dispatch with abort/fallback
- Option C 防禦:mid-utterance ≥2s silence 觸發 `_encoder_use_batch_fallback = True`、跳過 streaming 直接走 batch (避免 silence-mid 的 Lev=21 drift)
- 11 個新 state-machine test (lazy spawn、dual-write、stop event、crash fallback、finalize timeout、consecutive failure suppression、short tap、processing flag independence、Option C silence detect、100-cycle race stress)、`_install_inert_mocks` 加 `streaming` kwarg、`fresh_daemon` fixture auto-cleanup worker threads
- v0.8.1 deferred:`_Qwen3MlxImpl.supports_streaming() = False` (MLX 端框架 ready 但需 Mac 驗證)

### Day 13-14 bench 實測

(`tmp/bench_v080_latency.py`、daemon-driven、50ms realtime feed、encoder thread 真的在背景跑)

| Sample | audio | batch | stream | saved | % | Lev |
|--------|-------|-------|--------|-------|---|-----|
| sample.wav (zh-en) | 20s | 2.83s | 2.88s | -0.06s | -2% | 2 |
| sample_english | 20s | 2.82s | 2.85s | -0.03s | -1% | 0 |
| sample_long (zh-en) | 40s | 6.99s | **6.81s** | **+0.18s** | +3% | 21 |
| sample_silence-mid | 30s | 4.54s | 4.47s | +0.07s | +2% | 21 |

**Root cause**:plan 假設 encoder forward ~3-5s for 40s audio;實測 RTX 5080 + Qwen3-ASR-0.6B (audio_tower ≈ 200M params) **~0.2s**。Off by 15-25x。Decoder (autoregressive ~200 tokens) 才是 ~95% post-release time 的瓶頸。

**Ship decision**:framework 留著 (架構乾淨、test coverage 高、abstractions 對未來有用),但 `ENCODER_PIPELINING = False` 預設、`__version__ = 0.7.3` 而非 0.8.0。Daemon runtime 行為 = v0.7.2 byte-identical。需要時 `ENCODER_PIPELINING = True` 一鍵啟用。

### 未來何時重評估(觸發條件)

- Qwen3-ASR-FP8 checkpoint 由 Alibaba 官方 ship → FP8 decoder 速度可能 2x → encoder 佔比上升、pipelining 開始有意義
- llama.cpp + GGUF Q8_0 backend swap 落地 (原 v0.8.0 plan candidate B、2-3x decoder 加速) → 同上
- 改用更大的 ASR (1.7B 或未來更大版本) → encoder forward 變長、pipelining 改善幅度跟著放大
- Apple Silicon MLX 端 `audio_tower` 速度數據 (deferred 到能跑 Mac 時)

### Lessons

1. **Bench-first 又一次救命** — agent / plan 預估 50% 改善,實測 3%。第二次驗證「Windows + PyTorch CUDA 路徑充滿性能直覺反例」這個 v0.7.2 提到的教訓
2. **Plan-vs-actual 數據要寫進 commit message + README** — 不止防回歸,也防未來重複犯同樣的錯
3. **建好框架但不 enable** 是合理的 ship 方式 — 比拆掉重建好,也誠實 (用戶 opt-in 才付代價)
4. 真正的 latency 瓶頸在 decoder。下一輪優化從 decoder 下手 (llama.cpp、FP8、speculative decode 對小模型的可行性)

---

## v0.7.4 — Polish prompt 標點保留修正 (live-log discovered, bench-validated)

**Symptom**:用戶在實際 dictation 中發現中文標點在 polish 後**不一致地**被刪除 — 短句保留、多句中間的「。」被吃掉換成空格或直接消失。`%TEMP%/stt-daemon.log` 證據:

```
raw:      我剛剛在測試這個工具的過程中，發現了一個小問題。我發現中文的輸出都沒有標點符號。
polished: 我剛剛在測試這個工具的過程中發現了一個小問題 中文的輸出都沒有標點符號
                                                ^                                ^
                                                「。」變空格                    句尾「。」消失
```

**Root cause**:POLISH_PROMPT 規則不對稱 — 只說「**補**必要標點」(正向、處理缺漏),從沒禁止「**刪**原有標點」(負向、處理多餘)。Qwen3-4B-Instruct-2507 把「最小修飾」解讀為流暢度優化,把句末「。」當成可合併的單字元編輯,**特別是多句長輸入**。短句 + ？/，多半保留;長段 + 「。」 隔開的多句就遭殃。

**修正過程** — 單軸修法在 bench 中證實不足,需三軸並進:

| 嘗試 | bench 結果 |
|------|------------|
| (僅) 加 `刪除或替換原有標點` 進 嚴禁 list | 4/5 punct case 仍 fail — 4B 模型忽略埋在第三行末尾的負向約束 |
| (僅) 加正向約束 `原有標點(。？！，)完整保留` 到第一行 | 未獨立測,但結合其他軸 OK |
| (僅) 加多句保留 few-shot example | 未獨立測 |
| **三軸並進 (正向 + 負向 + example)** | **5/5 punct case 過、原 18 case 零回歸、bench 23/23 全綠** |

**Lesson**:對 4B 等級 instruction-tuned 模型,要改變「default rewriting」這種強先驗行為,光加負向 constraint 不夠 — 需要 (a) 把約束**前置**到 prompt 第一行 (model attention 集中區)、(b) 給**具體 example** 展示期望行為 (few-shot signal 比 negative constraint 強)、(c) 同時保留**負向 constraint** 作雙重保險。

**Side fix**:順手修了 `id_underscore_prefix` fixture 的 substring-match bug (`"USE_TORCH_COMPILE 設成"` 是 `"_USE_TORCH_COMPILE 設成"` 的子字串、即使 `_` 被正確保留也會 false-positive 失敗;改用 `" USE_TORCH_COMPILE"` leading-space pattern 精確偵測 underscore 被刪)。修完 bench 由 22/23 → 23/23 真正全綠。

**Investment**:~1 hr (log 分析 + prompt 三輪迭代 + 5 個 fixture case + bench 三輪 + 1 個 side fixture fix)。Prefill cost 從 ~110 token → ~140 token (+27%) per polish,但 prefix-cache 吃掉、首次以外的 latency 影響零。

**為什麼這個 bug 拖到 v0.7.4 才發現**:v0.7.0 換 polish model 到 Qwen3-4B-Instruct-2507 時,18 個 quality bench case 全部是 single-sentence 或無標點輸入。沒有任何 case 測 multi-sentence punctuation。**Bench 沒覆蓋的行為就會被悄悄迴歸。** 5 個新 case 補進 fixture 之後現在有了。

---

## v0.7.5 — Voice-edit 模式 (Right Cmd / F13 熱鍵,clipboard round-trip 抓選取)

**第二個 trigger 熱鍵分支**。原本只有「按住 dictate key 講話 → 貼字」單一模式;v0.7.5 加「按住 edit key + 選取文字 → 講指令 → LLM 改寫選取」第二模式。完全不需要 Accessibility API / UI Automation — 純靠 clipboard 模擬 Cmd+C/Ctrl+C 抓選取,跨平台都可以做。

**用法**:

1. 在任何 app 選取文字
2. 按住 **edit trigger**(Win 預設 F13、Mac 預設 Right Command)講指令 — 例如「改成正式語氣」「translate to English」「縮短」「改寫成命令句」
3. 放開 → daemon 把選取文字 + 指令送進 polish LLM(同 Qwen3-4B,不同 prompt)→ 結果取代選取
4. 原 clipboard 自動還原(`try/finally` 保證)

**平台預設(per-platform,定義在 `stt_platform_{win,mac}.py` 的 `default_edit_trigger_keys`)**:

- **Win**: `{Key.f13}` — full-size 鍵盤幾乎都有 F13,不衝突任何 OS 快捷;TKL/筆電鍵盤沒 F13 需要 override
- **Mac**: `{Key.cmd_r}` — Right Command,跟 Right Option(dictate)在 space 右側對稱、所有 Mac 鍵盤都有(含 MacBook),且不會跟 Left Option 的死鍵(`Option+e` → é 等)衝突

**設定範例**(在 `scripts/stt-daemon.py` 頂端):

```python
EDIT_TRIGGER_KEYS = None                     # 用平台預設(Win: F13 / Mac: Right Cmd)
EDIT_TRIGGER_KEYS = {Key.f13}                # 強制 F13 不論平台
EDIT_TRIGGER_KEYS = {Key.cmd_r}              # 強制 Right Command(Mac)
EDIT_TRIGGER_KEYS = {Key.menu}               # Win 沒 F13 的 TKL 鍵盤可選 Menu 鍵
EDIT_TRIGGER_KEYS = set()                    # 停用 voice-edit
```

**Selection 抓不到時**:daemon 播一個 220 Hz 的「dull」失敗 beep(跟啟動 880 Hz / 結束 660 Hz 都不同),log 寫 `[stt] voice-edit: no selection captured`。常見原因:(a) 你沒選取文字、(b) focused app 不支援 Cmd+C / Ctrl+C(影像檢視器、終端輸出 pane 等)。

### 架構重點

- **Pasteboard ABC 加 3 個 method**:`get_text` / `clipboard_seqno` / `simulate_copy`。Win 用 `GetClipboardSequenceNumber` + SendInput Ctrl+C(對應現有 paste 的 Ctrl+V);Mac 用 `NSPasteboard.changeCount` + Quartz CGEvent Cmd+C(對應現有 Cmd+V),osascript fallback 處理 Accessibility 未授權狀態。**沒引入新 input library** — 跟現有 paste path 用同個機制。
- **TextPostProcessor.edit() 新介面**:`edit(selection, instruction) -> str | None`。`None` 是顯性失敗訊號(polish() 是回傳 input,但 edit 回傳 input 等於 paste 回原文 — 用戶以為沒按到 → 不清楚)。`NoopPolisher.edit() = None`(沒 polish 就沒辦法 edit)、`MlxLocalLlmPolisher` + `TorchLocalLlmPolisher` 共用新抽出的 `_run_generation` private helper(polish() 也 refactor 走同一個 helper、零回歸 — 23 個原 polish bench case 全綠)。
- **EDIT_PROMPT 雙語設計**:中文 + 英文兩段,核心規則「輸出語言預設與選取的文字相同;若指令明確要求換語言則依指令」。實測 6 個 fixture case 全 pass(中保中、英保英、中→英 explicit translate、英→中 explicit translate、formality change、shorten、識別字保留)。
- **Edit budget heuristic**:`max(256, min(3 × selection_tokens, max_tokens))`。Polish 用 1.2× 因為 minimum-edit;edit 可能 expand(「expand」「translate from Chinese to English」常常變長),3× 加 floor 256 給足空間。
- **No prefix cache for edit (v0.7.5)**:edit 的 system prompt 跟 polish 不同,要另建 cache 增加 startup 150-300 ms;edit 預估呼叫頻率比 polish 低 ~10×,amortise 不過。若實測 edit latency 不可接受再加 (`v0.7.5.1` follow-up)。

**Bench**(local-only,`pytest tests/ --run-polish-bench`):**29/29 全綠**(23 polish + 6 edit)。**0 polish regression** — `_run_generation` refactor 沒破壞任何既有行為。

**Investment**:~4 day solo(原 plan 估)。實際:**~半天** — Pasteboard ABC + Win/Mac 約 1.5 hr、polisher.edit + refactor 約 1.5 hr、daemon 接線約 1 hr、9 state-machine tests + 6 fixture case + bench loader 約 1.5 hr。

**Key-repeat hotfix**(smoke test 時抓到):Windows OS 在 F13 被按住時會持續發 key-repeat 事件(~24×/s),原版的 `_active_trigger` 早期 return 寫在 `_capture_selection` **之後**,導致每個 repeat 都跑 100 ms 選取偵測 + 模擬 Ctrl+C + 放失敗 beep + flood log。每按一次 F13 聽到 20+ 個失敗音,蓋過實際成功訊號讓用戶以為「沒運作」。修正:把早期 return hoisted 到 selection capture 之前,加 `test_edit_press_skips_capture_on_key_repeat` regression guard。

**Tests + CI**(v0.7.5 同步):新增 9 個 voice-edit state-machine test + 1 個 key-repeat regression test(hotkey 路由、selection capture、edit-mode dispatch、clipboard 還原 on success / on polish failure / on busy、key-repeat skip)+ 6 個 edit fixture case(語言保留 × 2、explicit translate × 2、formality、shorten、技術識別字保留)。整套狀態機 test 從 33 → 43 個(全 mock 不需 GPU,本地 + CI 都 ~5s)。
