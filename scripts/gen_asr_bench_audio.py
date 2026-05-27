"""Generate Edge TTS audio fixtures for the ASR quality benchmark.

Produces 16 kHz mono WAV files in tests/fixtures/audio/ and writes the
companion fixture JSON to tests/fixtures/asr_bench_cases.json. Idempotent:
re-running overwrites existing files.

Requirements: ``pip install edge-tts``

Usage::

    python scripts/gen_asr_bench_audio.py
"""
from __future__ import annotations

import asyncio
import io
import json
import struct
import wave
from pathlib import Path


VOICE = "zh-TW-HsiaoChenNeural"
SAMPLE_RATE = 16000
REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "tests" / "fixtures" / "audio"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "asr_bench_cases.json"

CASES: list[dict] = [
    {
        "id": "short_single_syllable",
        "category": "short_utterance",
        "text": "好",
        "ground_truth": "好",
        "must_contain": ["好"],
        "must_not_contain": [],
        "note": "最短單音節回應",
    },
    {
        "id": "short_agreement",
        "category": "short_utterance",
        "text": "對啊就是這樣",
        "ground_truth": "對啊就是這樣",
        "must_contain": ["對"],
        "must_not_contain": [],
        "note": "短句口語回應",
    },
    {
        "id": "short_request",
        "category": "short_utterance",
        "text": "幫我改一下",
        "ground_truth": "幫我改一下",
        "must_contain": ["幫我"],
        "must_not_contain": [],
        "note": "短句請求",
    },
    {
        "id": "codesw_review_commit",
        "category": "code_switching",
        "text": "幫我review這個commit然後push到main",
        "ground_truth": "幫我review這個commit然後push到main",
        "must_contain": ["review", "commit", "push", "main"],
        "must_not_contain": ["提交", "推送"],
        "note": "中英混合技術動詞 — 英文關鍵字不應被翻譯",
    },
    {
        "id": "codesw_python_function",
        "category": "code_switching",
        "text": "這個Python function的async部分需要改",
        "ground_truth": "這個Python function的async部分需要改",
        "must_contain": ["Python", "function", "async"],
        "must_not_contain": ["函式", "函數", "非同步"],
        "note": "英文技術關鍵字必須保留原文",
    },
    {
        "id": "codesw_prebuilt_wheel",
        "category": "code_switching",
        "text": "找不到prebuilt wheel只好自己compile",
        "ground_truth": "找不到prebuilt wheel只好自己compile",
        "must_contain": ["prebuilt", "wheel", "compile"],
        "must_not_contain": ["預建", "輪子", "編譯"],
        "note": "英文技術名詞不應音譯或翻譯",
    },
    {
        "id": "pure_zh_problem",
        "category": "pure_chinese",
        "text": "我剛剛在測試這個工具的過程中，發現了一個小問題。",
        "ground_truth": "我剛剛在測試這個工具的過程中，發現了一個小問題。",
        "must_contain": ["測試", "工具", "問題"],
        "must_not_contain": [],
        "note": "純中文多句含標點",
    },
    {
        "id": "pure_zh_multi_sentence",
        "category": "pure_chinese",
        "text": "目前我輸出的文字都是透過這個工具來輸出。你可以發現前面的句子是沒有標點符號，後面又突然出現標點符號，請幫我分析一下。",
        "ground_truth": "目前我輸出的文字都是透過這個工具來輸出。你可以發現前面的句子是沒有標點符號，後面又突然出現標點符號，請幫我分析一下。",
        "must_contain": ["輸出", "標點符號", "分析"],
        "must_not_contain": [],
        "note": "長段落多句含句號 — 標點保留是關鍵指標",
    },
    {
        "id": "tech_identifier",
        "category": "technical",
        "text": "把USE TORCH COMPILE設成True試試",
        "ground_truth": "把_USE_TORCH_COMPILE設成True試試",
        "must_contain": ["TORCH", "COMPILE", "True"],
        "must_not_contain": [],
        "note": "技術識別符 — TTS 無法唸底線，ground truth 含底線用於 CER 參考",
    },
    {
        "id": "tech_numbers",
        "category": "technical",
        "text": "polish從4.26秒降到1.9秒",
        "ground_truth": "polish從4.26秒降到1.9秒",
        "must_contain": ["4.26", "1.9"],
        "must_not_contain": [],
        "note": "數字精度保留 — ASR 不應改動數字",
    },
    {
        "id": "tech_pytorch_transformers",
        "category": "code_switching",
        "text": "PyTorch跟transformers都要先裝好",
        "ground_truth": "PyTorch跟transformers都要先裝好",
        "must_contain": ["PyTorch", "transformers"],
        "must_not_contain": [],
        "note": "品牌/套件名必須保留原文",
    },
    {
        "id": "punct_question",
        "category": "punctuation",
        "text": "你可以透過GH CLI來確認pipeline的執行狀況嗎？",
        "ground_truth": "你可以透過GH CLI來確認pipeline的執行狀況嗎？",
        "must_contain": ["GH CLI", "pipeline"],
        "must_not_contain": [],
        "note": "問號保留 + 英文術語保留",
    },
    {
        "id": "punct_mixed_comma_question",
        "category": "punctuation",
        "text": "是我需要有講話有停頓，然後才會有標點符號，還是其他原因？",
        "ground_truth": "是我需要有講話有停頓，然後才會有標點符號，還是其他原因？",
        "must_contain": ["停頓", "標點符號"],
        "must_not_contain": [],
        "note": "逗號 + 問號混合標點",
    },
    {
        "id": "long_technical_paragraph",
        "category": "long_form",
        "text": "我們目前的架構是用Qwen3 ASR做語音辨識，然後接一個Qwen3 4B的polish模型做後處理。整個pipeline的延遲大概在3到5秒左右，主要瓶頸在decoder。我想評估看看MediaTek的Breeze ASR模型是否能改善辨識品質，特別是在中英混合的場景。",
        "ground_truth": "我們目前的架構是用Qwen3 ASR做語音辨識，然後接一個Qwen3 4B的polish模型做後處理。整個pipeline的延遲大概在3到5秒左右，主要瓶頸在decoder。我想評估看看MediaTek的Breeze ASR模型是否能改善辨識品質，特別是在中英混合的場景。",
        "must_contain": ["Qwen3", "ASR", "pipeline", "decoder", "MediaTek", "Breeze"],
        "must_not_contain": [],
        "note": "長段落技術描述 — 多個英文術語 + 多句結構",
    },
    {
        "id": "fact_int4_slower",
        "category": "semantic",
        "text": "INT4量化在這台機器上反而更慢",
        "ground_truth": "INT4量化在這台機器上反而更慢",
        "must_contain": ["INT4", "更慢"],
        "must_not_contain": ["更快"],
        "note": "語義保真 — 「更慢」不可被反轉為「更快」",
    },
    {
        "id": "subject_help_me",
        "category": "semantic",
        "text": "幫我改一下這段程式碼",
        "ground_truth": "幫我改一下這段程式碼",
        "must_contain": ["幫我", "程式碼"],
        "must_not_contain": ["幫你"],
        "note": "主詞保留 — 「幫我」不可變成「幫你」",
    },
]


async def _generate_one(text: str, out_path: Path) -> None:
    """Generate a single WAV file via Edge TTS, resampled to 16 kHz mono."""
    import edge_tts

    communicate = edge_tts.Communicate(text, VOICE)
    mp3_buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buf.write(chunk["data"])
    mp3_buf.seek(0)

    _mp3_to_wav_16k(mp3_buf.read(), out_path)


def _mp3_to_wav_16k(mp3_bytes: bytes, out_path: Path) -> None:
    """Decode MP3 → resample to 16 kHz mono WAV using pydub or ffmpeg."""
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
        seg.export(str(out_path), format="wav")
        return
    except ImportError:
        pass

    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(mp3_bytes)
        tmp_mp3 = tmp.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_mp3,
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        Path(tmp_mp3).unlink(missing_ok=True)


async def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    fixture_cases = []

    for case in CASES:
        wav_name = f"{case['id']}.wav"
        wav_path = AUDIO_DIR / wav_name
        print(f"  generating {wav_name} ... ", end="", flush=True)
        await _generate_one(case["text"], wav_path)
        print(f"OK ({wav_path.stat().st_size:,} bytes)")

        fixture_cases.append({
            "id": case["id"],
            "category": case["category"],
            "audio_path": f"tests/fixtures/audio/{wav_name}",
            "ground_truth": case["ground_truth"],
            "must_contain": case["must_contain"],
            "must_not_contain": case["must_not_contain"],
            "note": case["note"],
        })

    fixture = {
        "_meta": {
            "purpose": "ASR quality benchmark fixture — Breeze-ASR-25 vs Qwen3-ASR-0.6B evaluation.",
            "audio_source": f"Edge TTS ({VOICE}), resampled to 16 kHz mono WAV.",
            "schema": {
                "id": "stable case id",
                "category": "short_utterance | code_switching | pure_chinese | technical | punctuation | long_form | semantic",
                "audio_path": "relative path from repo root to WAV file",
                "ground_truth": "expected transcription (reference text)",
                "must_contain": "substrings the ASR output MUST contain",
                "must_not_contain": "substrings that MUST NOT appear",
                "note": "human-readable rationale",
            },
        },
        "cases": fixture_cases,
    }

    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=2)
    print(f"\nFixture written: {FIXTURE_PATH}")
    print(f"Audio files:     {AUDIO_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
