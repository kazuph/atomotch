"""STT + Gemini server: Speech-to-Text -> LLM response on port 8002."""
import io
import logging
import os
import re
import time
from collections import defaultdict, deque

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt")

app = FastAPI(title="STT+LLM Server", version="2.6.0")

_whisper_model = None
_gemini_client = None

MAX_HISTORY = 5
_conversation_history: dict[str, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

CHARACTER_PROMPTS = {
    "アンパンボーヤ": "あなたはアンパンボーヤです。正義の味方で、困っている人を助けるのが大好き。明るく優しい口調で、子供に話しかけるように短く返答してください。",
    "はやぶさ": "あなたは新幹線はやぶさ（E5系）のキャラクターです。速いものが大好きで元気いっぱい。電車や乗り物の話が得意。子供に話しかけるように短く返答してください。",
    "もこ": "あなたはうさぎモチーフの天然キャラ『もこ』です。のんびりしていてちょっと抜けてるけど、心は温かい。ふわふわした口調で短く返答してください。",
}
DEFAULT_PROMPT = "あなたはゴッチ風の育成ペットのキャラクターです。子供に話しかけるように、短く楽しく返答してください。"

COMMON_RULES = """
【絶対ルール】
- 返答は30文字以内の日本語のみ。
- 絵文字・顔文字・特殊記号は絶対に使わないこと。ひらがな・カタカナ・漢字・数字・句読点（、。！？）のみ使用可。
- しりとりを求められたら単語1つだけ返す。
- 人物・ニュース・時事問題の質問には必ずGoogle検索で最新情報を確認してから答えること。
- 今日は2026年2月です。古い情報で答えないこと。
"""

GEMINI_TOOLS = [types.Tool(google_search=types.GoogleSearch())]

# 絵文字除去: 実際の絵文字範囲のみ（CJK拡張漢字を含まないよう注意）
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U0001F1E0-\U0001F1FF"  # Flags (Enclosed)
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U000024FF"  # Enclosed Alphanumerics (subset)
    "\u2640-\u2642"          # Gender symbols
    "\u2600-\u2B55"          # Misc symbols
    "\u200d"                 # Zero-width joiner
    "\u23cf"                 # Eject symbol
    "\u23e9-\u23f3"          # Media control
    "\u231a-\u231b"          # Watch/Hourglass
    "\ufe0f"                 # Variation selector
    "\u3030"                 # Wavy dash
    "\u2934-\u2935"          # Arrows
    "\u25aa-\u25ab"          # Squares
    "\u25fb-\u25fe"          # Squares
    "\u2b05-\u2b07"          # Arrows
    "\u2b1b-\u2b1c"          # Squares
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


def get_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    logger.info("Loading faster-whisper large-v3 device=%s compute=%s", device, compute)
    _whisper_model = WhisperModel("large-v3", device=device, compute_type=compute)
    logger.info("Whisper model loaded")
    return _whisper_model


def get_gemini():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set - LLM responses disabled")
        return None
    _gemini_client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized")
    return _gemini_client


def ask_gemini(user_text: str, character: str = "") -> str:
    client = get_gemini()
    if not client:
        return user_text

    char_prompt = CHARACTER_PROMPTS.get(character, DEFAULT_PROMPT)
    history_key = character or "__default__"
    history = _conversation_history[history_key]

    # system_instruction: キャラ設定 + ルール + 会話履歴
    system_parts = [char_prompt, COMMON_RULES]
    if history:
        system_parts.append("【最近の会話】")
        for entry in history:
            system_parts.append(f"子供「{entry['user']}」 あなた「{entry['reply']}」")
    system_text = "\n".join(system_parts)

    # contents: ユーザーの発話（フレーミング付きで確実に応答させる）
    contents_text = f"子供が話しかけています: 「{user_text}」\nキャラクターとして日本語で返答してください。"

    try:
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents_text,
            config=types.GenerateContentConfig(
                system_instruction=system_text,
                tools=GEMINI_TOOLS,
            ),
        )
        elapsed = time.perf_counter() - t0
        reply = response.text.strip() if response.text else ""

        reply = strip_emoji(reply)
        # 30文字制限の強制（超えてたら切る）
        if len(reply) > 30:
            reply = reply[:30]

        # 空応答フォールバック: Geminiが何も返さない場合
        if not reply:
            logger.warning("Gemini returned empty reply for input=%s, using fallback", user_text[:40])
            reply = "うん、きいてるよ！"

        history.append({"user": user_text, "reply": reply})

        cand = response.candidates[0] if response.candidates else None
        gm = getattr(cand, "grounding_metadata", None)
        searched = False
        if gm:
            queries = getattr(gm, "web_search_queries", None)
            searched = queries is not None and len(queries) > 0
            if searched:
                logger.info("Gemini SEARCHED: %s", queries)

        logger.info("Gemini %.3fs char=%s hist=%d searched=%s input=%s reply=%s",
                     elapsed, character, len(history), searched, user_text[:40], reply[:60])
        return reply
    except Exception as exc:
        logger.exception("Gemini failed")
        return user_text


def transcribe(content: bytes, language: str = "ja"):
    model = get_whisper()
    segments, info = model.transcribe(
        audio=io.BytesIO(content),
        beam_size=5,
        language=language or None,
        vad_filter=True,
        without_timestamps=True,
    )
    text = "".join(seg.text for seg in segments).strip()
    return text, info.language


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/stt")
async def stt(file: UploadFile = File(...), language: str = Form("ja")):
    t0 = time.perf_counter()
    try:
        content = await file.read()
        if len(content) < 44:
            raise HTTPException(400, "Audio too short")
        text, lang = transcribe(content, language)
        elapsed = time.perf_counter() - t0
        logger.info("STT %.3fs lang=%s text=%s", elapsed, lang, text[:80])
        return {"text": text, "language": lang, "elapsed": round(elapsed, 3)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("STT failed")
        raise HTTPException(500, f"STT failed: {exc}")


@app.post("/v1/stt-raw")
async def stt_raw(request: Request):
    t0 = time.perf_counter()
    try:
        content = await request.body()
        if len(content) < 44:
            raise HTTPException(400, "Audio too short")

        character = request.headers.get("X-Character", "")
        logger.info("STT-raw: %d bytes, character=%s", len(content), character)

        stt_text, lang = transcribe(content, "ja")
        stt_elapsed = time.perf_counter() - t0
        logger.info("STT: %.3fs text=%s", stt_elapsed, stt_text[:80])

        if not stt_text:
            return {"text": "", "stt": "", "language": lang, "elapsed": round(stt_elapsed, 3)}

        reply = ask_gemini(stt_text, character)
        total_elapsed = time.perf_counter() - t0
        logger.info("Total %.3fs stt=%s reply=%s", total_elapsed, stt_text[:40], reply[:60])

        return {"text": reply, "stt": stt_text, "language": lang, "elapsed": round(total_elapsed, 3)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("STT-raw failed")
        raise HTTPException(500, f"STT failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    get_whisper()
    get_gemini()
    uvicorn.run(app, host="0.0.0.0", port=8002)
