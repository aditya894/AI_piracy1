import os
import io
import re
import asyncio
import tempfile
import logging
import glob
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build
from telethon import TelegramClient
from telethon.errors import RPCError
from PIL import Image, UnidentifiedImageError
import pandas as pd
import aiohttp
import cv2
import numpy as np
from datetime import datetime

# ---------- Setup ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
if os.name == "nt" and "HOME" not in os.environ:
    os.environ["HOME"] = os.environ.get("USERPROFILE", os.getcwd())

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = os.getenv("CUSTOM_SEARCH_CX", "")
TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID", "0") or "0")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")

SIM_THRESHOLD_PHOTO = float(os.getenv("SIM_THRESHOLD_PHOTO", "0.67"))
SIM_THRESHOLD_FRAME = float(os.getenv("SIM_THRESHOLD_FRAME", "0.70"))
# sample frames more densely by default (2s)
FRAME_STRIDE_SEC = int(float(os.getenv("FRAME_STRIDE_SEC", "2")))

MAX_DIALOGS = int(os.getenv("MAX_DIALOGS", "80"))
MAX_MSGS_PER_DIALOG = int(os.getenv("MAX_MSGS_PER_DIALOG", "60"))
MAX_GLOBAL_MEDIA = int(os.getenv("MAX_GLOBAL_MEDIA", "300"))
TG_GLOBAL_SEARCH_LIMIT = int(os.getenv("TG_GLOBAL_SEARCH_LIMIT", "80"))

MAX_SAMPLE_SECONDS = int(os.getenv("MAX_SAMPLE_SECONDS", "10"))

app = FastAPI(title="Piracy Scanner", version="5.2")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates") if os.path.isdir("templates") else None

# ---------- Globals ----------
_models: List[SentenceTransformer] = []
try:
    _models.append(SentenceTransformer("clip-ViT-B-32"))
except Exception as e:
    logging.error(f"Model load error (ViT-B/32): {e}")
try:
    _models.append(SentenceTransformer("clip-ViT-B-16"))
except Exception as e:
    logging.warning(f"Optional model load error (ViT-B/16): {e}")

LIVE_LOGS: List[str] = []
KEYWORDS: List[str] = []
KEYPHRASES: List[str] = []
LAST_RESULTS: List[Dict[str, Any]] = []
REFERENCE_LOGO_EMBS: List[Any] = []
REFERENCE_LOGO_NAMES: List[str] = []

# Brand-specific phrases for eConceptuals
BRAND_PHRASES = [
    "conceptual radiology",
    "conceptual medicine",
    "conceptual pediatrics",
    "conceptual paediatrics",
    "econceptual",
    "econceptuals",
    "e conceptual",
    "e-conceptual",
    "conceptual video",
    "conceptual videos",
    "conceptual classes",
]


def push_log(msg: str):
    LIVE_LOGS.append(msg)
    if len(LIVE_LOGS) > 400:
        del LIVE_LOGS[: len(LIVE_LOGS) - 400]
    logging.info(msg)


def _tokenize_for_bow(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2]


def brand_hit(text: str) -> bool:
    """
    Strong textual signal that this is about eConceptuals / Conceptual Radiology/etc.
    Used to flag spam posts that sell your videos even without logo.
    """
    if not text:
        return False
    text_l = text.lower()
    # phrase-level match first
    if any(p in text_l for p in BRAND_PHRASES):
        return True
    toks = set(_tokenize_for_bow(text_l))
    if "conceptual" in toks and (
        "radiology" in toks
        or "medicine" in toks
        or "pediatrics" in toks
        or "paediatrics" in toks
    ):
        return True
    if "econceptual" in toks or "econceptuals" in toks:
        return True
    return False


def load_tabular_keywords():
    """Load bag-of-words and phrases from CSV/XLSX in root or ./data."""
    global KEYWORDS, KEYPHRASES
    KEYWORDS, KEYPHRASES = [], []
    paths: List[str] = []
    for pat in ["*.csv", "*.xlsx", "data/*.csv", "data/*.xlsx"]:
        paths.extend(glob.glob(pat))

    if not paths:
        push_log("📚 No CSV/XLSX found; will use fallbacks at query time.")

    for path in paths:
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception as e:
            push_log(f"CSV/XLSX load error for {path}: {e}")
            continue

        # include your video name columns too
        for col in [
            "title",
            "description",
            "name",
            "course",
            "keywords",
            "video_name",
            "video_title",
        ]:
            if col in df.columns:
                for val in df[col].fillna("").astype(str):
                    v = val.strip()
                    if not v:
                        continue
                    KEYPHRASES.append(v.lower())
                    KEYWORDS.extend(_tokenize_for_bow(v))

    # also seed with brand phrases
    for p in BRAND_PHRASES:
        KEYPHRASES.append(p.lower())
        KEYWORDS.extend(_tokenize_for_bow(p))

    KEYWORDS = list(dict.fromkeys(KEYWORDS))[:1000]
    KEYPHRASES = list(dict.fromkeys(KEYPHRASES))[:1000]
    push_log(
        f"📚 Loaded {len(KEYWORDS)} tokens, {len(KEYPHRASES)} phrases from tabular data"
    )


def compute_embeddings_all_models(img_bytes: bytes) -> List[Any]:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    embs: List[Any] = []
    for idx, m in enumerate(_models):
        try:
            e = m.encode(img, convert_to_tensor=True)
            embs.append(e)
            push_log(f"🔧 Logo embedded with model {idx}")
        except Exception as e:
            push_log(f"Embedding error: {e}")
    return embs


def max_similarity_score(embs_a: List[Any], embs_b: List[Any]) -> float:
    best = 0.0
    for ea in embs_a:
        for eb in embs_b:
            try:
                s = float(util.cos_sim(ea, eb).cpu().numpy()[0][0])
                if s > best:
                    best = s
            except Exception as e:
                push_log(f"Similarity error: {e}")
    return best


def kw_score(text: str) -> float:
    """Keyword score boosted strongly when brand is mentioned."""
    if not text:
        return 0.0
    text_l = text.lower()
    ph = sum(1 for p in KEYPHRASES if p in text_l)
    tokens = set(_tokenize_for_bow(text_l))
    tk = sum(1 for k in KEYWORDS if k in tokens)

    base = min(1.0, ph * 0.5 + tk / 20.0)

    if brand_hit(text_l):
        base = max(base, 0.95)  # treat brand mentions as very strong evidence

    return base


# ---------- Google Search ----------
async def fetch_image_embeddings(url: str) -> Optional[List[Any]]:
    try:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url) as r:
                if r.status == 200:
                    b = await r.read()
                    Image.open(io.BytesIO(b)).verify()
                    return compute_embeddings_all_models(b)
    except Exception as e:
        push_log(f"Google image fetch error: {e}")
    return None


def _build_google_queries() -> List[str]:
    # Use BOW phrases + filename tokens + brand phrases
    base = [
        "pirated video",
        "education video",
        "medical video",
        "course leak",
        "telegram video",
    ]

    # brand phrases
    base.extend(BRAND_PHRASES)

    # logo file names
    fn_tokens: List[str] = []
    for n in REFERENCE_LOGO_NAMES:
        fn_tokens.extend(_tokenize_for_bow(n))
    if fn_tokens:
        base.append(" ".join(fn_tokens))

    if KEYWORDS:
        base.append(" ".join(KEYWORDS[:4]))
    if KEYPHRASES:
        base.append(KEYPHRASES[0][:100])

    queries = [q.strip() for q in base if q and q.strip()]
    # keep logic same, just with extra brand terms
    return list(dict.fromkeys(queries))[:6] or ["education video pirated"]


async def search_google(ref_embs: List[Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        push_log("⚠️ Google API not configured")
        return results

    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    queries = _build_google_queries()
    push_log("🔍 Google Image Search started...")

    for q in queries:
        try:
            res = service.cse().list(
                q=q, cx=GOOGLE_CX, searchType="image", num=8
            ).execute()
            for item in res.get("items", []):
                url = item.get("link", "")
                title = item.get("title", "")
                embs = await fetch_image_embeddings(url)
                if embs is None:
                    continue
                sim = max_similarity_score(ref_embs, embs)
                kw = kw_score(title)
                if sim >= SIM_THRESHOLD_PHOTO or kw > 0.7:
                    results.append(
                        {
                            "source": "Google",
                            "title": title,
                            "link": url,
                            "channel": "",
                            "similarity": sim,
                            "kw_score": kw,
                            "score": max(sim, kw),
                            "matched_logo": REFERENCE_LOGO_NAMES,
                        }
                    )
                    push_log(
                        f"✅ Google: {title[:80]} sim={sim:.2f} kw={kw:.2f}"
                    )
        except Exception as e:
            push_log(f"Google error: {e}")
    return results


# ---------- Telegram helpers ----------
def _safe_peer_key(msg) -> Any:
    peer = getattr(msg, "peer_id", None)
    try:
        if hasattr(peer, "channel_id"):
            return ("channel", peer.channel_id, msg.id)
        if hasattr(peer, "chat_id"):
            return ("chat", peer.chat_id, msg.id)
        if hasattr(peer, "user_id"):
            return ("user", peer.user_id, msg.id)
    except Exception:
        pass
    return ("unknown", msg.id)


async def process_tg_msg(
    msg,
    ref_embs: List[Any],
    results: List[Dict[str, Any]],
    channel: str,
    counters: Dict[str, int],
):
    if counters["media"] >= MAX_GLOBAL_MEDIA:
        return

    txt = getattr(msg, "message", "") or ""
    brand = brand_hit(txt)
    kw = kw_score(txt)

    # Build link
    try:
        peer = getattr(msg, "peer_id", None)
        chan_id = getattr(peer, "channel_id", None)
        if chan_id:
            link = f"https://t.me/c/{chan_id}/{msg.id}"
        else:
            username = None
            try:
                chat = getattr(msg, "chat", None)
                username = getattr(chat, "username", None)
            except Exception:
                username = None
            if username:
                link = f"https://t.me/{username}/{msg.id}"
            else:
                link = f"https://t.me/{msg.id}"
    except Exception:
        link = f"https://t.me/{msg.id}"

    # --- Photo ---
    if getattr(msg, "photo", None):
        try:
            try:
                b = await msg.download_media(file=bytes)
            except asyncio.CancelledError:
                push_log(
                    f"⚠️ TG photo download cancelled {channel} msg {msg.id} (rate limit / disconnect)"
                )
                return

            if not b:
                return

            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                img.load()
            except (UnidentifiedImageError, OSError):
                push_log(
                    f"⚠️ TG {channel}: unsupported or corrupted photo for msg {msg.id}"
                )
                return

            embs: List[Any] = []
            for m in _models:
                try:
                    embs.append(m.encode(img, convert_to_tensor=True))
                except Exception as e:
                    push_log(f"TG photo embed error: {e}")

            sim = max_similarity_score(ref_embs, embs) if embs else 0.0

            if sim >= SIM_THRESHOLD_PHOTO or brand or kw > 0.9:
                results.append(
                    {
                        "source": "Telegram",
                        "channel": channel,
                        "title": (txt[:80] or "Photo"),
                        "link": link,
                        "similarity": sim,
                        "kw_score": kw,
                        "score": max(sim, kw),
                        "matched_logo": REFERENCE_LOGO_NAMES,
                    }
                )
                push_log(
                    f"✅ TG photo {channel} sim={sim:.2f} kw={kw:.2f} brand={brand}"
                )
            counters["media"] += 1
        except Exception as e:
            push_log(f"❌ TG photo error {channel}: {e}")

    # --- Video ---
    elif getattr(msg, "video", None):
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_name = tmp.name
            tmp.close()

            try:
                await msg.download_media(file=tmp_name)
            except asyncio.CancelledError:
                push_log(
                    f"⚠️ TG video download cancelled {channel} msg {msg.id} (rate limit / disconnect)"
                )
                return

            cap = cv2.VideoCapture(tmp_name)
            if not cap.isOpened():
                raise RuntimeError("OpenCV could not open downloaded file")

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_sec = (total_frames / fps) if (fps and total_frames) else 0

            frames_to_check = int(
                min(MAX_SAMPLE_SECONDS, max(duration_sec, MAX_SAMPLE_SECONDS)) * fps
            )

            stride = max(1, int(fps * max(1, FRAME_STRIDE_SEC)))
            best_sim = 0.0

            for idx in range(0, frames_to_check, stride):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    embs = compute_embeddings_all_models(buf.tobytes())
                    sim = max_similarity_score(ref_embs, embs) if embs else 0.0
                    if sim > best_sim:
                        best_sim = sim

            cap.release()
            cv2.destroyAllWindows()

            await asyncio.sleep(0.3)
            try:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
            except PermissionError:
                push_log(
                    f"⚠️ Couldn't delete temp video (in use): {tmp_name}"
                )

            if best_sim >= SIM_THRESHOLD_FRAME or brand or kw > 0.9:
                results.append(
                    {
                        "source": "Telegram",
                        "channel": channel,
                        "title": (txt[:80] or "Video"),
                        "link": link,
                        "similarity": best_sim,
                        "kw_score": kw,
                        "score": max(best_sim, kw),
                        "matched_logo": REFERENCE_LOGO_NAMES,
                    }
                )
                push_log(
                    f"✅ TG video {channel} sim={best_sim:.2f} kw={kw:.2f} brand={brand}"
                )
            counters["media"] += 1
        except Exception as e:
            push_log(f"❌ TG video error {channel}: {e}")

    # --- Text-only piracy (brand spam like your screenshots) ---
    elif txt and brand:
        # require brand_hit so generic job groups are filtered out
        results.append(
            {
                "source": "Telegram",
                "channel": channel,
                "title": txt[:160],
                "link": link,
                "similarity": 0.0,
                "kw_score": kw,
                "score": kw,
                "matched_logo": REFERENCE_LOGO_NAMES,
            }
        )
        push_log(f"✅ TG text {channel} brand piracy kw={kw:.2f}")


async def scan_telegram(ref_embs: List[Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        push_log("⚠️ Telegram not configured")
        return results

    client = TelegramClient("anon", TELEGRAM_API_ID, TELEGRAM_API_HASH)

    try:
        await client.start()
        push_log("📡 Telegram connected")

        base_q = " ".join((KEYWORDS[:3] or [])) or "education video"
        push_log(f"🌐 Telegram global search: {base_q}")

        counters = {"media": 0}
        seen = set()

        # 1) global text search
        try:
            async for msg in client.iter_messages(
                None, search=base_q, limit=TG_GLOBAL_SEARCH_LIMIT
            ):
                key = _safe_peer_key(msg)
                if key in seen:
                    continue
                seen.add(key)

                channel_name = ""
                try:
                    chat = getattr(msg, "chat", None)
                    channel_name = (
                        getattr(chat, "title", "")
                        or getattr(chat, "username", "")
                        or "Global"
                    )
                except Exception:
                    channel_name = "Global"

                await process_tg_msg(
                    msg, ref_embs, results, channel_name, counters
                )

                if counters["media"] >= MAX_GLOBAL_MEDIA:
                    break

                await asyncio.sleep(0.05)  # be nice to Telegram
        except asyncio.CancelledError:
            push_log(
                "⚠️ Telegram global search cancelled (rate limit / disconnect)"
            )
        except Exception as e:
            push_log(f"TG global search error: {e}")

        # 2) iterate dialogs/channels you are in
        try:
            async for dialog in client.iter_dialogs(limit=MAX_DIALOGS):
                channel_name = (
                    getattr(dialog.entity, "title", None)
                    or getattr(dialog.entity, "username", None)
                    or "Unknown"
                )
                try:
                    async for msg in client.iter_messages(
                        dialog.entity, limit=MAX_MSGS_PER_DIALOG
                    ):
                        key = _safe_peer_key(msg)
                        if key in seen:
                            continue
                        seen.add(key)

                        await process_tg_msg(
                            msg, ref_embs, results, channel_name, counters
                        )

                        if counters["media"] >= MAX_GLOBAL_MEDIA:
                            break

                        await asyncio.sleep(0.05)

                except asyncio.CancelledError:
                    push_log(
                        f"⚠️ Telegram dialog scan cancelled for {channel_name}"
                    )
                    break
                except RPCError:
                    continue

                if counters["media"] >= MAX_GLOBAL_MEDIA:
                    break
        except Exception as e:
            push_log(f"TG dialogs error: {e}")

    finally:
        try:
            await client.disconnect()
        except Exception:
            pass
        push_log("🔌 Telegram disconnected")

    return results


# ---------- Scan orchestrator ----------
async def run_scan_multi(logos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    global REFERENCE_LOGO_EMBS
    reference_embs: List[Any] = []
    for logo in logos:
        reference_embs.extend(logo["embs"])
    REFERENCE_LOGO_EMBS = reference_embs

    push_log("📌 Logos embedded. Running searches...")

    g_results = await search_google(reference_embs)
    push_log("📡 Starting Telegram scan... (bounded, no hard timeout)")

    t_results = await scan_telegram(reference_embs)

    merged = g_results + t_results
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    sources: Dict[str, int] = {}
    for r in merged:
        src = r.get("source", "Unknown")
        sources[src] = sources.get(src, 0) + 1
    for s, n in sources.items():
        push_log(f"📊 {s}: {n} positive matches")
    push_log(f"✅ Total matches: {len(merged)}")

    return merged


# ---------- Routes ----------
@app.on_event("startup")
async def startup():
    load_tabular_keywords()
    push_log("✅ System ready.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if templates:
        # template version (remember to update index.html, see below)
        return templates.TemplateResponse("index.html", {"request": request})

    # fallback inline UI
    html = """
    <html><body style='font-family:Arial;padding:2rem'>
    <h2>Piracy Scanner</h2>
    <form action="/upload_logos" method="post" enctype="multipart/form-data">
      <p>Select one or more logos (Conceptual Pediatrics / Medicine / EC etc.)</p>
      <input type="file" name="files" accept="image/*" multiple required>
      <button type="submit">Upload Logos & Scan</button>
    </form>
    <p><a href="/logs" target="_blank">Open Live Logs</a></p>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/logs", response_class=HTMLResponse)
async def get_logs():
    return HTMLResponse("<br/>".join(LIVE_LOGS[-200:]))


@app.post("/upload_logos")
async def upload_logos(files: List[UploadFile] = File(...)):
    global LAST_RESULTS, REFERENCE_LOGO_NAMES
    logos_data: List[Dict[str, Any]] = []
    REFERENCE_LOGO_NAMES = []

    if not files:
        return HTMLResponse("<h3>❌ No files uploaded</h3>", status_code=400)

    for file in files:
        content = await file.read()
        try:
            img = Image.open(io.BytesIO(content))
            img.load()  # verify
        except (UnidentifiedImageError, OSError):
            return HTMLResponse(
                f"<h3>❌ Invalid or corrupted image: {file.filename}</h3>",
                status_code=400,
            )

        REFERENCE_LOGO_NAMES.append(file.filename or "logo")
        logos_data.append(
            {"embs": compute_embeddings_all_models(content), "name": file.filename}
        )

    try:
        LAST_RESULTS = await run_scan_multi(logos_data)
    except Exception as e:
        push_log(f"❌ Fatal scan error: {e}")
        return HTMLResponse(
            "<h3>❌ Internal error during scan. Check /logs for details.</h3>",
            status_code=500,
        )

    return RedirectResponse("/results", status_code=303)


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    if templates:
        # pass logo names for display in template
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "results": LAST_RESULTS, "logos": REFERENCE_LOGO_NAMES},
        )

    # fallback inline results
    rows = []
    for r in LAST_RESULTS:
        source = r.get("source", "")
        channel = r.get("channel", "")
        title = r.get("title") or r.get("link") or ""
        link = r.get("link", "#")
        sim = float(r.get("similarity", 0.0))
        kw = float(r.get("kw_score", 0.0))
        matched_logos = (
            ", ".join(r.get("matched_logo", []))
            if isinstance(r.get("matched_logo", list))
            else r.get("matched_logo", "")
        )

        rows.append(
            f"<tr>"
            f"<td>{source}</td>"
            f"<td>{channel}</td>"
            f"<td><a href='{link}' target='_blank'>{title}</a></td>"
            f"<td>{sim:.2f}</td>"
            f"<td>{kw:.2f}</td>"
            f"<td>{matched_logos}</td>"
            f"<td><a href='/report?platform={source}&url={link}' target='_blank'>Report</a></td>"
            f"</tr>"
        )

    html = f"""
    <html><body style='font-family:Arial;padding:2rem'>
      <h2>Scan Results</h2>
      <p>Logos used: {", ".join(REFERENCE_LOGO_NAMES) or "N/A"}</p>
      <p>Total items: {len(LAST_RESULTS)}</p>
      <table border="1" cellspacing="0" cellpadding="6">
        <tr><th>Source</th><th>Channel</th><th>Item</th><th>Logo Sim</th><th>CSV/Keyword Score</th><th>Matched Logo(s)</th><th>Action</th></tr>
        {''.join(rows) if rows else '<tr><td colspan="7">No matches</td></tr>'}
      </table>
      <p style="margin-top:1rem;">
        <a href="/download_csv" target="_blank">⬇ Download report as CSV</a>
      </p>
      <p><a href="/logs" target="_blank">View Logs</a></p>
      <p><a href="/">← Back</a></p>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/download_csv")
async def download_csv():
    try:
        if not LAST_RESULTS:
            df = pd.DataFrame(
                columns=[
                    "source",
                    "channel",
                    "title",
                    "link",
                    "similarity",
                    "kw_score",
                    "score",
                    "matched_logo",
                ]
            )
        else:
            normalized = []
            for r in LAST_RESULTS:
                normalized.append(
                    {
                        "source": r.get("source", ""),
                        "channel": r.get("channel", ""),
                        "title": r.get("title") or r.get("link") or "",
                        "link": r.get("link", ""),
                        "similarity": float(r.get("similarity", 0.0)),
                        "kw_score": float(r.get("kw_score", 0.0)),
                        "score": float(r.get("score", 0.0)),
                        "matched_logo": (
                            ", ".join(r.get("matched_logo", []))
                            if isinstance(r.get("matched_logo", list))
                            else r.get("matched_logo", "")
                        ),
                    }
                )
            df = pd.DataFrame(normalized)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"piracy_scan_{ts}.csv"

        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        push_log(f"❌ CSV export error: {e}")
        return HTMLResponse(
            "<h3>❌ Error generating CSV. Check /logs.</h3>", status_code=500
        )


@app.get("/report", response_class=HTMLResponse)
async def report(request: Request, platform: str, url: str):
    plat = (platform or "").lower()

    if plat == "telegram":
        abuse_to = "abuse@telegram.org"
        subject = (
            "Unauthorized distribution of copyrighted educational content on Telegram"
        )
        intro = f"You can email Telegram Abuse at <b>{abuse_to}</b> and paste the template below."
        body = f"""
To: {abuse_to}
Subject: {subject}

Dear Telegram Abuse Team,

I am a rights holder (or representing the rights holder) for proprietary educational video content.
We have detected unauthorized distribution of our content on Telegram.

Infringing URL:
{url}

Our content is identified by the logos / brand:
{', '.join(REFERENCE_LOGO_NAMES)} (eConceptuals / Conceptual Radiology/Medicine/Pediatrics).

We request that you review this content and take appropriate action, including removal
of the infringing material and, if necessary, restriction of the infringing channel/group.

Sincerely,
[Your Name]
[Your Company]
[Contact details]
        """.strip()
    elif plat == "google":
        abuse_to = "legal@google.com"
        subject = (
            "DMCA / Unauthorized distribution of copyrighted content (Google Search result)"
        )
        intro = (
            f"For Google, you can either use their copyright form or email <b>{abuse_to}</b>."
        )
        body = f"""
To: {abuse_to}
Subject: {subject}

Dear Google Legal Team,

I am a rights holder (or representing the rights holder) for proprietary educational video content.
We have detected search results that lead to unauthorized copies of our content.

Infringing URL:
{url}

Our content is identified by the logos / brand:
{', '.join(REFERENCE_LOGO_NAMES)} (eConceptuals / Conceptual Radiology/Medicine/Pediatrics).

We request removal or de-indexing of this content from Google's services and search results.

Sincerely,
[Your Name]
[Your Company]
[Contact details]
        """.strip()
    else:
        abuse_to = "[platform abuse email]"
        subject = "Unauthorized distribution of copyrighted content"
        intro = (
            "Use the template below and send it to the abuse team of the platform."
        )
        body = f"""
To: {abuse_to}
Subject: {subject}

Infringing URL:
{url}

Our content is identified by the logos / brand:
{', '.join(REFERENCE_LOGO_NAMES)} (eConceptuals family).

[Describe your ownership and request takedown]
        """.strip()

    html = f"""
    <html><body style='font-family:Arial;padding:2rem'>
      <h2>Report Violation</h2>
      <p>Platform: {platform}</p>
      <p>URL: <a href="{url}" target="_blank">{url}</a></p>
      <p>{intro}</p>
      <h3>Email Template</h3>
      <textarea style="width:100%;height:260px;">{body}</textarea>
      <p style="margin-top:1rem;">
        You can copy the text above into your email client and send it to <b>{abuse_to}</b>.
      </p>
      <p><a href="/results">← Back to results</a></p>
    </body></html>
    """
    return HTMLResponse(html)
