"""Local web UI on top of the pipeline: a "Capture" button (and an optional
continuous ~1fps capture mode) that feeds real frames into the real ingest
path (embed -> encode -> encrypt -> upload to Railway) via a background
queue, and a text box that queries it and shows Claude's answer + the
reconstructed image.

Run with: python app.py, then open http://127.0.0.1:5000

Captured/reconstructed thumbnails are saved under static/ purely for the UI
to display -- that's on-device only and doesn't change the e2ee boundary
either way (the device already has plaintext access before encryption).

Capture and processing are decoupled: capturing a frame (~12ms) is cheap,
but fully processing one (embed + VAE encode + upload, ~2.15s on this
CPU-only machine) is not. Rather than chase per-frame compute speed (a long
investigation found ONNX/QNN-NPU/multiprocessing all failed to help on this
hardware -- see todo.md), captures are enqueued and a single background
worker thread drains the queue as fast as it can. This lets capture happen
at a steady ~1fps cadence while processing trails behind and catches up
afterward.
"""

import os
import queue
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from imgint.answer import ClaudeAnswerer
from imgint.capture import Camera, CameraError
from imgint.codec import SdxlVaeCodec
from imgint.embedder import SigLipEmbedder
from imgint.ocr import ClaudeOcrReader, RapidOcrReader
from imgint.pipeline import Pipeline
from imgint.store import Store, make_client

EMBED_DIM = 768  # google/siglip-base-patch16-224 pooled output size
CAMERA_INDEX = 1
MIN_INTERVAL_S = 0.1

# Local webcam is OFF by default -- the product direction is phone/glasses
# capture via POST /ingest, and running the app for that should never open
# your laptop camera. Set CAMERA_ENABLED=1 to use the local-webcam capture
# buttons ("Take picture" / continuous) on the desktop page.
CAMERA_ENABLED = os.environ.get("CAMERA_ENABLED", "0").lower() in ("1", "true", "yes")

# Shared secret for POST /ingest (phone/glasses/any-camera uploads). Unset =
# endpoint disabled. Plain bearer token over HTTP -- fine on a trusted LAN,
# NOT safe over the open internet (no TLS); see todo.md.
INGEST_TOKEN = os.environ.get("INGEST_TOKEN")

# Two-stage dedup, cheapest first. Both bypassed by manual/force captures.
# pHash (dHash Hamming <= threshold): near-exact/frozen-camera dupes, skipped
# before the embed even runs. Calibrated on real photos: identical=0, same
# scene=2-8, different scenes=17-33, so 4 catches frozen frames with margin.
PHASH_THRESHOLD = int(os.environ.get("PHASH_THRESHOLD", "4"))
# Semantic (embedding cosine >= threshold): "same scene" dupes that differ
# pixel-wise. Real captures: same scene 0.93-0.99, different 0.44-0.84.
DEDUP_THRESHOLD = float(os.environ.get("DEDUP_THRESHOLD", "0.95"))

# Extract text (OCR) from each kept frame at ingest and store it searchable --
# the core of the "lose nothing you read" prototype (kept-frame path only,
# after dedup). Backend:
#   claude   (default) -- vision LLM; reads handwriting/receipts/messy scenes
#             that traditional OCR mangles. ~<1c + a few seconds per frame.
#   rapidocr -- local, free, fast; good on clean printed text only.
#   none     -- disable the text channel.
OCR_BACKEND = os.environ.get("OCR_BACKEND", "claude").lower()

BASE_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = BASE_DIR / "static" / "captures"
QUERY_RESULTS_DIR = BASE_DIR / "static" / "query_results"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
QUERY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

_pipeline: Pipeline | None = None
_camera: Camera | None = None

# Every frame -- manual, continuous, or uploaded via /ingest -- goes through
# this one queue and is ingested only by _worker_loop. Giving any path its own
# synchronous ingest() call would let two threads run SigLIP+VAE inference at
# once on the same CPU, the exact contention pattern that made multiprocessing
# unreliable during latency testing (see todo.md) -- just with threads
# instead of processes. Queue items are (frame, force) tuples: force=True
# bypasses the dedup gate (deliberate captures are always kept).
_frame_queue: queue.Queue = queue.Queue()
_camera_lock = threading.Lock()
_status_lock = threading.Lock()
_stop_event = threading.Event()
_capture_thread: threading.Thread | None = None

_status = {
    "running": False,
    "interval_s": None,
    "captured_count": 0,
    "processed_count": 0,
    "skipped_count": 0,
    "failed_count": 0,
    "last_record_id": None,
    "last_error": None,
}


def get_pipeline() -> Pipeline:
    """Build the real pipeline (loads SigLIP + the VAE) once and reuse it.
    Called eagerly at startup (see __main__) so this isn't a race between
    the worker thread and Flask request threads.
    """
    global _pipeline
    if _pipeline is None:
        embedder = SigLipEmbedder()
        codec = SdxlVaeCodec()
        store = Store(make_client("http"))
        answerer = ClaudeAnswerer()
        if OCR_BACKEND == "claude":
            ocr = ClaudeOcrReader()
        elif OCR_BACKEND == "rapidocr":
            ocr = RapidOcrReader()
        else:
            ocr = None
        _pipeline = Pipeline(
            embedder,
            codec,
            store,
            answerer,
            keys_dim=EMBED_DIM,
            dedup_threshold=DEDUP_THRESHOLD,
            phash_threshold=PHASH_THRESHOLD,
            ocr=ocr,
        )
    return _pipeline


def get_camera() -> Camera:
    """Open the camera once and keep it open -- re-opening + re-warming up
    per capture (as the old capture_frame() call did here) was ~1.6s of pure
    waste on every single capture. Called eagerly at startup (see __main__).
    """
    global _camera
    if not CAMERA_ENABLED:
        raise CameraError(
            "local webcam is disabled -- capture from your phone via POST /ingest, "
            "or set CAMERA_ENABLED=1 to use the local camera"
        )
    if _camera is None:
        _camera = Camera(index=CAMERA_INDEX)
    return _camera


THUMBS_DIR = BASE_DIR / "static" / "thumbs"
THUMBS_DIR.mkdir(parents=True, exist_ok=True)


def _save_thumbnail(record_id: str, frame, width: int = 320) -> None:
    """Save a small JPEG of a just-ingested frame, named by record id, so the
    mobile 'recent'/'search' views can show it instantly without a VAE decode.
    Best-effort: a thumbnail failure must never fail the ingest.
    """
    try:
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)))
        cv2.imwrite(str(THUMBS_DIR / f"{record_id}.jpg"), frame)
    except Exception:
        pass


def _worker_loop(frame_queue, get_pipeline_fn, status, status_lock):
    """The only caller of Pipeline.ingest(). Runs forever on a daemon thread,
    started once at startup. Wrapped in try/except: without it, one bad frame
    (camera glitch, transient Chroma error) would kill the thread silently and
    processing would stop forever with no obvious symptom.

    Dependencies are passed in rather than read from module globals so tests
    can run this against throwaway queue/status objects -- a leftover daemon
    thread from a finished test then keeps looping harmlessly against objects
    only it still references, instead of fighting the next test's real state.
    """
    while True:
        frame, force = frame_queue.get()
        try:
            record_id = get_pipeline_fn().ingest(frame, force=force)
            with status_lock:
                if record_id is None:
                    status["skipped_count"] += 1
                else:
                    status["processed_count"] += 1
                    status["last_record_id"] = record_id
            if record_id is not None:
                _save_thumbnail(record_id, frame)
        except Exception as e:
            with status_lock:
                status["failed_count"] += 1
                status["last_error"] = str(e)
        finally:
            frame_queue.task_done()


def _capture_timer_loop(interval_s, frame_queue, get_camera_fn, camera_lock, status, status_lock, stop_event):
    """Enqueues a frame every interval_s seconds until stop_event is set.
    Uses stop_event.wait() instead of time.sleep() so /capture/stop is
    near-instant, and subtracts elapsed capture time so cadence stays close
    to the requested interval regardless of how long the capture itself took.
    """
    while not stop_event.is_set():
        t0 = time.monotonic()
        try:
            with camera_lock:
                frame = get_camera_fn().read()
            frame_queue.put((frame, False))
            with status_lock:
                status["captured_count"] += 1
        except CameraError as e:
            with status_lock:
                status["last_error"] = str(e)
        elapsed = time.monotonic() - t0
        stop_event.wait(max(0.0, interval_s - elapsed))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    try:
        with _camera_lock:
            frame = get_camera().read()
    except CameraError as e:
        return jsonify({"error": str(e)}), 500

    thumb_name = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(CAPTURES_DIR / thumb_name), frame)

    # force=True: a deliberate button press should always be kept, even if
    # the scene hasn't changed since the last stored frame
    _frame_queue.put((frame, True))
    with _status_lock:
        _status["captured_count"] += 1
        queue_position = _frame_queue.qsize()

    return jsonify(
        {
            "queue_position": queue_position,
            "thumbnail_url": f"/static/captures/{thumb_name}",
        }
    )


@app.route("/ingest", methods=["POST"])
def ingest():
    """Camera-less ingestion: any device that can POST an image (phone,
    glasses, a script watching a folder) feeds the same queue/worker as the
    local webcam. multipart/form-data with an `image` field; optional
    `force=1` bypasses the dedup gate. Requires `Authorization: Bearer
    <INGEST_TOKEN>`.
    """
    if not INGEST_TOKEN:
        return jsonify({"error": "ingest disabled: set INGEST_TOKEN in .env to enable"}), 503
    if request.headers.get("Authorization", "") != f"Bearer {INGEST_TOKEN}":
        return jsonify({"error": "invalid or missing bearer token"}), 401

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "multipart field 'image' is required"}), 400
    frame = cv2.imdecode(np.frombuffer(file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    force = request.form.get("force", "0").lower() in ("1", "true", "yes")

    thumb_name = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(CAPTURES_DIR / thumb_name), frame)

    _frame_queue.put((frame, force))
    with _status_lock:
        _status["captured_count"] += 1
        queue_position = _frame_queue.qsize()

    return jsonify(
        {
            "queue_position": queue_position,
            "thumbnail_url": f"/static/captures/{thumb_name}",
        }
    )


@app.route("/capture/start", methods=["POST"])
def capture_start():
    global _capture_thread

    body = request.json or {}
    interval_s = body.get("interval_s", 1.0)
    if interval_s <= MIN_INTERVAL_S:
        return jsonify({"error": f"interval_s must be > {MIN_INTERVAL_S}"}), 400

    # probe the camera up front so a disabled/unavailable webcam returns a clean
    # error here instead of silently failing on every timer tick
    try:
        get_camera()
    except CameraError as e:
        return jsonify({"error": str(e)}), 503

    with _status_lock:
        if _status["running"]:
            return jsonify({"error": "continuous capture already running"}), 409
        _status["running"] = True
        _status["interval_s"] = interval_s

    _stop_event.clear()
    _capture_thread = threading.Thread(
        target=_capture_timer_loop,
        args=(interval_s, _frame_queue, get_camera, _camera_lock, _status, _status_lock, _stop_event),
        daemon=True,
    )
    _capture_thread.start()
    return jsonify({"running": True, "interval_s": interval_s})


@app.route("/capture/stop", methods=["POST"])
def capture_stop():
    with _status_lock:
        was_running = _status["running"]
    if not was_running:
        return jsonify({"running": False})

    _stop_event.set()
    if _capture_thread is not None:
        _capture_thread.join(timeout=5.0)

    with _status_lock:
        _status["running"] = False
    return jsonify({"running": False})


@app.route("/status")
def status():
    with _status_lock:
        body = dict(_status)
    body["queue_depth"] = _frame_queue.qsize()
    return jsonify(body)


@app.route("/mobile")
def mobile():
    """Phone-friendly capture + search page. Uses a file input with
    capture=environment (opens the camera, works over plain LAN HTTP -- no
    getUserMedia/HTTPS needed), uploads to /ingest, and searches stored OCR
    text. The ingest token is rendered in so the page can authenticate;
    LAN-only, see the /ingest security note in todo.md.
    """
    return render_template("mobile.html", ingest_token=INGEST_TOKEN or "")


@app.route("/search", methods=["POST"])
def search():
    """Cheap text search over stored OCR text -- no vector, no Claude. Returns
    matching frames (reconstructed thumbnail + the text found + when). This is
    the 'find what I read' loop; use /query for a Claude answer instead.
    """
    body = request.json or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    limit = int(body.get("limit", 10))
    since = body.get("since")
    until = body.get("until")

    found = get_pipeline().search_text(text, limit=limit, since=since, until=until)
    return jsonify({"matches": [_match_payload(m) for m in found]})


def _match_payload(m: dict) -> dict:
    """Shared shape for /search and /recent results: a per-record thumbnail
    (if one was saved at ingest), the OCR text read, and when."""
    thumb = THUMBS_DIR / f"{m['id']}.jpg"
    return {
        "id": m["id"],
        "thumbnail_url": f"/static/thumbs/{m['id']}.jpg" if thumb.exists() else None,
        "text": m["text"] or "",
        "timestamp": m["timestamp"],
    }


@app.route("/recent")
def recent():
    """The last N captures with the text OCR read from each -- so you can see
    that a capture processed (and what became searchable) instead of guessing.
    """
    limit = int(request.args.get("limit", 8))
    return jsonify({"matches": [_match_payload(m) for m in get_pipeline().recent(limit=limit)]})


@app.route("/query", methods=["POST"])
def query():
    """Synthesized search: retrieves the top-k matches (optionally scoped to
    a [since, until] epoch-seconds window), and ONE Claude call answers
    across all of them -- so "what did I do this afternoon" works, not just
    "identify this single frame".
    """
    body = request.json or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    top_k = int(body.get("top_k", 3))
    since = body.get("since")
    until = body.get("until")

    pipeline = get_pipeline()
    result = pipeline.query_synthesize(question, top_k=top_k, since=since, until=until)
    if not result["matches"]:
        return jsonify({"answer": None, "message": "no matches found in that time window"})

    matches_payload = []
    for match in result["matches"]:
        image_name = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(
            str(QUERY_RESULTS_DIR / image_name),
            cv2.cvtColor(match["image"], cv2.COLOR_RGB2BGR),
        )
        matches_payload.append(
            {
                "image_url": f"/static/query_results/{image_name}",
                "distance": match["distance"],
                "timestamp": match["timestamp"],
            }
        )

    return jsonify({"answer": result["answer"], "matches": matches_payload})


if __name__ == "__main__":
    get_pipeline()
    if CAMERA_ENABLED:
        get_camera()  # eager warm-up only when the local webcam is actually in use
    threading.Thread(
        target=_worker_loop,
        args=(_frame_queue, get_pipeline, _status, _status_lock),
        daemon=True,
    ).start()
    # APP_HOST=0.0.0.0 exposes the app on the LAN so phones/other devices can
    # reach /ingest -- only do that with INGEST_TOKEN set, and only on a
    # network you trust (no TLS on the dev server).
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "5000"))
    app.run(host=host, port=port, debug=False, threaded=True)
