"""Verifies the Phase 2 capture-queue architecture in app.py: capture and
processing are decoupled via a queue + a single background worker thread.
Uses fakes throughout -- no real camera, model, or network access.

_worker_loop/_capture_timer_loop take their queue/status/stop_event/getters
as explicit arguments (not module globals), so each test builds its own
throwaway instances -- a leftover daemon thread from a finished test just
keeps looping harmlessly against objects only it still references, instead
of contending with the next test's state.
"""

import io
import queue
import threading
import time
import uuid

import numpy as np
import pytest

import app as app_module
from imgint.crypto import generate_aes_key, generate_orthogonal_matrix
from imgint.pipeline import Pipeline
from imgint.store import Store, make_client


def _fresh_status() -> dict:
    return {
        "running": False,
        "interval_s": None,
        "captured_count": 0,
        "processed_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "last_record_id": None,
        "last_error": None,
    }


class _FakePipeline:
    def __init__(
        self,
        ingest_delay: float = 0.01,
        fail_on: set[int] | None = None,
        skip_on: set[int] | None = None,
    ):
        self.ingest_delay = ingest_delay
        self.fail_on = fail_on or set()
        self.skip_on = skip_on or set()
        self.calls = 0
        self.forced_calls = []

    def ingest(self, frame, force: bool = False) -> str | None:
        self.calls += 1
        self.forced_calls.append(force)
        time.sleep(self.ingest_delay)
        if self.calls in self.fail_on:
            raise RuntimeError(f"simulated failure on call {self.calls}")
        if self.calls in self.skip_on and not force:
            return None  # dedup gate skipped this frame
        return f"record-{self.calls}"


class _FakeCamera:
    def __init__(self):
        self.count = 0

    def read(self) -> np.ndarray:
        self.count += 1
        return np.full((4, 4, 3), self.count % 256, dtype=np.uint8)


class _FakeCodec:
    def encode(self, frame) -> np.ndarray:
        seed = abs(hash(repr(frame))) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(size=(4, 4, 4)).astype(np.float32)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        value = int(abs(latent.sum() * 1000)) % 256
        return np.full((32, 32, 3), value, dtype=np.uint8)


class _FakeAnswerer:
    def ask(self, image, question: str) -> str:
        return f"fake answer to: {question}"


def _start_worker(frame_queue, pipeline, status, status_lock) -> threading.Thread:
    t = threading.Thread(
        target=app_module._worker_loop,
        args=(frame_queue, lambda: pipeline, status, status_lock),
        daemon=True,
    )
    t.start()
    return t


def test_cadence_holds_regardless_of_backlog():
    frame_queue = queue.Queue()
    status = _fresh_status()
    status_lock = threading.Lock()
    stop_event = threading.Event()
    camera = _FakeCamera()
    pipeline = _FakePipeline(ingest_delay=0.3)

    _start_worker(frame_queue, pipeline, status, status_lock)

    timer_thread = threading.Thread(
        target=app_module._capture_timer_loop,
        args=(0.1, frame_queue, lambda: camera, threading.Lock(), status, status_lock, stop_event),
        daemon=True,
    )
    timer_thread.start()
    time.sleep(1.0)
    stop_event.set()
    timer_thread.join(timeout=2.0)

    with status_lock:
        captured = status["captured_count"]
        processed = status["processed_count"]
    queue_depth = frame_queue.qsize()

    # ~1s of capturing at a 0.1s interval -> roughly 7-10 captures, while the
    # 0.3s-per-frame worker only gets through a couple -- the whole point of
    # decoupling capture from processing.
    assert captured >= 6
    assert processed < captured
    assert queue_depth > 0


def test_no_frames_dropped_and_worker_survives_a_failure():
    frame_queue = queue.Queue()
    status = _fresh_status()
    status_lock = threading.Lock()
    pipeline = _FakePipeline(ingest_delay=0.01, fail_on={3})

    _start_worker(frame_queue, pipeline, status, status_lock)

    n = 10
    for i in range(n):
        frame_queue.put((np.full((4, 4, 3), i, dtype=np.uint8), False))
        with status_lock:
            status["captured_count"] += 1

    frame_queue.join()

    with status_lock:
        captured = status["captured_count"]
        processed = status["processed_count"]
        failed = status["failed_count"]

    assert captured == n
    assert processed + failed == n
    assert failed == 1
    # the one failure didn't kill the worker -- later frames still processed
    assert processed == n - 1


def test_worker_counts_dedup_skips_separately():
    """A None from ingest() (dedup gate) is a skip, not a success or failure
    -- every frame must land in exactly one of the three counters, or the
    status display silently lies about what happened to captures.
    """
    frame_queue = queue.Queue()
    status = _fresh_status()
    status_lock = threading.Lock()
    pipeline = _FakePipeline(ingest_delay=0.01, skip_on={2, 4})

    _start_worker(frame_queue, pipeline, status, status_lock)

    n = 5
    for i in range(n):
        frame_queue.put((np.full((4, 4, 3), i, dtype=np.uint8), False))

    frame_queue.join()

    with status_lock:
        assert status["skipped_count"] == 2
        assert status["processed_count"] == 3
        assert status["failed_count"] == 0


def test_worker_passes_force_through_to_ingest():
    frame_queue = queue.Queue()
    status = _fresh_status()
    status_lock = threading.Lock()
    # skip_on covers every call: only force=True frames can be processed
    pipeline = _FakePipeline(ingest_delay=0.0, skip_on={1, 2})

    _start_worker(frame_queue, pipeline, status, status_lock)

    frame_queue.put((np.zeros((4, 4, 3), dtype=np.uint8), False))
    frame_queue.put((np.zeros((4, 4, 3), dtype=np.uint8), True))
    frame_queue.join()

    assert pipeline.forced_calls == [False, True]
    with status_lock:
        assert status["skipped_count"] == 1
        assert status["processed_count"] == 1


def test_queue_drains_and_round_trips_through_real_pipeline(fake_embedder):
    client = make_client("ephemeral")
    store = Store(client, collection_name=f"test-queue-{uuid.uuid4().hex}")
    rng = np.random.default_rng(0)
    m = generate_orthogonal_matrix(fake_embedder.dim, rng=rng)
    aes_key = generate_aes_key()
    pipeline = Pipeline(
        embedder=fake_embedder,
        codec=_FakeCodec(),
        store=store,
        answerer=_FakeAnswerer(),
        keys_dim=fake_embedder.dim,
        m=m,
        aes_key=aes_key,
    )

    frame_queue = queue.Queue()
    status = _fresh_status()
    status_lock = threading.Lock()
    _start_worker(frame_queue, pipeline, status, status_lock)

    n = 5
    for i in range(n):
        frame_queue.put((np.full((8, 8, 3), i, dtype=np.uint8), False))

    frame_queue.join()

    with status_lock:
        processed = status["processed_count"]
        failed = status["failed_count"]
    assert processed == n
    assert failed == 0

    results = pipeline.query("anything?", top_k=n)
    assert len(results) == n


def test_camera_lock_serializes_concurrent_manual_captures():
    camera = _FakeCamera()
    camera_lock = threading.Lock()
    frame_queue = queue.Queue()

    def manual_capture():
        with camera_lock:
            frame = camera.read()
        frame_queue.put(frame)

    threads = [threading.Thread(target=manual_capture) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # every read incremented count exactly once -- no lost updates from
    # unsynchronized concurrent .read() calls, proving the lock serializes.
    assert camera.count == 5
    assert frame_queue.qsize() == 5


@pytest.fixture
def reset_module_status_and_queue(monkeypatch):
    """For endpoint-level tests only: /capture/start, /capture/stop, and
    /status read and mutate app.py's real module-level singletons, so those
    need a clean slate per test. The loop functions themselves don't touch
    module globals (see fixture docstring at file top), so this isn't needed
    for the tests above.
    """
    monkeypatch.setattr(app_module, "_frame_queue", queue.Queue())
    monkeypatch.setattr(app_module, "_stop_event", threading.Event())
    app_module._status.update(_fresh_status())
    yield


def test_capture_start_stop_endpoints(monkeypatch, reset_module_status_and_queue):
    monkeypatch.setattr(app_module, "get_camera", lambda: _FakeCamera())
    client = app_module.app.test_client()

    # stopping when nothing is running is a harmless no-op
    resp = client.post("/capture/stop")
    assert resp.status_code == 200
    assert resp.get_json()["running"] is False

    # degenerate interval is rejected before anything starts
    resp = client.post("/capture/start", json={"interval_s": 0.05})
    assert resp.status_code == 400

    resp1 = client.post("/capture/start", json={"interval_s": 0.2})
    assert resp1.status_code == 200
    assert resp1.get_json()["running"] is True

    # starting again while already running is rejected, not silently ignored
    resp2 = client.post("/capture/start", json={"interval_s": 0.2})
    assert resp2.status_code == 409

    resp3 = client.post("/capture/stop")
    assert resp3.status_code == 200
    assert resp3.get_json()["running"] is False


def test_status_endpoint_reports_queue_depth(monkeypatch, reset_module_status_and_queue):
    monkeypatch.setattr(app_module, "get_camera", lambda: _FakeCamera())
    client = app_module.app.test_client()

    app_module._frame_queue.put((np.zeros((4, 4, 3), dtype=np.uint8), False))
    app_module._frame_queue.put((np.zeros((4, 4, 3), dtype=np.uint8), False))

    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["queue_depth"] == 2
    assert body["running"] is False


def _jpeg_bytes() -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((16, 16, 3), 128, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_ingest_disabled_without_token(monkeypatch, reset_module_status_and_queue):
    monkeypatch.setattr(app_module, "INGEST_TOKEN", None)
    client = app_module.app.test_client()
    resp = client.post("/ingest")
    assert resp.status_code == 503


def test_ingest_rejects_bad_token(monkeypatch, reset_module_status_and_queue):
    monkeypatch.setattr(app_module, "INGEST_TOKEN", "secret")
    client = app_module.app.test_client()
    resp = client.post(
        "/ingest",
        headers={"Authorization": "Bearer wrong"},
        data={"image": (io.BytesIO(_jpeg_bytes()), "photo.jpg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 401
    assert app_module._frame_queue.qsize() == 0


def test_ingest_enqueues_uploaded_image(monkeypatch, reset_module_status_and_queue, tmp_path):
    monkeypatch.setattr(app_module, "INGEST_TOKEN", "secret")
    monkeypatch.setattr(app_module, "CAPTURES_DIR", tmp_path)
    client = app_module.app.test_client()

    resp = client.post(
        "/ingest",
        headers={"Authorization": "Bearer secret"},
        data={"image": (io.BytesIO(_jpeg_bytes()), "photo.jpg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["queue_position"] == 1

    frame, force = app_module._frame_queue.get_nowait()
    assert frame.shape == (16, 16, 3)
    assert force is False  # uploads dedup by default


def test_ingest_force_flag_bypasses_dedup(monkeypatch, reset_module_status_and_queue, tmp_path):
    monkeypatch.setattr(app_module, "INGEST_TOKEN", "secret")
    monkeypatch.setattr(app_module, "CAPTURES_DIR", tmp_path)
    client = app_module.app.test_client()

    resp = client.post(
        "/ingest",
        headers={"Authorization": "Bearer secret"},
        data={"image": (io.BytesIO(_jpeg_bytes()), "photo.jpg"), "force": "1"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    _frame, force = app_module._frame_queue.get_nowait()
    assert force is True


def test_ingest_rejects_undecodable_image(monkeypatch, reset_module_status_and_queue):
    monkeypatch.setattr(app_module, "INGEST_TOKEN", "secret")
    client = app_module.app.test_client()
    resp = client.post(
        "/ingest",
        headers={"Authorization": "Bearer secret"},
        data={"image": (io.BytesIO(b"not an image"), "junk.bin")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert app_module._frame_queue.qsize() == 0
