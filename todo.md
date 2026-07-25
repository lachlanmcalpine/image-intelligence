# todo.md — deliberate MVP shortcuts to revisit

- **[FIXED, not an open item] Chroma collection was silently using L2 distance instead of
  cosine.** `Store.__init__` created the "frames" collection without specifying `hnsw:space`,
  so Chroma defaulted to L2 (squared Euclidean). `transform_embedding()` (`imgint/crypto.py`)
  only preserves **cosine** similarity under the orthogonal-matrix transform, and SigLIP
  embeddings aren't unit-normalized, so L2 nearest-neighbor search silently ranked matches
  differently than the whole design assumes. Caught live: a query returned `distance: 643.5`
  (a plausible squared-L2 value, not a cosine distance/similarity, which is bounded ~[0,2]) and
  the wrong photo. Fixed by explicitly passing `metadata={"hnsw:space": "cosine"}` in
  `Store.__init__`. Since Chroma fixes `hnsw:space` at collection-creation time and ignores this
  metadata on an already-existing collection, the live 81-record Railway collection had to be
  migrated (`scripts/dev_fix_chroma_distance_space.py`: back up every record, delete the
  collection, recreate with cosine space, re-insert all 81 unchanged — no photos lost). Added
  `tests/test_store.py::test_ranks_by_cosine_similarity_not_l2_distance` as a regression test,
  since the existing tests all queried with near-identical vectors that pass under either metric
  and never actually exercised the distinction.

- **Multi-tenant scaling design (future — no multi-user code exists yet).** From planning a
  hypothetical 5fps/512² continuous-capture deployment on a rented GPU, three related decisions
  worth preserving before they're needed:
  - **Each user MUST get their own separate Chroma database/collection, never a shared collection
    filtered by a user-id field.** This is a hard requirement, not a nice-to-have: it's the
    mechanism that guarantees one user's ANN search can't ever surface another user's photos (a
    shared-collection-plus-filter design relies on every query remembering to filter correctly —
    one bug away from a real privacy leak), and it makes per-user cost/usage tracking and billing
    straightforward (delete/suspend/measure one user = delete/suspend/measure one database, no
    cross-user accounting). Build this in from the start of any multi-user work, not bolted on
    after.
  - **Product quantization (PQ) likely removes the need for hot/cold time-tiering, at least at
    personal-to-small-multi-user scale.** Implemented in `imgint/pq_compression.py` (numpy-only
    k-means codebooks, no new dependency): compresses a 768-dim fp32 embedding (3,072 bytes) to
    `num_subvectors` bytes (8 bytes at the default 8 subvectors — ~384x smaller), lossy. Verified
    with `tests/test_pq_compression.py`, including the property that actually matters for
    retrieval (not just "does it compress"): ranking against true cosine similarity is preserved
    for well-separated clusters — the same category of bug as the L2-vs-cosine issue above, so
    it's tested the same way. Math: 5fps continuous for one user is ~13M vectors/month raw
    (~40GB/month fp32) but only ~104MB/month PQ-compressed — cheap enough to just keep everything
    resident indefinitely, no need to evict old data from RAM. **Not yet wired into live
    retrieval** — see the next entry for why.
  - **PQ codes can't be searched by Chroma as-is.** Chroma's HNSW index expects real float
    vectors to compute distances on; it has no concept of "these bytes are PQ codes, decode
    before comparing." Storing raw PQ codes as a Chroma embedding would make its ANN search
    meaningless (same failure shape as the L2-vs-cosine bug, for a different reason). Live
    integration needs an index that understands PQ natively — FAISS's `IndexIVFPQ`, or
    Milvus/Weaviate/Qdrant's built-in quantization support — which means a store migration, not a
    drop-in config change. Until multi-user/high-volume scale actually arrives, keep storing raw
    (or int8-scalar-quantized, cheaper to adopt) embeddings in Chroma as today.
  - **Hot/cold time-tiering (day/week-sharded collections, spin old shards in from cold storage
    on demand) is still worth keeping in mind, not deleted as an idea** — PQ raises the point at
    which it becomes necessary by ~380x, but doesn't remove the underlying fact that truly
    unbounded retention × many users eventually costs something. The two techniques compose
    (PQ-compress *within* a hot tier too) rather than compete; revisit tiering specifically if
    PQ's headroom ever actually gets exhausted, rather than building it preemptively.

- **OCR backend is Claude (vision LLM) by default, not local RapidOCR — a real cost/quality
  call.** RapidOCR (local, free, fast) was the first backend but failed on real phone captures:
  it read a handwritten whiteboard as "baby J Job -fum on dil" and garbled an angled receipt.
  Traditional OCR is trained on clean printed text; handwriting, faint/angled thermal receipts,
  and stylized labels are exactly where it breaks — and exactly what a user photographs. Claude
  reads all of them near-perfectly (transcribed a full pizza receipt line-by-line incl. Total
  $43.00, and the whiteboard exactly incl. "I love you handsome"), and also returns a one-line
  description so search works for "receipt"/"whiteboard" too. Cost: ~<1c + ~3-12s per KEPT frame
  (after dedup). Fine for tap-to-capture personal use (dozens/day = cents/day); NOT viable for
  continuous 5fps (thousands of frames = real money + the query-time e2ee exception now also
  applies at *capture* time — the raw frame goes to Anthropic on every kept capture, not just at
  query). `OCR_BACKEND=rapidocr` switches back to the free/local reader for clean-text or
  cost-sensitive use; `OCR_BACKEND=none` disables. For continuous capture later: use RapidOCR (or
  a cheap detector) as a filter and only spend a Claude read on frames that look text-heavy, or
  do the read on-device once a local VLM is fast enough.

- **Text search filters client-side (fetches all docs) — prototype-scale only.** `get_by_text`
  and `get_recent` fetch every record's document and filter/sort in Python, because Chroma's
  `where_document {"$contains"}` is case-SENSITIVE and that caused a real live "nothing found"
  (searching "BABY" missed stored "baby"). Client-side lowercasing fixed it and is fine at
  personal-prototype scale (hundreds of records), but it's O(all records) per query — at scale
  this must move to a proper case-insensitive full-text index (pgvector `tsvector`, or a
  lowercased indexed field). Tied to the pgvector migration already noted.

- **OCR text is stored server-side as plaintext — a deliberate exception to "server sees
  nothing."** The OCR channel (`imgint/ocr.py`, RapidOCR) extracts text from each kept frame at
  ingest, on the RAW full-res frame *before* the lossy VAE (small text is exactly what the
  autoencoder discards, so it can't go through the latent), and stores it as Chroma's document
  field so `search_text`/`/search` can substring-match it with no vector and no Claude call. That
  text sits unencrypted on the Railway server, unlike the image (AES-encrypted) and the embedding
  (orthogonal-transformed). Accepted for the prototype because text is far lower-stakes than the
  image and the whole point is a cheap searchable index — but it IS a real widening of what the
  server can see (a receipt total, a name, a screen's contents are now readable server-side).
  Before anything real: either encrypt the document too and do text search client-side (loses
  Chroma's `where_document`), or keep the text on a local index and only the image/embedding in
  the cloud. Also note OCR adds ~2-5s/frame on CPU (kept-frame path only, after dedup) and, like
  everything, benefits from the sharper 1920×1080 capture — the 640×480 version of the same
  Opal-card photo OCR'd to nothing, the 1080p version read "opal" cleanly.

- **`/ingest` bearer token travels over plain HTTP — LAN-only until TLS exists.** The new
  `POST /ingest` endpoint (phone/glasses/any-device uploads into the same queue/worker as the
  webcam) authenticates with a static bearer token from `.env`'s `INGEST_TOKEN`, sent unencrypted
  because the Flask dev server has no TLS. Fine on a trusted home LAN (`APP_HOST=0.0.0.0` to
  expose it); NOT safe over the open internet — anyone on-path can read the token and the images.
  Before any remote/cellular use: a reverse proxy with TLS (Caddy makes this nearly free) or a
  tunnel (Tailscale would also solve it with zero code). Also note the token is all-or-nothing —
  one shared secret, no per-device identity or revocation.

- **Dedup threshold (0.95) calibrated on a handful of photos, not a real capture stream.**
  `Pipeline.ingest()` now skips frames whose embedding cosine-similarity to the last *kept* frame
  is ≥ `DEDUP_THRESHOLD` (default 0.95, env-tunable; manual captures and `force=1` uploads
  bypass). Calibration data (real photos, this machine): same scene seconds apart = 0.93–0.99,
  different scenes = 0.44–0.84 — 0.95 sits above the different-scene ceiling with margin. But the
  measured near-dupe pairs spanned *resolution changes* (which depress similarity); consecutive
  same-camera frames should score higher, and no long-running continuous stream has been measured
  yet. Revisit the threshold after the first real multi-hour capture session — the skipped-count
  in `/status` vs. reality is the check. Note the gate only compares against the *last kept*
  frame, not all history — A→B→A oscillation (looking away and back) stores every swing.

- **Hold-then-promote architecture (from design discussions, not yet built).** Today every
  non-deduped frame is fully processed (embed + VAE encode + upload) and stored forever. The
  agreed better shape for continuous capture: a cheap rolling raw buffer (dashcam-style, hold
  minutes-to-hours), with only *selected* moments promoted through the expensive
  encode/encrypt/store path — selection via dedup, scene-change spikes, explicit user action, or
  a periodic summarization pass. This is what makes 5fps continuous capture affordable (the GPU
  encodes a small promoted fraction, not the full stream) and it's the natural home for
  day-summary generation. The dedup gate just built is the first slice of it; the raw buffer and
  promotion policy are not started.

- **Capture queue is in-memory and unbounded.** `app.py`'s `_frame_queue` (added when capture
  was decoupled from processing to allow a ~1fps capture cadence while processing trails behind)
  has no `maxsize` and isn't persisted — queued raw frames are lost if the process crashes before
  they're processed. If continuous capture runs unattended for a long time, the backlog of
  held-in-memory raw frames grows roughly (capture rate − processing rate) frames/sec net — a
  slow but real memory-growth risk on long unattended runs, since processing (~0.47 fps on this
  CPU) is slower than a typical ~1 fps capture cadence. Not fixed now (would need a bounded-queue
  + drop policy, or backpressure that slows capture to match processing); revisit if this is ever
  left running unattended for hours, or if it moves to hardware fast enough that this never
  triggers.

- **Downgraded embedding model — re-measured, still not adopting SO400M.** Using
  `google/siglip-base-patch16-224` (813 MB) instead of `google/siglip-so400m-patch14-384`
  (3.51 GB). Directly measured both on this machine (`scripts/dev_compare_siglip_models.py`):
  SO400M's steady-state embed latency is **15.3x slower** (4.66s vs. 0.30s/frame) for only a
  marginal retrieval-quality gain (cross-modal correct-vs-next-best margin +0.099 vs. +0.087 —
  a small, not dramatic, improvement) and +1.5 KB/image storage (1152-dim vs. 768-dim). Since
  VAE encode already dominates per-frame latency, adding a 4.66s embed step would roughly
  triple total capture-to-searchable time for a small quality win. Not worth it on this
  hardware. Swapping is still a one-line `model_id` change in `imgint/embedder.py` — revisit if
  this ever runs on hardware where SO400M's cost stops mattering.

- **VAE encode resolution: 128² is CPU-only-hardware-only — MUST move to 256² or 512² once off
  this machine, not just a "maybe."** Measured 128²/256²/512² side by side on this CPU-only
  machine (`scripts/dev_compare_vae_resolution.py`, results reviewed visually via a generated
  comparison page): encode latency 558.8ms / 1927.0ms / 6253.2ms, full cloud payload 4,630B /
  8,246B / 23,074B, PSNR 30.72 / 35.93 / 39.01 dB. Chose **128²** for the ~3.5x latency win and
  ~44% smaller payload vs. the previous 256² default, accepting the real quality cost (5.2 dB
  lower PSNR, visibly softer reconstructions) as the right trade *only because this CPU-bound
  prototype is bottlenecked on VAE encode time*. That justification evaporates the moment compute
  moves to a cloud/rented GPU (see the GPU-acceleration entry below) — encode latency stops being
  the constraint, so there is no remaining reason to accept 128²'s quality loss. **Action item for
  that migration: bump `imgint/codec.py`'s `TARGET_SIZE` back to 256 (or 512, re-measure PSNR/
  latency on the GPU first to pick between them) as part of the same change**, not something to
  leave at 128 out of inertia.

- **Center-crop-to-square permanently discards part of the camera's field of view — now ~21.9%
  per side, up from 12.5%.** `resize_short_side_and_center_crop()` — used by every real capture,
  not just dev scripts — squares off whatever the camera delivers, which always crops the long
  side symmetrically regardless of `TARGET_SIZE` (the fraction is fixed by the aspect ratio, not
  the target resolution). At the old 640×480 (4:3) default that was 12.5%/side (25% total); at
  the now-default 1920×1080 (16:9, see the capture-resolution entry below) it's **~21.9%/side
  (~43.75% total)** — confirmed by direct inspection, a real captured photo's edge content is
  missing from the cropped/encoded version. **Accepted as a known trade-off** (kept
  center-cropping rather than switching to a non-square encode or letterboxing) in exchange for
  the sharpness win below — but this is a real, structural gap in what this "visual recall"
  system can ever recall: anything only visible at the left/right edges of the camera's view is
  discarded before the VAE ever sees it, and that fraction is now bigger than it was. Revisit if
  edge content ever turns out to matter (e.g. switching `codec.py` to encode a non-square,
  aspect-matching shape instead of forcing a square, or padding instead of cropping).

- **Capture resolution bumped to 1920×1080 (from a 640×480 default) — this was the real fix for
  "why are all the images blurry."** The webcam was never asked for a resolution, so the MSMF
  driver fell back to 640×480 VGA. Direct side-by-side capture (`scripts/dev_cross_matrix.py`)
  confirmed the sensor natively supports up to ~2560×1440, and reconstructions sourced from
  1920×1080 are visibly sharper than 640×480 at every VAE encode size tested — even though PSNR
  (measured against each capture's own reference crop) goes slightly *down* as capture resolution
  goes up. That's not a regression: a sharper native photo is a harder signal for the VAE's
  fixed-size latent bottleneck to losslessly reproduce, so it scores marginally worse against its
  own sharper reference while still looking better in absolute terms. `imgint/capture.py` now
  requests `DEFAULT_WIDTH`/`DEFAULT_HEIGHT` = 1920×1080 at open time (must be set before the
  first `read()` — MSMF can't reliably switch resolution on an already-streaming capture, it
  raises a Mat assertion error). Trade-off: 1920×1080 is 16:9, not the old 640×480's 4:3, so the
  center-crop-to-square now discards more of the frame (see the entry above).

- **No real-time throughput; real GPU acceleration deferred.** This MVP captures on-demand / at
  whatever rate CPU allows, not continuous 5 FPS. Investigated getting there on this machine's
  own hardware (camera-handling fix, INT8 quantization, ONNX export, the Hexagon NPU via QNN,
  multiprocessing) — camera fix was a big real win (~6s → ~2.6s/frame), everything else either
  didn't help or made things worse (ONNX Runtime and the NPU were both *slower* than plain
  PyTorch for these models on this ARM64-under-x64-emulation CPU; multiprocessing capped out
  around 0.22 fps and was unreliable, likely thermal throttling under sustained load). The
  clearest remaining path to real 5 fps is moving embed + VAE-encode to a **cloud GPU instance**
  (rented, e.g. RunPod/Lambda/AWS/GCP/Azure/Vast.ai) — a GPU would likely cut VAE encode from
  ~1.8s to tens of milliseconds. Deliberately not done yet because it's a real e2ee tradeoff, not
  just a speed tweak: today nothing leaves the device unencrypted except the one accepted
  query-time exception (Claude sees the reconstructed photo). Moving compute to a cloud GPU means
  the **raw captured frame** travels there on *every capture*, not just occasional queries — a
  bigger, continuous exposure. Somewhat mitigated if it's a cloud instance the user personally
  rents/controls (closer to "own compute elsewhere" than "a new company sees your data," similar
  in spirit to already trusting Railway with the encrypted DB) — but still a deliberate call to
  make, not something to wire in silently. If pursued: a small FastAPI server on the GPU box
  serving just embed+encode over HTTPS; local pipeline sends the raw frame, gets back the
  embedding+latent, then does encryption + Railway upload locally as today.

- **Key storage is a plain local file.** The AES key and orthogonal matrix `M` live under
  `./keys/` as plain files. Consider Windows DPAPI/Credential Manager if this ever moves beyond
  a single-user prototype — this is the entire confidentiality boundary for the embeddings.

- **Running under x64 emulation, not native ARM64.** The Python 3.13 install on this machine
  is the x86_64 (AMD64) build, running under Windows-on-ARM's Prism emulator (confirmed via
  `sys.version` showing `MSC v.1943 64 bit (AMD64)` despite `platform.machine()` correctly
  reporting the true ARM64 hardware). All packages installed are ordinary win_amd64 wheels
  (torch 2.12.1+cpu, transformers, diffusers, chromadb, etc. — no source builds needed, so this
  wasn't a blocker). If CPU performance ever becomes the bottleneck, installing a native ARM64
  Python build and recreating the venv would remove the emulation tax — not needed for MVP
  correctness, worth revisiting only if speed becomes a real complaint.

- **Model cache stays on C:\\, no thumb drive.** Original plan considered a 64 GB USB drive for
  model weights, but C:\\ had 17.75 GB free by the time we got here (comfortably above budget)
  and the drive already held an unrelated face-anonymiser copy — reformatting it wasn't worth
  the risk for no real benefit. Revisit only if C:\\ space gets tight again.

- **Railway-hosted Chroma has no auth.** Chroma dropped all built-in server authentication in
  its v1.0.0 Rust rewrite (the old `CHROMA_SERVER_AUTHN_*` env vars are legacy/non-functional),
  so `image-intelligence-production.up.railway.app` is reachable by anyone who has/guesses the
  URL. Confidentiality is still intact — embeddings are stored orthogonal-matrix-transformed
  (meaningless without the local `M`) and latents are AES-GCM encrypted (undecryptable without
  the local key) — but there's no protection against someone writing garbage into the collection
  or deleting it (integrity/availability, not confidentiality). Accepted for MVP. Revisit with a
  reverse-proxy (Caddy/nginx) bearer-token layer in front of Chroma before this is anything
  beyond a personal single-user test.

- **Text-answer step uses Claude (Sonnet 5), not DeepSeek.** The original plan called for the
  DeepSeek API, but its hosted API turned out to be text-only (no image input) as of this
  writing — confirmed against DeepSeek's own docs. Switched to Anthropic's Claude API (Sonnet 5),
  which the user already has an account for. Same e2ee tradeoff as originally planned: the
  decrypted, reconstructed image leaves the device to this one third party at query time.

- **Single model (Sonnet 5) for all answers, no Haiku/Sonnet tiering.** Considered routing
  "simple" questions to Haiku and "deep" ones to Sonnet, but at this MVP's query volume the cost
  difference is a fraction of a cent per query — not worth the complexity of classifying
  questions by difficulty. Sonnet 5 for everything also removes model capability as a variable
  while validating pipeline correctness. If a manual override is ever wanted (e.g. a `--deep`
  flag the user passes explicitly rather than automatic classification), that's a cheap addition
  later — not needed now.

- **Real-ESRGAN's general x4 model alters faces even without `face_enhance`.** Tried
  `realesr-general-x4v3` (`scripts/dev_enhance_vae_res.py`, pure-PyTorch re-implementation of
  `SRVGGNetCompact`, no `basicsr`/`realesrgan` packages — `basicsr` fails to even build on this
  machine with current setuptools, on top of the already-known `torchvision.transforms.functional_tensor`
  removal) on all 6 comparison images from the resolution test. Result: rejected. Even with
  `face_enhance`/GFPGAN never invoked, the general model still visibly reshapes eyebrows/lips and
  gives skin a waxy, over-smoothed look on this face-heavy content — a milder version of the
  exact hallucination risk `face_enhance` was already ruled out for. Milestone 9 (optional
  upscale polish) should not use this model as-is if faithfulness matters more than sharpness;
  worth trying a restoration-focused (non-generative) sharpening step instead if polish is ever
  revisited.

- **Stored latent is int8-quantized (lossy), not full float precision.** Cloud storage per
  image dropped from ~25 KB (fp32) to ~8 KB via int8 quantization + shuffle+zlib compression.
  Measured cost: ~0.07 dB of end-to-end reconstruction quality (negligible — the VAE's own
  256²→4×32×32 compression already dominates the error). Full investigation, numbers, and
  rollback instructions: `docs/storage-compression.md`.
