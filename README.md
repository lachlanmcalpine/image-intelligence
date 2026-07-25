# image-intelligence — searchable visual recall, encrypted on the client

*Continuously capture what you see, make it searchable by natural language, and keep the
image content unreadable to the server that stores it. A VAE compresses frames to latents,
AES-GCM encrypts them, and an orthogonal-matrix transform lets semantic search run over
embeddings the store can't read.*

**Headline:** the vector store never sees a plaintext embedding or a plaintext image, and
`store.py` is structurally prevented from reaching key material — enforced by a test, not
by discipline.

---

## Why I built it

Two things at once.

I wanted **video to be searchable** — ask a question in plain language and get the moment
back — and I wanted it to cost almost nothing to store, because continuous capture is
worthless if the storage bill scales with how much you capture. Those two goals fight each
other, which is what made it interesting.

It's also my worked example of **RAG**: embed, retrieve, then hand the retrieved context to
a model to answer over. The twist is that the retrieval index here is visual, and the store
holding it isn't trusted.

---

## The problem

Anything that records what you see is a privacy disaster by default. The obvious
architecture — ship frames to a server, embed them, search them — means the server holds
your entire visual history in the clear.

But end-to-end encryption and semantic search pull in opposite directions: search needs to
compute distances between embeddings, and you can't compute distances on ciphertext. So
the design question is **how much can you encrypt before search stops working**, and how
honest can you be about the part you couldn't.

---

## What went wrong, and what I changed

Four things had to change from the original design. Each is recorded in the module
docstrings, because a reversal you don't write down is one you repeat.

- **DeepSeek → Claude.** The Q&A layer was built for DeepSeek, then DeepSeek's hosted API
  turned out to be **text-only**. The whole answer path had to move to Claude vision.
- **Capture resolution.** The driver default was 640×480. Comparing reconstructions
  directly showed 1920×1080 was visibly sharper at every VAE encode size tested, so it was
  worth the wider 16:9 crop. Also learned the hard way that MSMF can't reliably switch
  resolution on an already-streaming capture — it raises a Mat assertion mid-stream — so
  resolution is requested at open time, before the first read.
- **Compression only works in one order.** Byte-shuffle + zlib on the latents does nothing
  after encryption, because ciphertext is high-entropy by design. It has to be
  compress-*then*-encrypt.
- **Text can't go through the VAE.** Small text is exactly what an autoencoder discards —
  so OCR was pulled out into a separate channel running on the raw full-resolution frame.

---

## Architecture, and the calls I made

**Write path:** capture → embed → dedup gate → VAE encode → encrypt → store
**Read path:** text query → embed → transform → search → decrypt → VAE decode → ask Claude

- **A pure autoencoder, never diffusion.** Reconstruction uses the SDXL VAE family with no
  diffusion transformer, so the system can't hallucinate content that wasn't in the frame.
  For a recall tool that's not an optimisation, it's a correctness requirement.
- **SDXL's VAE, not FLUX's.** FLUX's VAE has 16 latent channels against SDXL's 4. Since the
  diffusion model is never used, those channels buy nothing and would quietly **quadruple
  per-frame storage**.
- **Orthogonal-matrix transform on embeddings.** An orthogonal transform preserves cosine
  similarity, so search still works while the raw vector is obscured. **I've documented that
  this is obfuscation, not cryptographically hard encryption** — enough known
  (embedding, transformed) pairs would let an attacker recover `M` via Procrustes. Keys are
  per-user, so a break is limited to one user. That limitation is written into the module
  docstring rather than left for someone else to find.
- **AES-GCM on the latents**, stored as opaque metadata. The image content is genuinely
  encrypted; only the search vectors are merely transformed.
- **`store.py` deliberately never imports `imgint.crypto`.** Key material must not be
  *reachable* from the code path that talks to the cloud, and `tests/test_store.py` enforces
  it. Making that a structural property with a test behind it, instead of a rule I'd
  remember, is the design decision I'd defend hardest.
- **Two dedup gates, cheapest first.** dHash catches literally-identical pixels with a
  grayscale resize and a bit compare; only frames that survive pay ~0.3s for a SigLIP
  embedding, which then catches "same scene".
- **Two orthogonal compression levers.** Product quantization shrinks the *embedding*
  (768-dim fp32 → 8 bytes, ~384×); int8 quantization shrinks the *image latent*. They
  compose because they touch different things.
- **One deliberate e2ee exception, stated out loud.** At query time the decrypted,
  reconstructed image leaves the device for Anthropic's API. That's a real hole in the
  privacy claim, accepted as an MVP tradeoff and documented as such rather than glossed.

Claude wrote most of the implementation. The architecture is mine — compress-then-encrypt,
keeping OCR out of the lossy VAE path, and making `store.py` structurally unable to reach
key material rather than merely trusting it not to.

---

## Results

Product quantization takes a 768-dim fp32 embedding from 3,072 bytes to 8 — roughly
**384× smaller** — at the cost of a lossy, approximate reconstruction used only for search
ranking. The VAE latent carries the image content on a separate compression lever
(shuffle + zlib, then int8 quantization), so the two compose without overlapping.

**Measured** — one real indoor capture, one real Railway record: the full per-image cloud
payload is **8,110 B** with the shipped int8 + shuffle+zlib latent, down from **25,054 B** at
fp32 (32% of baseline), at an end-to-end quality cost of **0.07 dB** against the original photo.
The dedup gate separates cleanly on real frames too — same scene scores 0.93–0.99 cosine
similarity against different scenes at 0.44–0.84, so the 0.95 skip threshold clears the
different-scene ceiling with margin. Full breakdown:
[`docs/storage-compression.md`](docs/storage-compression.md).

---

## What's next

The orthogonal-matrix transform is the weak link in the privacy story and wants replacing
with something that has a hardness argument behind it. The query-time e2ee exception —
where the reconstructed image leaves the device for Anthropic's API — wants closing with a
local VLM.

---

## Run it

```bash
pip install -r requirements.txt          # or requirements-cpu.txt
python scripts/run_pipeline.py
python scripts/dev_ask.py                # query it
```

Deployable via `Dockerfile` / `railway.json`. Keys live local-only under `./keys/`
(gitignored).
