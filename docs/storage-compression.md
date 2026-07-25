# Cloud storage compression

## Why

Each captured frame stores two things in the Railway-hosted Chroma collection: the
orthogonal-matrix-transformed SigLIP embedding (searchable, 768-dim), and the encrypted VAE
latent (the reconstructable image content, opaque metadata). The latent dominates the per-image
size, so it's the only thing worth optimizing. All numbers below were measured directly against
a real captured frame and a real Railway-hosted record, not estimated.

## What changed, in order

1. **fp32 → fp16.** The VAE naturally produces a fp32 latent; there's no reason to store it at
   that precision. Casting to fp16 before encryption (and back to fp32 before decoding) is
   standard practice for diffusion VAE latents.
2. **+ shuffle-then-zlib.** Byte-shuffling regroups a float array's bytes by position (all
   byte-0s, then all byte-1s, ...) instead of by value. VAE latents are spatially smooth with
   similar-magnitude neighboring values, so the high-order bytes repeat far more once grouped
   this way — the same trick HDF5/Blosc use for scientific float arrays. Lossless: lets zlib
   actually find redundancy that's invisible to it in the original byte order.
3. **fp16 → int8 quantization (adopted).** Per-tensor symmetric quantization (scale so the max
   abs value maps to 127, round to int8; the scale travels alongside the data as metadata to
   dequantize). This is lossy, and was the one change that needed real measurement before
   trusting it — see below.

Implementation: `imgint/compression.py` (`shuffle_bytes`/`unshuffle_bytes`,
`compress_latent`/`decompress_latent`, `quantize_int8`/`dequantize_int8`), wired into
`imgint/pipeline.py`'s `ingest()`/`query()`. Compress-then-encrypt is the required order —
encrypted ciphertext is high-entropy by design and has nothing left to compress.

## Measured results (real capture, real Railway record)

Per-image cloud payload = embedding (3,072 B fixed) + encrypted-latent-base64 + nonce +
metadata + record id.

| Method | Full payload | vs fp32-VAE recon (marginal loss) | **vs real photo (total loss)** | Encode | Decode |
|---|---|---|---|---|---|
| fp32 (original baseline) | 25,054 B (100%) | lossless | 31.97 dB | — | — |
| fp16 | 14,130 B (56%) | 67.21 dB | 31.97 dB | 0.04 ms | 0.03 ms |
| fp16 + shuffle+zlib | 13,401 B (53%) | 67.21 dB (lossless vs fp16) | 31.97 dB | 0.35 ms | 0.14 ms |
| **int8 + shuffle+zlib (shipped)** | **8,110 B (32%)** | 45.83 dB | **31.90 dB** | 0.22 ms | 0.07 ms |

Two PSNR columns matter for different reasons:
- **vs fp32-VAE recon** isolates the marginal cost of *this specific step* on top of an already
  fp32-VAE-decoded image. This is where int8 looks bad in isolation (45.83 dB vs 67.21 dB).
- **vs real photo** is the total end-to-end quality a user actually sees. Here int8 costs
  **0.07 dB** — indistinguishable from noise. The VAE's own 256×256→4×32×32 compression already
  discards the vast majority of the information; quantizing what's left to 8 bits barely moves
  the total error, because the two error sources don't add linearly in dB terms and the VAE's
  error dominates by roughly 10x in variance.

All processing times are sub-millisecond and irrelevant next to the ~2-4 second VAE encode step
or the multi-second embed/upload round trip.

## Caveat

This is one real test image (a single indoor room capture). The reasoning behind "VAE error
dominates" is sound and should generalize (it doesn't depend on this image's specific content),
but it hasn't been verified across a range of scenes (bright/dark, high-detail/smooth,
faces/text). If reconstruction quality ever looks off on a real capture, this is the first thing
to check.

## How to roll back

If int8 quantization ever needs to be pulled:

1. In `imgint/pipeline.py`, `ingest()`: replace the `quantize_int8` + `compress_latent` calls
   with either plain `latent.astype(np.float16)` (matches the "fp16" row above) or
   `compress_latent(latent.astype(np.float16))` (the "fp16 + shuffle+zlib" row), and drop the
   `latent_scale` metadata field.
2. In `query()`: replace the `decompress_latent(..., np.int8, ...)` + `dequantize_int8` calls
   with the matching decode for whichever format you rolled back to.
3. **Existing stored records use whatever format was active when they were written** — there is
   no per-record format tag. Changing the pipeline's format makes previously-stored records
   undecodable (this project has hit that twice already switching fp32→fp16→int8; each time the
   fix was deleting the handful of test records in the Railway collection, since there was no
   real user data to lose). If real data exists when this matters, either keep a `latent_format`
   metadata field going forward, or re-ingest.
