# todo.md — deliberate MVP shortcuts to revisit

- **Downgraded embedding model.** Using `google/siglip-base-patch16-224` (813 MB) instead of
  the originally-planned `google/siglip-so400m-patch14-384` (3.51 GB), to keep the CPU dev loop
  fast and stay within the disk budget on this ARM64/no-GPU machine. Swapping back to SO400M is
  a one-line model-id change in `imgint/embedder.py` — revisit once running on hardware where
  SO400M's slower CPU inference no longer matters.

- **No real-time throughput.** This MVP captures on-demand / at whatever rate CPU allows, not
  continuous 5 FPS. Revisit once on GPU (or NPU-accelerated) hardware.

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
