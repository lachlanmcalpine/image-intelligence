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
