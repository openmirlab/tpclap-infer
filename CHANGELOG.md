# Changelog

## Unreleased

- Gate pushes and pull requests with Python 3.10–3.15 offline tests, lint, and a clean wheel-from-sdist import check; keep both real-profile parity tests separate.
- Reorganized the README around upstream credit, citation, install, usage, manual asset downloads, development, license, and support.

## 0.1.0 — 2026-11-13

- Extracted the author inference model, fusion layers and configuration.
- Added NSynth and MagnaTagATune checkpoint profiles, verified assets, independent
  lifecycle, plain/conditioned audio and text embeddings, cosine similarity.
- Pinned CED and BERT config/tokenizer without downloading their redundant weights.
- Recorded author-output fixtures before extraction and compared both profiles.
