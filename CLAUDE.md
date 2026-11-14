# tpclap-infer maintainer guide

Standalone Apache-2.0 inference-only TP-CLAP package. The model and research are by Mohan Li, Rama Doddipatla, and Philip C. Woodland; the source is `mohan-li/TP-CLAP` and the author checkpoint host is `mohanli/TP-CLAP` on Hugging Face. Scope: author audio/text/fusion model, pinned CED frontend, BERT tokenizer, checkpoint resolver, lifecycle and native embedding semantics. No training, datasets, evaluation CLI, bundled weights or Tactus-specific code. Upstream source commit `39455a4fe4710e27eedf53a5598a6ae6404d0382`; copied TP-CLAP model files are under `src/tpclap_infer/_impl/`. Changes to those files must preserve source attribution and rerun author-output comparisons.

Profiles: `nsynth` 4 sec and `mtt` 10 sec, mono 16 kHz, exact length. Each has its own embedding space. Outputs are [N,1024] L2-normalized float arrays. Conditioned queries compare with plain audio candidates. Cosine scores are uncalibrated. The caller owns long-audio windows and any index. Do not claim production quality from fixture parity.

Checkpoint manifests live in `src/tpclap_infer/config/checkpoints.toml`; all files are SHA checked before loading. `asset_paths` supports manual offline setup. Do not add weights to Git or wheels. CED uses pinned HF model code and optional `torchaudio`; its base weights and BERT base weights are intentionally skipped because the full TP-CLAP checkpoint provides them. The temporary Transformers <5 ceiling and torchaudio matching risk are tracked in `docs/compatibility.md`.

Verification:

`.github/workflows/ci.yml` gates Python 3.10–3.15 offline tests and lint, plus
a Python 3.12 clean installed-wheel smoke. It excludes the `model` marker from
default CI because each profile needs an approximately 837 MiB author asset.
Run the marked tests separately to verify author-output parity before release.

```bash
python -m pytest
python -m pytest -m model
python -m ruff check src tests
python -m build
```

The `model` tests download/check ~837 MiB per profile unless assets are already cached. Exact float parity only applies to the fixture's recorded CPU runtime. Keep README and CHANGELOG current with public API and install changes. See `docs/source-provenance.md` for the original-to-package file map.

Public GitHub installation is `pip install 'tpclap-infer[audio] @ git+https://github.com/openmirlab/tpclap-infer.git'`; no PyPI release or Tactus integration exists yet. Load-bearing source files carry title/rationale/`Reads:` navigation headers; keep them accurate when editing. The README credits and cites the upstream work before describing package features.
