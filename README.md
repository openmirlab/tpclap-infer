# tpclap-infer

Search audio with plain or text-conditioned TP-CLAP embeddings.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## Why this exists

[TP-CLAP](https://github.com/mohan-li/TP-CLAP) learns audio representations conditioned on a text question, alongside ordinary audio and text embeddings. The author's repository centers on evaluation scripts and a broad research dependency set. This standalone package extracts its inference model into a small verified lifecycle for use on an audio library; it does not depend on the `tp-clap` evaluation package. The model and research belong to the original authors.

The model's usefulness on production loops is **still being evaluated**. Public packaging establishes a reproducible inference surface; it is not evidence that conditioned retrieval beats ordinary CLAP on a particular library.

## Acknowledgments

- Original research: Mohan Li, Rama Doddipatla, and Philip C. Woodland ([TP-CLAP paper](https://arxiv.org/abs/2607.25085)).
- Original code: [mohan-li/TP-CLAP](https://github.com/mohan-li/TP-CLAP).
- Author checkpoint host: [Hugging Face, mohanli/TP-CLAP](https://huggingface.co/mohanli/TP-CLAP). CED model code and BERT config/tokenizer come from their separately pinned author repositories; OpenMIRLab did not train these weights.

## Citation

Please cite the [authors' TP-CLAP paper](https://arxiv.org/abs/2607.25085):

```bibtex
@article{li2026tpclap,
  title={Text-Prompted {CLAP}: Learning Text-Conditioned Audio Representations via Contrastive Learning},
  author={Li, Mohan and Doddipatla, Rama and Woodland, Philip C.},
  year={2026},
  eprint={2607.25085},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2607.25085}
}
```

## Features

- Plain audio, text, and prompt-conditioned audio embeddings in a 1024-dimensional space.
- Separate NSynth 4-second and MagnaTagATune 10-second checkpoint profiles.
- Strict checkpoint loading, SHA-256 verified assets, cache inspection, and CPU/CUDA device selection.

## Scope

The wheel includes the author model's inference core and a small embedding API. It omits training, datasets, evaluation harnesses, and weights. It does not build a search index or resample/window arbitrary-length recordings; callers own those steps.

## Install

```bash
pip install 'tpclap-infer[audio] @ git+https://github.com/openmirlab/tpclap-infer.git'
```

This is the GitHub install path; there is no PyPI release yet. `torchaudio` is optional because its compiled extension must match the installed Torch build; install a matching build for your CPU/CUDA environment. The base install supports import, checkpoint inspection and semantics discovery. Python 3.10+, NumPy 2.x and Transformers 4.57+ are supported; Transformers 5.x is temporarily excluded pending parity checks. See [compatibility](docs/compatibility.md).

## Quick start

Supply mono, 16 kHz `float32` NumPy audio. `nsynth` uses exactly 4 seconds (64,000 samples) and `mtt` exactly 10 seconds (160,000 samples). The caller handles resampling, long-file windows and aggregation.

```python
import numpy as np
from tpclap_infer import TPClapSession

t = np.arange(64_000, dtype=np.float32) / 16_000
audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
with TPClapSession(profile="nsynth", device="auto") as encoder:
    plain = encoder.embed_audio(audio, 16_000)  # [1, 1024]
    text = encoder.embed_text(["A steady electronic tone plays."])
    query = encoder.embed_conditioned(
        audio, 16_000, ["What is the instrument playing the music?"]
    )
    text_score = encoder.similarity(plain, text)
    # For conditioned audio retrieval, compare the query to PLAIN audio candidates.
    candidate_score = encoder.similarity(query, plain)
```

## Usage and output

All embeddings are L2-normalized. Similarities are uncalibrated cosine scores, not probabilities. Each checkpoint defines a separate embedding space; never mix `nsynth` and `mtt` vectors in one index. `session.semantics()` returns machine-readable output meanings. A session supports `load()`, `infer(...)`, `release()`, `close()`, `status`, `cache_info()` and context-manager use. `infer` dispatches to the three embedding calls and requires a loaded session.

## Weights and offline setup

On first `load()`, the package downloads only the chosen TP-CLAP checkpoint and pinned CED config/code and BERT config/tokenizer into the standard Hugging Face cache (`~/.cache/huggingface/hub`). SHA-256 is verified before deserializing checkpoint data. The author checkpoints are ~837 MiB each and are not in the wheel. Use `cache_dir="/path/to/hf-cache"` to change the HF cache root.

For manual/offline setup, prefill a Hugging Face cache while online. Download the checkpoint for the profile you need (`nsynth` or `mtt`) and both auxiliary file sets:

```bash
hf download mohanli/TP-CLAP tp-clap_nsynth.pt \
  --revision b14090f5210715958f5ab7056d3c47f5f61cb582 --cache-dir /your/cache
# For the mtt profile, download tp-clap_mtt.pt with the same revision instead.
hf download mispeech/ced-base config.json configuration_ced.py feature_extraction_ced.py modeling_ced.py preprocessor_config.json \
  --revision db3e14a8db4c21b56b165261c39649741a900e7f --cache-dir /your/cache
hf download google-bert/bert-base-uncased config.json tokenizer.json tokenizer_config.json vocab.txt \
  --revision 86b5e0934494bd15c9632b12f734a8a67f723594 --cache-dir /your/cache
```

Then use `TPClapSession(cache_dir="/your/cache", local_files_only=True)`. Alternatively, verify the SHA-256 values in [checkpoint metadata](src/tpclap_infer/config/checkpoints.toml), arrange each asset in its own directory, and pass `asset_paths={"nsynth": "/path/to/tp-clap_nsynth.pt", "ced": "/path/to/ced-dir", "bert": "/path/to/bert-dir"}` with `local_files_only=True`. For `mtt`, substitute its checkpoint and key. `cache_info()` uses the same resolver as `load()` and checks all hashes.

## What this project will never bundle

The wheel never includes checkpoint weights, training loops, datasets, or evaluation harnesses. It fetches only the verified auxiliary model code/config/tokenizer required for inference.

## Development

```bash
python -m pytest
python -m pytest -m model
python -m build
```

The model tests compare both profiles against output fixtures recorded with the author implementation before extraction. The tested CPU runtime is in each fixture JSON. See [source provenance](docs/source-provenance.md).

## License

Package code is [Apache-2.0](LICENSE). Author checkpoints and auxiliary model files have separate declarations recorded in the [checkpoint manifest](src/tpclap_infer/config/checkpoints.toml). See [NOTICE](NOTICE) for source provenance.

## Support

Report package bugs in [openmirlab/tpclap-infer issues](https://github.com/openmirlab/tpclap-infer/issues). Research and model questions belong with the [original TP-CLAP repository](https://github.com/mohan-li/TP-CLAP).
