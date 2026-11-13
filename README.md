# tpclap-infer

Standalone, inference-only packaging of [Mohan Li's TP-CLAP](https://github.com/mohan-li/TP-CLAP). TP-CLAP makes plain audio, text and text-conditioned audio embeddings for retrieval. This package includes the needed author model code and omits training, datasets, evaluation harnesses, and model weights. It does not depend on the `tp-clap` evaluation package.

The model's usefulness on production loops is **still being evaluated**. Publishing this package establishes an installable, reproducible inference surface; it is not a claim that conditioned retrieval beats ordinary CLAP on any particular library.

## Install

```bash
pip install 'tpclap-infer[audio] @ git+https://github.com/openmirlab/tpclap-infer.git'
```

This is the GitHub install path; there is no PyPI release yet. `torchaudio` is optional because its compiled extension must match the installed Torch build; install a matching build for your CPU/CUDA environment. The base install supports import, checkpoint inspection and semantics discovery. Python 3.10+, NumPy 2.x and Transformers 4.57+ are supported; Transformers 5.x is temporarily excluded pending parity checks. See [compatibility](docs/compatibility.md).

## Use

Supply mono, 16 kHz `float32` NumPy audio. `nsynth` uses exactly 4 seconds (64,000 samples) and `mtt` exactly 10 seconds (160,000 samples). The caller handles resampling, long-file windows and aggregation.

```python
import numpy as np
from tpclap_infer import TPClapSession

audio = np.zeros(64_000, dtype=np.float32)
with TPClapSession(profile="nsynth", device="auto") as encoder:
    plain = encoder.embed_audio(audio, 16_000)  # [1, 1024]
    text = encoder.embed_text(["A piano plays."])
    query = encoder.embed_conditioned(
        audio, 16_000, ["What is the instrument playing the music?"]
    )
    text_score = encoder.similarity(plain, text)
    # For conditioned audio retrieval, compare the query to PLAIN audio candidates.
    candidate_score = encoder.similarity(query, plain)
```

All embeddings are L2-normalized. Similarities are uncalibrated cosine scores, not probabilities. Each checkpoint defines a separate embedding space; never mix `nsynth` and `mtt` vectors in one index. `session.semantics()` returns machine-readable output meanings. A session supports `load()`, `infer(...)`, `release()`, `close()`, `status`, `cache_info()` and context-manager use. `infer` dispatches to the three embedding calls and requires a loaded session.

## Weights and offline setup

On first `load()`, the package downloads only the chosen TP-CLAP checkpoint and pinned CED config/code and BERT config/tokenizer into the standard Hugging Face cache (`~/.cache/huggingface/hub`). SHA-256 is verified before deserializing checkpoint data. The author checkpoints are ~837 MiB each and are not in the wheel. Use `cache_dir="/path/to/hf-cache"` to change the HF cache root.

For manual/offline setup, download the exact files listed in [checkpoint metadata](src/tpclap_infer/config/checkpoints.toml), verify their SHA-256 values and arrange each asset in its own directory. Pass `asset_paths={"nsynth": "/path/to/tp-clap_nsynth.pt", "ced": "/path/to/ced-dir", "bert": "/path/to/bert-dir"}` with `local_files_only=True`. For `mtt`, substitute its checkpoint and key. `cache_info()` uses the same resolver as `load()` and checks all hashes. Author weights and auxiliary model files have their own licenses as recorded in the metadata; the package code is Apache-2.0.

## Verify

```bash
python -m pytest
python -m pytest -m model
python -m build
```

The model tests compare both profiles against output fixtures recorded with the author implementation before extraction. The tested CPU runtime is in each fixture JSON. See [source provenance](docs/source-provenance.md).
