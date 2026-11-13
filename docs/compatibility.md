# Compatibility notes

The official TP-CLAP source currently declares `torch>=2.10` and includes dataset, evaluation and CLI dependencies. The extracted inference path was run against Torch 2.7.0, Transformers 4.57.6 and NumPy 2.5.3. This package declares the tested Torch 2.7 floor. The Transformers `<5` cap is temporary: CED's pinned Hugging Face code needs a separate parity run with Transformers 5.x before removal. `torchaudio` is optional because it must match the local Torch binary. The CED frontend requires it when `load()` runs.

The CED/BERT models are constructed from pinned config only. TP-CLAP's checkpoint then loads strictly with no missing or unexpected keys. This avoids fetching redundant backbone weights. Model and tokenizer source revision and hashes are pinned in the package config.
