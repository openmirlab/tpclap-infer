"""Independent TP-CLAP model lifecycle and exact-profile encoders.

Reads: pinned checkpoint assets, author model core, CED feature extractor,
BERT tokenizer. Callers own slicing, resampling and any aggregate over clips.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np

from .checkpoints import Checkpoints, default_checkpoints, resolve_asset
from .results import result_semantics


def _resolve_device(device: str) -> str:
    import torch

    if device == "mps":
        raise ValueError("TP-CLAP does not support device='mps'")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return device
    if device == "cuda" or re.fullmatch(r"cuda:[0-9]+", device):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested CUDA device is unavailable: {device}")
        index = torch.device(device).index
        if index is not None and index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device is unavailable: {device}")
        return device
    raise ValueError(f"Unsupported device: {device}")


class TPClapSession:
    """Resident TP-CLAP encoder; one instance is not safe for concurrent calls."""

    def __init__(
        self, *, profile: str = "nsynth", device: str = "auto",
        cache_dir: str | Path | None = None, asset_paths: dict[str, str | Path] | None = None,
        checkpoints: Checkpoints | None = None, local_files_only: bool = False,
    ) -> None:
        if profile not in ("nsynth", "mtt"):
            raise ValueError("profile must be 'nsynth' or 'mtt'")
        self.profile = profile
        self.device = device
        self.cache_dir = cache_dir
        self.asset_paths = dict(asset_paths or {})
        if set(self.asset_paths) - {"nsynth", "mtt", "ced", "bert"}:
            raise ValueError("Unknown asset_paths key")
        self.checkpoints = checkpoints or default_checkpoints()
        self.local_files_only = local_files_only
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._active_device = None
        self._status = "unloaded"

    @property
    def status(self) -> str:
        return self._status

    def _resolve(self, key: str, *, local_files_only: bool) -> Path:
        return resolve_asset(
            getattr(self.checkpoints, key), cache_dir=self.cache_dir,
            local_files_only=local_files_only, path=self.asset_paths.get(key),
        )

    def cache_info(self) -> dict:
        result = {}
        for key in (self.profile, "ced", "bert"):
            try:
                path = self._resolve(key, local_files_only=True)
            except (OSError, ValueError, FileNotFoundError):
                path = None
            result[key] = {
                "cached": path is not None, "path": str(path) if path else None,
                "revision": getattr(self.checkpoints, key).revision,
            }
        return result

    def load(self) -> TPClapSession:
        if self._status == "closed":
            raise RuntimeError("TP-CLAP session is closed")
        if self._model is not None:
            return self
        device = _resolve_device(self.device)
        weight_dir = self._resolve(self.profile, local_files_only=self.local_files_only)
        ced_dir = self._resolve("ced", local_files_only=self.local_files_only)
        bert_dir = self._resolve("bert", local_files_only=self.local_files_only)

        from transformers import AutoFeatureExtractor, AutoTokenizer

        from ._impl.configuration_tp_clap import TPCLAPConfig
        from ._impl.modeling_tp_clap import TPCLAPModel

        try:
            processor = AutoFeatureExtractor.from_pretrained(
                str(ced_dir), local_files_only=True, trust_remote_code=True,
            )
        except ImportError as exc:
            raise ImportError("TP-CLAP audio frontend needs torchaudio; install tpclap-infer[audio]") from exc
        tokenizer = AutoTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
        model = TPCLAPModel(TPCLAPConfig(
            audio_encoder=str(ced_dir), text_encoder=str(bert_dir),
        ))
        filename = next(iter(getattr(self.checkpoints, self.profile).hashes))
        missing, unexpected = model.load_checkpoint(str(weight_dir / filename), strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self._model = model.to(device).eval()
        self._processor = processor
        self._tokenizer = tokenizer
        self._active_device = device
        self._status = "ready"
        return self

    def _audio(self, audio: np.ndarray, sample_rate: int):
        self._require_ready()
        if sample_rate != 16000:
            raise ValueError("TP-CLAP requires 16000 Hz; resample before calling")
        array = np.asarray(audio)
        if array.ndim == 1:
            array = array[None, :]
        length = 64000 if self.profile == "nsynth" else 160000
        if array.ndim != 2 or array.shape[1] != length or array.shape[0] == 0:
            raise ValueError(f"Expected mono [{length}] or batch [B, {length}] for {self.profile}")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("Audio samples must be numeric")
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Audio contains NaN or infinity")
        waveforms = [np.ascontiguousarray(row) for row in array]
        return self._processor(waveforms, sampling_rate=sample_rate, return_tensors="pt").input_values.to(
            self._active_device
        )

    def _text(self, texts: list[str] | tuple[str, ...]):
        self._require_ready()
        if not isinstance(texts, (list, tuple)) or not texts or any(
            not isinstance(value, str) or not value.strip() for value in texts
        ):
            raise ValueError("texts must be a nonempty list of nonempty strings")
        tokens = self._tokenizer(texts, padding=True, return_tensors="pt")
        return tokens.input_ids.to(self._active_device), tokens.attention_mask.to(self._active_device)

    def embed_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Plain L2-normalized [batch,1024] audio embeddings."""
        import torch

        with torch.inference_mode():
            result = self._model.get_audio_embeds(audios=self._audio(audio, sample_rate))
        return result.float().cpu().numpy().copy()

    def embed_text(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        """L2-normalized [batch,1024] text embeddings."""
        import torch

        ids, mask = self._text(texts)
        with torch.inference_mode():
            result = self._model.get_text_embeds(ids, mask)
        return result.float().cpu().numpy().copy()

    def embed_conditioned(
        self, audio: np.ndarray, sample_rate: int, prompts: list[str] | tuple[str, ...],
    ) -> np.ndarray:
        """Prompt-conditioned queries; compare against plain candidate audio embeddings."""
        import torch

        values = self._audio(audio, sample_rate)
        ids, mask = self._text(prompts)
        if ids.shape[0] != values.shape[0]:
            raise ValueError("Provide one prompt per audio item")
        with torch.inference_mode():
            result = self._model.get_conditioned_audio_embeds(
                audios=values, prompt_ids=ids, prompt_attention_mask=mask,
            )
        return result.float().cpu().numpy().copy()

    @staticmethod
    def similarity(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix [queries,candidates], not probabilities."""
        q = np.asarray(queries, dtype=np.float32)
        c = np.asarray(candidates, dtype=np.float32)
        if (q.ndim != 2 or c.ndim != 2 or q.shape[1:] != (1024,)
                or c.shape[1:] != (1024,) or not len(q) or not len(c)):
            raise ValueError("Expected nonempty [N,1024] and [M,1024] embeddings")
        if not np.isfinite(q).all() or not np.isfinite(c).all():
            raise ValueError("Embeddings contain NaN or infinity")
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        c_norm = np.linalg.norm(c, axis=1, keepdims=True)
        if np.any(q_norm == 0) or np.any(c_norm == 0):
            raise ValueError("Embeddings must be nonzero")
        return (q / q_norm) @ (c / c_norm).T

    def infer(
        self, *, audio: np.ndarray | None = None, sample_rate: int | None = None,
        texts: list[str] | tuple[str, ...] | None = None,
        prompts: list[str] | tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """Ready-only embedding dispatcher."""
        self._require_ready()
        if texts is not None:
            if audio is not None or prompts is not None:
                raise ValueError("Provide text or audio, not both")
            return self.embed_text(texts)
        if audio is None or sample_rate is None:
            raise ValueError("audio and sample_rate are required")
        if prompts is not None:
            return self.embed_conditioned(audio, sample_rate, prompts)
        return self.embed_audio(audio, sample_rate)

    def semantics(self) -> dict:
        return result_semantics(self.profile)

    def _require_ready(self) -> None:
        if self._status != "ready":
            raise RuntimeError("Call load() before inference")

    def release(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._active_device = None
        gc.collect()
        if self._status != "closed":
            self._status = "unloaded"

    def close(self) -> None:
        self.release()
        self._status = "closed"

    def __enter__(self) -> TPClapSession:  # noqa: PYI034 - Python 3.10 lacks typing.Self
        return self.load()

    def __exit__(self, *_exc: object) -> None:
        self.close()
