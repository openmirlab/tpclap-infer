"""Single resolver for checkpoint loading and cache inspection.

Reads: package TOML, Hugging Face snapshots, SHA-256. The author pickle is
opened only after verification. Generic local path overrides allow offline use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class Asset:
    repo_id: str
    revision: str
    hashes: dict[str, str]
    source: str
    license: str


@dataclass(frozen=True)
class Checkpoints:
    nsynth: Asset
    mtt: Asset
    ced: Asset
    bert: Asset


def default_checkpoints() -> Checkpoints:
    raw = tomllib.loads(files("tpclap_infer").joinpath("config/checkpoints.toml").read_text())

    def make(key: str) -> Asset:
        item = raw[key]
        hashes = dict(item["files"]) if "files" in item else {item["filename"]: item["sha256"]}
        if not hashes or any(len(digest) != 64 for digest in hashes.values()):
            raise ValueError(f"Malformed checkpoint hashes for {key}")
        return Asset(item["repo_id"], item["revision"], hashes, item["source"], item["license"])

    return Checkpoints(make("nsynth"), make("mtt"), make("ced"), make("bert"))


def resolve_asset(
    asset: Asset, *, cache_dir: str | Path | None = None,
    local_files_only: bool = False, path: str | Path | None = None,
) -> Path:
    if path is None:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(
            repo_id=asset.repo_id, revision=asset.revision,
            allow_patterns=list(asset.hashes),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        ))
    else:
        supplied = Path(path).expanduser()
        root = supplied.parent if supplied.is_file() else supplied
    for name, expected in asset.hashes.items():
        candidate = root / name
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing {candidate}")
        hasher = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {candidate}")
    return root
