"""Fast public-surface and asset-boundary tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tpclap_infer import TPClapSession, __version__
from tpclap_infer.checkpoints import Asset, default_checkpoints, resolve_asset


def test_version_profiles_and_semantics():
    assert __version__ == "0.1.0"
    assert TPClapSession(profile="nsynth").semantics()["input_samples"] == 64000
    assert TPClapSession(profile="mtt").semantics()["input_samples"] == 160000
    assert TPClapSession().semantics()["similarity"].startswith("cosine")
    with pytest.raises(ValueError, match="profile"):
        TPClapSession(profile="other")


def test_lifecycle_guards_and_device():
    session = TPClapSession(device="mps", local_files_only=True)
    assert session.status == "unloaded"
    with pytest.raises(RuntimeError, match="load"):
        session.infer(texts=["piano"])
    with pytest.raises(ValueError, match="mps"):
        session.load()
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()


def test_similarity_is_cosine_and_validates():
    a = np.zeros((1, 1024), np.float32)
    b = np.zeros((2, 1024), np.float32)
    a[0, 0] = 2
    b[0, 0] = 1
    b[1, 1] = 3
    np.testing.assert_array_equal(TPClapSession.similarity(a, b), [[1, 0]])
    with pytest.raises(ValueError, match="nonzero"):
        TPClapSession.similarity(np.zeros_like(a), b)
    with pytest.raises(ValueError, match="1024"):
        TPClapSession.similarity(a[:, :10], b)


def test_asset_resolver_and_cache_share_verified_path(tmp_path: Path):
    payload = b"checkpoint bytes"
    (tmp_path / "dummy.pt").write_bytes(payload)
    asset = Asset("example/model", "revision", {"dummy.pt": hashlib.sha256(payload).hexdigest()},
                  "test", "Apache-2.0")
    assert resolve_asset(asset, path=tmp_path, local_files_only=True) == tmp_path
    (tmp_path / "dummy.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        resolve_asset(asset, path=tmp_path, local_files_only=True)


def test_manifest_has_all_profiles():
    manifest = default_checkpoints()
    assert set(manifest.nsynth.hashes) == {"tp-clap_nsynth.pt"}
    assert set(manifest.mtt.hashes) == {"tp-clap_mtt.pt"}
    assert "model.safetensors" not in manifest.ced.hashes
    assert "model.safetensors" not in manifest.bert.hashes
