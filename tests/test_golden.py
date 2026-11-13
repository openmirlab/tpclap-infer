"""Author CPU outputs recorded before model extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tpclap_infer import TPClapSession

HERE = Path(__file__).parent / "fixtures"


@pytest.mark.model
@pytest.mark.parametrize("profile", ["nsynth", "mtt"])
def test_author_output_parity(profile: str):
    import torch
    import transformers

    info = json.loads((HERE / f"upstream_{profile}_cpu.json").read_text())
    old = np.load(HERE / f"upstream_{profile}_cpu.npz")
    audio = np.stack([sf.read(HERE / path, dtype="float32")[0] for path in info["audio"]])
    assert np.any(audio == 0) and np.any(audio != 0)
    torch.set_num_threads(info["threads"])
    local = os.getenv(f"TPCLAP_{profile.upper()}_CHECKPOINT")
    overrides = {profile: local} if local else None
    with TPClapSession(profile=profile, device="cpu", asset_paths=overrides) as encoder:
        assert encoder.cache_info()[profile]["cached"]
        outputs = {
            "plain": encoder.embed_audio(audio, 16000),
            "conditioned": encoder.embed_conditioned(audio, 16000, info["prompts"]),
            "text": encoder.embed_text(info["texts"]),
        }
        assert encoder.status == "ready"
        np.testing.assert_allclose(
            encoder.similarity(outputs["conditioned"], outputs["plain"]), old["score"],
            rtol=2e-5, atol=2e-6,
        )
        with pytest.raises(ValueError, match="16000"):
            encoder.embed_audio(audio, 44100)
        with pytest.raises(ValueError, match="prompt"):
            encoder.embed_conditioned(audio, 16000, ["one", "two", "three"])
        with pytest.raises(ValueError, match="Expected mono"):
            encoder.embed_audio(audio[..., :-1], 16000)
        exact_runtime = (
            torch.__version__ == info["torch"]
            and transformers.__version__ == info["transformers"]
            and np.__version__ == info["numpy"]
        )
        for key, actual in outputs.items():
            assert actual.shape == tuple(info["shapes"][key])
            assert np.max(np.abs(actual)) > 0.01
            if exact_runtime:
                np.testing.assert_array_equal(actual, old[key])
            else:
                np.testing.assert_allclose(actual, old[key], rtol=2e-4, atol=2e-5)
    assert encoder.status == "closed"
