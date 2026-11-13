"""Machine-readable native embedding and score meanings."""


def result_semantics(profile: str = "nsynth") -> dict:
    if profile not in ("nsynth", "mtt"):
        raise ValueError("Unknown TP-CLAP profile")
    return {
        "model": "TP-CLAP",
        "profile": profile,
        "sample_rate_hz": 16000,
        "input_samples": 64000 if profile == "nsynth" else 160000,
        "input_channels": 1,
        "window_seconds": 4 if profile == "nsynth" else 10,
        "scope": "supplied_window",
        "embedding_dimension": 1024,
        "embedding_normalization": "L2",
        "audio_pooling": "mean over CED time tokens before projection",
        "text_pooling": "attention-mask mean over BERT tokens before projection",
        "conditioned_query": "fusion of CED audio tokens and BERT prompt tokens",
        "candidate_space": "plain audio embedding for conditioned retrieval",
        "similarity": "cosine [-1,1], uncalibrated, not a probability",
        "temporal_output": "none",
        "aggregation": "none; caller owns slicing and aggregation",
    }
