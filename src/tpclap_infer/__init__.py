"""Standalone inference-only TP-CLAP embeddings."""

from .__about__ import __version__
from .session import TPClapSession

__all__ = ["TPClapSession", "__version__"]
