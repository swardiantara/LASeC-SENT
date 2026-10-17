"""Sentence segmentation for DJI app messages.

A cell of the decrypted flight log (APP.tip / APP.warning) may hold several app
messages that the decryption tool joined with "; ". We therefore split on the
semicolon first (message boundary) and only then run PySBD (sentence boundary).
"""
from functools import lru_cache
from typing import List

import pysbd


@lru_cache(maxsize=None)
def _segmenter(language: str = "en") -> pysbd.Segmenter:
    return pysbd.Segmenter(language=language, clean=False)


def split_messages(cell: str, sep: str = ";") -> List[str]:
    """Split a log cell into atomic app messages."""
    return [m.strip() for m in str(cell).split(sep) if m.strip()]


def split_sentences(message: str, language: str = "en") -> List[str]:
    """Split one atomic app message into sentences with PySBD."""
    return [s.strip() for s in _segmenter(language).segment(message) if s.strip()]


def segment(cell: str, sep: str = ";", language: str = "en") -> List[List[str]]:
    """Return the sentences of each atomic message in a cell: [[s11, s12], [s21], ...]."""
    return [split_sentences(m, language) for m in split_messages(cell, sep)]
