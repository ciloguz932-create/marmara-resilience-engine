"""Seeded random-stream architecture.

Every stochastic step in the engine draws from a generator obtained here. There
is no module-level global RNG and no bare ``np.random.*`` call anywhere in
``mre`` -- ``tests/test_reproducibility.py`` enforces that.

Two primitives:

``named_generator(seed, name)``
    A stable, independent stream identified by a string. Stage *A* consuming a
    different number of draws never shifts stage *B*, because each stage owns
    its own stream. Adding a new stage later cannot perturb existing results.

``child_generators(seed, n)``
    ``n`` independent streams for ``n`` Monte Carlo realisations. Realisation
    *i* always uses stream *i*, so results are order-independent and survive
    future parallelisation unchanged.

Stream names are mapped to ``SeedSequence`` spawn keys with CRC32, which is
stable across processes and Python versions -- unlike ``hash()``, which is
randomised per process and would silently destroy reproducibility.

(This module is not in the original Phase 1 tree. It exists so that
``mre.data`` and ``mre.simulation`` can both seed themselves without importing
each other.)
"""

from __future__ import annotations

import zlib

import numpy as np

__all__ = ["named_generator", "child_generators", "seed_sequence"]


def seed_sequence(seed: int, name: str) -> np.random.SeedSequence:
    """A ``SeedSequence`` for the named stream derived from ``seed``."""
    key = zlib.crc32(name.encode("utf-8"))
    return np.random.SeedSequence(entropy=seed, spawn_key=(key,))


def named_generator(seed: int, name: str) -> np.random.Generator:
    """An independent generator for the named stage.

    Names are hierarchical by convention: ``"city.buildings"``,
    ``"hazard:0"``, ``"damage:12"``.
    """
    return np.random.default_rng(seed_sequence(seed, name))


def child_generators(seed: int, n: int) -> list[np.random.Generator]:
    """``n`` independent generators for ``n`` realisations."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]
