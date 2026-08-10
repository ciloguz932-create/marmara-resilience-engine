"""Phase 3 — end-to-end reproducibility.

The contract: given the same seed and configuration, every stage is identical
bit for bit. Given a different seed, every stage changes. If this file passes,
any result the engine produces can be regenerated from its seed alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from mre.config import REPO_ROOT
from mre.data import build_synthetic_city
from mre.rng import child_generators, named_generator, seed_sequence
from mre.simulation import run_scenario

SEED = 20260810


# --- stream architecture --------------------------------------------------


def test_named_streams_are_stable_across_calls():
    assert seed_sequence(SEED, "hazard:0").spawn_key == seed_sequence(SEED, "hazard:0").spawn_key
    first = named_generator(SEED, "hazard:0").random(5)
    second = named_generator(SEED, "hazard:0").random(5)
    np.testing.assert_array_equal(first, second)


def test_different_names_give_independent_streams():
    a = named_generator(SEED, "hazard:0").random(5)
    b = named_generator(SEED, "damage:0").random(5)
    assert not np.array_equal(a, b)


def test_realisation_index_changes_the_stream():
    a = named_generator(SEED, "hazard:0").random(5)
    b = named_generator(SEED, "hazard:1").random(5)
    assert not np.array_equal(a, b)


def test_child_generators_are_order_independent():
    generators = child_generators(SEED, 5)
    values = [g.random() for g in generators]
    # Drawing realisation 3 alone must match drawing it within a batch.
    assert child_generators(SEED, 5)[3].random() == values[3]


def test_stream_names_do_not_depend_on_python_hash_randomisation():
    """CRC32, not hash(): hash() is salted per process and would silently
    destroy reproducibility across runs."""
    assert seed_sequence(SEED, "hazard:0").spawn_key == (
        __import__("zlib").crc32(b"hazard:0"),
    )


# --- no uncontrolled global randomness ------------------------------------


def test_engine_never_uses_global_numpy_random():
    """Only default_rng / SeedSequence / Generator are allowed.

    A stray np.random.uniform() would make results irreproducible in a way that
    no other test would catch.
    """
    allowed = {"default_rng", "SeedSequence", "Generator"}
    pattern = re.compile(r"np\.random\.(\w+)")

    offenders: list[str] = []
    for path in (REPO_ROOT / "mre").rglob("*.py"):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            if match.group(1) not in allowed:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {match.group(0)}")

    assert not offenders, f"uncontrolled global RNG: {offenders}"


def test_engine_never_seeds_the_global_rng():
    for path in (REPO_ROOT / "mre").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "np.random.seed" not in text
        assert "random.seed(" not in text


# --- full scenario reproducibility ---------------------------------------


@pytest.fixture(scope="module")
def repeated_runs(request):
    config = request.getfixturevalue("config")
    results = []
    for _ in range(2):
        city = build_synthetic_city(config, seed=SEED)
        results.append((city, run_scenario(city, config, seed=SEED, realisation=0)))
    return results


def test_same_seed_reproduces_the_whole_slice(repeated_runs):
    (_, first), (_, second) = repeated_runs

    np.testing.assert_array_equal(first.intensity, second.intensity)
    np.testing.assert_array_equal(first.damage.probabilities, second.damage.probabilities)
    np.testing.assert_array_equal(first.damage.states, second.damage.states)
    np.testing.assert_array_equal(
        first.damage.expected_damage_index, second.damage.expected_damage_index
    )
    np.testing.assert_array_equal(first.link_states, second.link_states)
    np.testing.assert_array_equal(
        first.link_closure_probabilities, second.link_closure_probabilities
    )
    assert first.summary() == second.summary()


def test_different_seed_changes_every_stage(config):
    city_a = build_synthetic_city(config, seed=SEED)
    city_b = build_synthetic_city(config, seed=SEED + 1)
    a = run_scenario(city_a, config, seed=SEED)
    b = run_scenario(city_b, config, seed=SEED + 1)

    assert not np.array_equal(a.intensity, b.intensity)
    assert not np.array_equal(a.damage.probabilities, b.damage.probabilities)
    assert not np.array_equal(a.damage.states, b.damage.states)
    assert not np.array_equal(a.link_states, b.link_states)


def test_realisations_differ_on_one_city(city, config):
    """Same city, different draw: the world is fixed, the event is not."""
    first = run_scenario(city, config, seed=SEED, realisation=0)
    second = run_scenario(city, config, seed=SEED, realisation=1)

    assert not np.array_equal(first.intensity, second.intensity)
    assert not np.array_equal(first.damage.states, second.damage.states)
    # The city itself is untouched.
    assert first.baseline_graph.number_of_edges() == second.baseline_graph.number_of_edges()


def test_realisation_is_identical_whether_run_alone_or_after_others(city, config):
    alone = run_scenario(city, config, seed=SEED, realisation=3)
    for index in range(3):
        run_scenario(city, config, seed=SEED, realisation=index)
    after = run_scenario(city, config, seed=SEED, realisation=3)
    np.testing.assert_array_equal(alone.damage.states, after.damage.states)
    np.testing.assert_array_equal(alone.link_states, after.link_states)


def test_scenario_result_carries_its_provenance(city, config):
    result = run_scenario(city, config, seed=SEED, realisation=2)
    assert result.seed == SEED
    assert result.realisation == 2
    assert result.scenario_id == config["hazard"]["scenario_id"]
