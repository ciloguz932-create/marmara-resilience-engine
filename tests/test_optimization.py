"""Phase 5 — prototype intervention optimization.

Covers the domain model (construction, cost, budget), the per-entity effects,
the common-random-numbers / paired evaluation, reproducibility, ranking, the
uncertainty summary, and the edge cases. Fast tests run on the reduced
``small_city``; the one test that needs the heavy tail of unreachability events
runs on the full city and is marked ``slow``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mre.models import InterventionType
from mre.optimization import (
    PRIMARY_METRIC,
    Intervention,
    Portfolio,
    building_retrofit,
    compare_portfolios,
    enumerate_feasible_portfolios,
    evaluate_portfolio,
    hospital_support,
    road_hardening,
    split_realisations,
)
from mre.simulation import run_monte_carlo, run_scenario

SEED = 20260810
N = 32


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def baseline(small_city, small_config):
    """A no-intervention Monte Carlo run used as the paired baseline."""
    return run_monte_carlo(small_city, small_config, n_simulations=N, seed=SEED)


@pytest.fixture(scope="module")
def comparison(small_city, small_config):
    return compare_portfolios(small_city, small_config, seed=SEED, n_simulations=N)


@pytest.fixture
def retrofit(small_city, small_config):
    return building_retrofit(small_city, small_config)


@pytest.fixture
def harden(small_city, small_config, baseline):
    return road_hardening(small_city, small_config, baseline.link_closed_frequency)


@pytest.fixture
def support(small_city, small_config):
    return hospital_support(small_city, small_config)


# --------------------------------------------------------------------------- #
# construction, cost, budget
# --------------------------------------------------------------------------- #


def test_interventions_construct_with_expected_types(retrofit, harden, support):
    assert retrofit.intervention_type is InterventionType.BUILDING_RETROFIT
    assert harden.intervention_type is InterventionType.ROAD_HARDENING
    assert support.intervention_type is InterventionType.HOSPITAL_SUPPORT
    for intervention in (retrofit, harden, support):
        assert intervention.target_ids
        assert intervention.cost > 0
        assert "PROTOTYPE" in intervention.assumptions


def test_cost_scales_with_target_count(retrofit, small_config):
    per_building = small_config["interventions"]["building_retrofit"]["cost_per_building"]
    assert retrofit.cost == pytest.approx(per_building * len(retrofit.target_ids))


def test_negative_cost_is_rejected():
    with pytest.raises(ValueError, match="cost must be >= 0"):
        Intervention(
            intervention_id="X",
            intervention_type=InterventionType.BUILDING_RETROFIT,
            target_ids=("B00000",),
            cost=-1.0,
            effect={"vulnerability_divisor": 1.35, "vulnerability_floor": 0.35},
        )


def test_portfolio_cost_is_the_sum(retrofit, harden, support):
    portfolio = Portfolio("P", (retrofit, harden, support))
    assert portfolio.total_cost == pytest.approx(
        retrofit.cost + harden.cost + support.cost
    )


def test_budget_feasibility_and_rejection(retrofit, harden, support):
    portfolio = Portfolio("ALL", (retrofit, harden, support))
    assert portfolio.within_budget(portfolio.total_cost + 1)
    assert not portfolio.within_budget(portfolio.total_cost - 1)


def test_enumeration_excludes_over_budget_portfolios(retrofit, harden, support):
    candidates = (retrofit, harden, support)
    total = sum(i.cost for i in candidates)
    # A budget below the triple's cost must drop the all-three portfolio.
    portfolios = enumerate_feasible_portfolios(candidates, total - 1)
    assert all(len(p.interventions) < 3 for p in portfolios)
    # All returned portfolios are within budget and non-empty.
    for p in portfolios:
        assert p.interventions
        assert p.within_budget(total - 1)


def test_enumeration_is_the_power_set_within_budget(retrofit, harden, support):
    candidates = (retrofit, harden, support)
    big = sum(i.cost for i in candidates) + 1
    portfolios = enumerate_feasible_portfolios(candidates, big)
    # 2^3 - 1 = 7 non-empty subsets, all feasible under a generous budget.
    assert len(portfolios) == 7


# --------------------------------------------------------------------------- #
# per-entity effects and baseline preservation
# --------------------------------------------------------------------------- #


def test_building_retrofit_lowers_targeted_vulnerability(retrofit, small_city, small_config):
    modified = retrofit.apply_to_city(small_city)
    divisor = small_config["interventions"]["building_retrofit"]["fragility_median_uplift"]
    floor = small_config["synthetic_city"]["buildings"]["vulnerability"]["clip"][0]
    targets = set(retrofit.target_ids)
    before = {b.building_id: b.vulnerability_index for b in small_city.buildings}
    for b in modified.buildings:
        if b.building_id in targets:
            assert b.vulnerability_index == pytest.approx(
                max(floor, before[b.building_id] / divisor)
            )
            assert b.vulnerability_index <= before[b.building_id] + 1e-12
        else:
            assert b.vulnerability_index == before[b.building_id]


def test_road_hardening_lowers_targeted_closure_and_susceptibility(harden, small_city, small_config):
    modified = harden.apply_to_city(small_city)
    mult = small_config["interventions"]["road_hardening"]["closure_probability_multiplier"]
    targets = set(harden.target_ids)
    before = {r.road_id: r for r in small_city.roads}
    for r in modified.roads:
        if r.road_id in targets:
            assert r.closure_probability == pytest.approx(before[r.road_id].closure_probability * mult)
            assert r.susceptibility == pytest.approx(before[r.road_id].susceptibility * mult)
        else:
            assert r.closure_probability == before[r.road_id].closure_probability


def test_hospital_support_raises_targeted_capacity(support, small_city, small_config):
    modified = support.apply_to_city(small_city)
    mult = small_config["interventions"]["hospital_support"]["emergency_capacity_multiplier"]
    targets = set(support.target_ids)
    before = {h.hospital_id: h.emergency_capacity for h in small_city.hospitals}
    for h in modified.hospitals:
        if h.hospital_id in targets:
            assert h.emergency_capacity == max(1, round(before[h.hospital_id] * mult))
            assert h.emergency_capacity >= before[h.hospital_id]
        else:
            assert h.emergency_capacity == before[h.hospital_id]


def test_apply_never_mutates_the_baseline_city(retrofit, harden, support, small_city):
    """The baseline city must survive intervention application untouched, or
    before/after comparison is impossible."""
    v_before = tuple(b.vulnerability_index for b in small_city.buildings)
    s_before = tuple(r.susceptibility for r in small_city.roads)
    cap_before = tuple(h.emergency_capacity for h in small_city.hospitals)

    Portfolio("P", (retrofit, harden, support)).apply_to_city(small_city)

    assert tuple(b.vulnerability_index for b in small_city.buildings) == v_before
    assert tuple(r.susceptibility for r in small_city.roads) == s_before
    assert tuple(h.emergency_capacity for h in small_city.hospitals) == cap_before


def test_empty_portfolio_is_a_no_op(small_city):
    empty = Portfolio("EMPTY", ())
    assert empty.total_cost == 0.0
    assert empty.apply_to_city(small_city) is small_city


def test_invalid_target_id_is_rejected(small_city):
    bad = Intervention(
        intervention_id="BAD",
        intervention_type=InterventionType.BUILDING_RETROFIT,
        target_ids=("NOT_A_REAL_BUILDING",),
        cost=1.0,
        effect={"vulnerability_divisor": 1.35, "vulnerability_floor": 0.35},
    )
    with pytest.raises(ValueError, match="absent from the city"):
        bad.validate_targets(small_city)


# --------------------------------------------------------------------------- #
# common random numbers / paired evaluation
# --------------------------------------------------------------------------- #


def test_roads_only_intervention_leaves_hazard_and_damage_draws_identical(
    harden, small_city, small_config
):
    """CRN: hardening touches only links, so the intensity field and the damage
    draws must be bit-identical to the baseline realisation."""
    modified = Portfolio("H", (harden,)).apply_to_city(small_city)
    base = run_scenario(small_city, small_config, seed=SEED, realisation=0)
    post = run_scenario(modified, small_config, seed=SEED, realisation=0)
    np.testing.assert_array_equal(base.intensity, post.intensity)
    np.testing.assert_array_equal(base.damage.states, post.damage.states)


def test_retrofit_leaves_hazard_identical_and_cannot_worsen_damage(
    retrofit, small_city, small_config
):
    """CRN: retrofit shares the same intensity field and the same damage uniform
    draws, so no targeted building can end up MORE damaged than at baseline."""
    modified = Portfolio("R", (retrofit,)).apply_to_city(small_city)
    base = run_scenario(small_city, small_config, seed=SEED, realisation=0)
    post = run_scenario(modified, small_config, seed=SEED, realisation=0)
    np.testing.assert_array_equal(base.intensity, post.intensity)
    assert np.all(post.damage.states <= base.damage.states)


def test_structural_interventions_never_worsen_unreachability(
    retrofit, harden, small_city, small_config, baseline
):
    """Under CRN, fewer collapses / fewer closures can only help: post-event
    unreachable population is <= baseline in every paired realisation."""
    base_unreach = baseline.metric(PRIMARY_METRIC)
    for intervention in (retrofit, harden):
        modified = Portfolio("P", (intervention,)).apply_to_city(small_city)
        result = run_monte_carlo(modified, small_config, n_simulations=N, seed=SEED)
        after = result.metric(PRIMARY_METRIC)
        assert np.all(after <= base_unreach + 1e-9)


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_reproduces_the_whole_comparison(small_city, small_config):
    first = compare_portfolios(small_city, small_config, seed=SEED, n_simulations=N)
    second = compare_portfolios(small_city, small_config, seed=SEED, n_simulations=N)
    assert first.baseline_outcome == second.baseline_outcome
    assert first.ranked_outcomes == second.ranked_outcomes


def test_identical_portfolios_give_identical_outcomes(retrofit, small_city, small_config, baseline):
    indices = tuple(range(N))
    a = evaluate_portfolio(
        small_city, small_config, Portfolio("A", (retrofit,)),
        seed=SEED, realisation_indices=indices, baseline_result=baseline,
    )
    b = evaluate_portfolio(
        small_city, small_config, Portfolio("A", (retrofit,)),
        seed=SEED, realisation_indices=indices, baseline_result=baseline,
    )
    assert a.primary_benefit_mean == b.primary_benefit_mean
    assert a.secondary == b.secondary


def test_different_seed_changes_the_stochastic_result(small_city, small_config):
    a = compare_portfolios(small_city, small_config, seed=SEED, n_simulations=N)
    b = compare_portfolios(small_city, small_config, seed=SEED + 101, n_simulations=N)
    # The damage the baseline sees must differ between seeds.
    assert a.baseline_outcome.secondary["n_collapse_after"] != (
        b.baseline_outcome.secondary["n_collapse_after"]
    )


# --------------------------------------------------------------------------- #
# benefit, secondary effects, uncertainty
# --------------------------------------------------------------------------- #


def test_baseline_outcome_has_zero_benefit_and_undefined_per_cost(comparison):
    baseline = comparison.baseline_outcome
    assert baseline.portfolio_cost == 0.0
    assert baseline.primary_benefit_mean == 0.0
    assert baseline.probability_of_improvement == 0.0
    assert math.isnan(baseline.benefit_per_cost)  # zero-cost edge case


def test_retrofit_reduces_expected_collapses(comparison):
    retrofit = _by_id(comparison, "RETROFIT")
    assert retrofit.secondary["delta_n_collapse"] < 0.0


def test_hospital_support_reduces_service_pressure_only(comparison):
    support = _by_id(comparison, "SUPPORT")
    # Its whole point: pressure down, primary objective untouched.
    assert support.secondary["delta_service_pressure"] < 0.0
    assert support.primary_benefit_mean == pytest.approx(0.0, abs=1e-9)


def test_benefit_is_the_paired_baseline_minus_after(comparison):
    for outcome in comparison.ranked_outcomes:
        assert outcome.primary_benefit_mean == pytest.approx(
            outcome.primary_objective_baseline - outcome.primary_objective_after
        )


def test_uncertainty_band_is_ordered(comparison):
    for outcome in comparison.ranked_outcomes:
        assert outcome.primary_benefit_p05 <= outcome.primary_benefit_p50 <= outcome.primary_benefit_p95
        assert 0.0 <= outcome.probability_of_improvement <= 1.0


def test_ranking_is_best_first_by_primary_benefit(comparison):
    benefits = [o.primary_benefit_mean for o in comparison.ranked_outcomes]
    assert benefits == sorted(benefits, reverse=True)


def test_best_is_the_top_ranked_feasible_portfolio(comparison):
    assert comparison.best is comparison.ranked_outcomes[0]
    assert comparison.best.within_budget


def test_all_ranked_portfolios_are_within_budget(comparison):
    for outcome in comparison.ranked_outcomes:
        assert outcome.portfolio_cost <= comparison.budget + 1e-6
        assert outcome.within_budget


# --------------------------------------------------------------------------- #
# Phase 5.1 — out-of-sample target-selection split
# --------------------------------------------------------------------------- #


def test_split_is_deterministic_and_correctly_sized():
    selection, evaluation = split_realisations(SEED, 100, 0.30)
    again = split_realisations(SEED, 100, 0.30)
    assert (selection, evaluation) == again  # deterministic given the seed
    assert len(selection) == 30
    assert len(evaluation) == 70


def test_split_has_no_overlap_and_covers_all_indices():
    selection, evaluation = split_realisations(SEED, 100, 0.30)
    assert set(selection).isdisjoint(evaluation)
    assert set(selection) | set(evaluation) == set(range(100))


def test_split_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="at least 2"):
        split_realisations(SEED, 1, 0.30)
    with pytest.raises(ValueError, match="selection_fraction"):
        split_realisations(SEED, 100, 0.0)
    with pytest.raises(ValueError, match="selection_fraction"):
        split_realisations(SEED, 100, 1.0)


def test_comparison_records_a_disjoint_split(comparison):
    assert set(comparison.selection_indices).isdisjoint(comparison.evaluation_indices)
    assert comparison.n_selection + comparison.n_evaluation == comparison.n_simulations
    assert comparison.n_selection >= 1 and comparison.n_evaluation >= 1


def test_hardening_targets_come_from_the_selection_set_only(comparison, small_city, small_config):
    """The HARDEN targets must be reproducible from the SELECTION realisations
    alone -- proof that the reported evaluation set played no part in selecting
    them."""
    harden_outcome = _by_id(comparison, "HARDEN")
    chosen = tuple(harden_outcome.target_ids["ROAD_HARDENING"])

    selection_run = run_monte_carlo(
        small_city, small_config,
        realisation_indices=comparison.selection_indices, seed=SEED,
    )
    from_selection = road_hardening(
        small_city, small_config, selection_run.link_closed_frequency
    ).target_ids
    assert chosen == from_selection


def test_reported_metrics_use_the_evaluation_set_only(comparison):
    """Every reported outcome is sized to the evaluation set, never the total or
    the selection set."""
    assert comparison.baseline_outcome.n_simulations == comparison.n_evaluation
    for outcome in comparison.ranked_outcomes:
        assert outcome.n_simulations == comparison.n_evaluation
    assert comparison.n_evaluation < comparison.n_simulations


def test_same_seed_reproduces_split_and_results(small_city, small_config):
    a = compare_portfolios(small_city, small_config, seed=SEED, n_simulations=N)
    b = compare_portfolios(small_city, small_config, seed=SEED, n_simulations=N)
    assert a.selection_indices == b.selection_indices
    assert a.evaluation_indices == b.evaluation_indices
    assert a.ranked_outcomes == b.ranked_outcomes


def test_different_seed_changes_the_split_or_the_draws(small_city, small_config):
    a = compare_portfolios(small_city, small_config, seed=SEED, n_simulations=N)
    b = compare_portfolios(small_city, small_config, seed=SEED + 313, n_simulations=N)
    # A different seed changes the permutation and/or the underlying draws, so
    # the baseline the evaluation set sees must differ.
    assert (a.selection_indices != b.selection_indices) or (
        a.baseline_outcome.secondary["n_collapse_after"]
        != b.baseline_outcome.secondary["n_collapse_after"]
    )


def test_crn_pairing_holds_inside_the_evaluation_set(comparison, small_city, small_config, harden):
    """Within the evaluation set, hardening can only help: post-event unreachable
    population is <= baseline in every paired evaluation realisation."""
    eval_idx = comparison.evaluation_indices
    base = run_monte_carlo(small_city, small_config, realisation_indices=eval_idx, seed=SEED)
    modified = Portfolio("H", (harden,)).apply_to_city(small_city)
    post = run_monte_carlo(modified, small_config, realisation_indices=eval_idx, seed=SEED)
    assert np.all(post.metric(PRIMARY_METRIC) <= base.metric(PRIMARY_METRIC) + 1e-9)


# --------------------------------------------------------------------------- #
# the real primary-objective signal (needs the heavy tail; full city)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_road_hardening_reduces_expected_unreachable_population(city, config):
    """On the full city, unreachability events are rare but real. Data-driven
    hardening must produce a strictly positive expected reduction, and never a
    negative one in any paired realisation."""
    n = 1000
    baseline = run_monte_carlo(city, config, n_simulations=n, seed=SEED)
    harden = road_hardening(city, config, baseline.link_closed_frequency)
    modified = Portfolio("H", (harden,)).apply_to_city(city)
    result = run_monte_carlo(modified, config, n_simulations=n, seed=SEED)

    base_unreach = baseline.metric(PRIMARY_METRIC)
    after = result.metric(PRIMARY_METRIC)
    benefit = base_unreach - after
    assert np.all(after <= base_unreach + 1e-9)  # CRN monotonicity
    assert benefit.mean() > 0.0  # a real, positive expected reduction
    assert (benefit > 0).sum() >= 1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _by_id(comparison, portfolio_id):
    for outcome in comparison.ranked_outcomes:
        if outcome.portfolio_id == portfolio_id:
            return outcome
    raise AssertionError(f"portfolio {portfolio_id!r} not found in ranked outcomes")
