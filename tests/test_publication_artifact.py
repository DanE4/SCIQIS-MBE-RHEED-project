import json
from itertools import pairwise
from pathlib import Path

import pytest
from reproduce_figure3 import LATTICE_SIZE, SEEDS

ROOT = Path(__file__).resolve().parents[1]


def test_publication_inputs_are_traceable_and_distinguish_signals() -> None:
    reference = json.loads(
        (ROOT / "data/reference/figure3_experimental_digitized.json").read_text()
    )
    artifact = json.loads((ROOT / "data/processed/figure3_simulated_reduced.json").read_text())

    assert reference["classification"] == "digitized visual reference; not raw experimental data"
    assert [trace["nominal_ga_n_ratio"] for trace in reference["traces"]] == [0.89, 0.82, 0.68]
    assert all(
        all(right > left for left, right in pairwise(trace["time_s"]))
        for trace in reference["traces"]
    )
    assert artifact["signal_definitions"]["reference"] != artifact["signal_definitions"][
        "simulation"
    ]
    # Against the generating script's defaults, so regenerating at a new size or ensemble
    # updates one constant instead of drifting silently away from the committed artifact.
    assert artifact["provenance"]["seeds"] == list(SEEDS)
    assert len(artifact["provenance"]["code_version"]["generation_source_sha256"]) == 64
    assert len(artifact["comparisons"]) == 3
    assert all(
        trace["simulation_config"]["lattice_size"] == LATTICE_SIZE
        for trace in artifact["traces"]
    )
    assert [
        frame["target_predicted_coverage_ml"]
        for frame in artifact["morphology_sequence"]["frames"]
    ] == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_morphology_montage_covers_both_coverages_at_every_paper_ratio() -> None:
    montage = json.loads(
        (ROOT / "data/processed/figure3_simulated_reduced.json").read_text()
    )["morphology_montage"]

    assert montage["nominal_ga_n_ratios"] == [0.68, 0.82, 0.89]
    assert montage["target_coverages_ml"] == [0.5, 1.0]
    assert {
        (panel["nominal_ga_n_ratio"], panel["target_predicted_coverage_ml"])
        for panel in montage["panels"]
    } == {(ratio, coverage) for ratio in [0.68, 0.82, 0.89] for coverage in [0.5, 1.0]}
    # The scope wording is the guard against this figure being read as a GaN/AlN SK result.
    assert "not a strain" in montage["classification"]

    for panel in montage["panels"]:
        # The figure ships as a committed PNG, so the artifact carries numbers, not surfaces.
        assert "height_ml" not in panel
        # The proxy is the complement of the step density it is computed from.
        assert panel["step_density"] + panel["rheed_proxy"] == pytest.approx(1.0)
        assert 0.0 <= panel["rheed_proxy"] <= 1.0
