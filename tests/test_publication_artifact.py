import json
from itertools import pairwise
from pathlib import Path

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
    assert artifact["provenance"]["seeds"] == [2026, 2027, 2028]
    assert len(artifact["provenance"]["code_version"]["generation_source_sha256"]) == 64
    assert len(artifact["comparisons"]) == 3
    assert all(trace["simulation_config"]["lattice_size"] == 7 for trace in artifact["traces"])
    assert [
        frame["target_predicted_coverage_ml"]
        for frame in artifact["morphology_sequence"]["frames"]
    ] == [0.0, 0.5, 1.0, 1.5, 2.0]
