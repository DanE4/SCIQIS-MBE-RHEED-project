"""Checks for the notebook's presentation layer.

The notebook itself is only exercised by `make check`; these cover the logic that was
pulled out of it, so a broken form mapping or figure builder fails in CI rather than in a
live demo.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from mbe_rheed_notebook import batch, controls, figures
from mbe_rheed_sim import rheed
from mbe_rheed_sim.paper import FIGURE3_NOMINAL_GA_N_RATIOS


def test_hand_tuned_parameters_map_to_a_valid_config() -> None:
    config, estimate, growth_rate, name, detail = controls.build_run(
        controls.DEFAULT_PARAMETERS
    )
    assert config.lattice_size == 16
    assert config.target_coverage_ml == 2.0
    assert config.target_time_s is None
    assert growth_rate is None
    assert estimate > 0
    assert "Hand-tuned" in name and "seed 7" in detail


def test_paper_mode_overrides_growth_conditions_but_keeps_numerical_choices() -> None:
    parameters = controls.DEFAULT_PARAMETERS | {
        "experiment_mode": controls.FROM_PAPER,
        "figure3_ratio": 0.82,
        "stop_mode": "Physical time",
        "duration_s": 4.0,
        "seed": 3,
    }
    config, _estimate, growth_rate, name, _detail = controls.build_run(parameters)

    assert config.target_time_s == 4.0
    assert config.target_coverage_ml is None
    assert config.seed == 3
    assert growth_rate is not None and growth_rate > 0
    # Paper barriers, not the form's teaching defaults.
    assert config.diffusion_barrier_ev != controls.DEFAULT_PARAMETERS["barrier_ev"]
    assert "0.82" in name


def _gallery() -> dict:
    index = Path(__file__).resolve().parents[1] / "data" / "gallery" / "index.json"
    return json.loads(index.read_text())


@pytest.mark.parametrize("name", list(_gallery()))
def test_a_preset_reproduces_the_stored_run_it_names(name: str) -> None:
    """The form values a preset loads must map back to the stored config exactly.

    Sliders are stepped, so a scenario whose parameters fell between steps would load as
    something subtly different from the trajectory the caption describes.
    """
    meta = _gallery()[name]
    values = controls.preset_parameters(meta)
    config, *_ = controls.build_run(values)
    assert asdict(config) == meta["config"]
    # And the form must actually build on them: every dropdown value needs a matching option.
    controls.parameter_form(FIGURE3_NOMINAL_GA_N_RATIOS, on_change=lambda _: None, values=values)


def test_hop_distance_is_clamped_to_the_periodic_lattice() -> None:
    # 16 would exceed half of a 7x7 lattice; SimulationConfig would reject it.
    config, *_ = controls.build_run(
        controls.DEFAULT_PARAMETERS | {"size": 7, "hop_distance": 16}
    )
    assert config.max_isolated_hop_distance == 3


def test_expensive_gate_trips_on_large_lattices() -> None:
    small, small_estimate, *_ = controls.build_run(controls.DEFAULT_PARAMETERS)
    large, large_estimate, *_ = controls.build_run(
        controls.DEFAULT_PARAMETERS | {"size": 128}
    )
    assert not controls.is_expensive(small, small_estimate)
    assert controls.is_expensive(large, large_estimate)


@pytest.mark.parametrize(
    "builder", [figures.height_surface, figures.hex_cells, figures.step_edges]
)
def test_surface_builders_produce_one_trace(builder) -> None:
    heights = np.array([[0, 1], [2, 1]], dtype=np.int64)
    figure = builder(heights, coverage=1.0, zmax=2)
    assert len(figure.data) == 1


def test_step_edge_view_counts_unequal_neighbours_and_reports_the_proxy() -> None:
    heights = np.zeros((6, 6), dtype=np.int64)
    figure = figures.step_edges(heights, coverage=0.0, zmax=1)
    # A flat surface has no steps anywhere, so the proxy is 1 and every marker reads zero.
    assert not np.asarray(figure.data[0].marker.color).any()
    assert "1 - S_d = 1.000" in figure.layout.title.text

    heights[3, 3] = 1
    figure = figures.step_edges(heights, coverage=0.03, zmax=1)
    counts = np.asarray(figure.data[0].marker.color).reshape(heights.shape)
    assert counts[3, 3] == 6
    assert counts.sum() == 12


def test_detector_screen_marks_the_specular_beam_and_floors_the_log_scale() -> None:
    angle = rheed.antiphase_grazing_angle_deg(3)
    pattern = rheed.diffraction_screen(np.zeros((8, 8), dtype=np.int64), grazing_angle_deg=angle)
    figure = figures.detector_screen(pattern, coverage=0.0)
    screen, specular = figure.data
    assert screen.z.min() == pytest.approx(-figures.SCREEN_LOG_DECADES)
    assert screen.z.max() == pytest.approx(0.0)
    assert specular.x == (0.0,) and specular.y == (angle,)
    assert figures.DIFFRACTION_LABEL in figure.layout.title.text
    assert "anti-phase" in figure.layout.title.text


def test_rheed_trace_marks_the_current_frame() -> None:
    coverage = np.linspace(0.0, 2.0, 21)
    proxy = 0.75 + 0.2 * np.cos(2 * np.pi * coverage)
    figure = figures.rheed_trace(coverage, proxy, frame=5, axis_label="coverage (ML)")
    current = figure.data[-1]
    assert current.x[0] == pytest.approx(coverage[5])
    assert current.y[0] == pytest.approx(proxy[5])
    assert "kinematic specular (00) intensity" not in {trace.name for trace in figure.data}

    specular = 0.5 * (1 + np.cos(2 * np.pi * coverage))
    overlaid = figures.rheed_trace(
        coverage, proxy, frame=5, axis_label="coverage (ML)", specular=specular
    )
    assert "kinematic specular (00) intensity" in {trace.name for trace in overlaid.data}
    # The frame marker must stay last, since the notebook and this suite both read it there.
    assert overlaid.data[-1].y[0] == pytest.approx(proxy[5])


def test_batch_workflow_labels_match_the_cli() -> None:
    """The dropdown must not offer a workflow run_workflow.py would reject."""
    from run_workflow import WORKFLOWS

    assert set(batch.WORKFLOW_LABELS.values()) <= set(WORKFLOWS)


def test_expensive_batch_confirmation_covers_the_64_size_override() -> None:
    request = {"workflow": "figure3-convergence", "sizes": "8,16,32,64"}
    assert batch.needs_confirmation(request)
    assert not batch.needs_confirmation({"workflow": "sweep", "sizes": "8,16"})
    assert batch.needs_confirmation({"workflow": "benchmark-sizes", "sizes": ""})


def test_save_result_round_trips_through_the_saved_directory(tmp_path: Path) -> None:
    from mbe_rheed_sim import SimulationConfig, run
    from mbe_rheed_sim.kmc import SimulationResult

    result = run(SimulationConfig(lattice_size=6, target_coverage_ml=0.2, seed=3))
    message = controls.save_result(result, tmp_path, "my run/2")

    saved = tmp_path / "my_run_2.npz"
    assert saved.exists() and str(saved) in message
    assert SimulationResult.load_npz(saved).config == result.config
