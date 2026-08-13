import pytest

from mbe_rheed_sim.paper import effective_ga_flux, figure3_config, figure3_parameters


def test_figure3_parameterization() -> None:
    high = figure3_parameters(0.89)
    low = figure3_parameters(0.68)

    assert high.temperature_k == pytest.approx(1003.15)
    assert high.nominal_ga_flux_ml_s == pytest.approx(0.2492)
    assert high.effective_ga_flux_ml_s == pytest.approx(0.26179078296)
    assert high.effective_ga_n_ratio == pytest.approx(0.934967081996)
    assert high.predicted_growth_rate_ml_s == pytest.approx(0.244999376552)
    assert high.diffusion_barrier_ev == pytest.approx(1.57500978115)
    assert high.lateral_bond_energy_ev == pytest.approx(0.29780395016)
    assert high.desorption_barrier_ev == pytest.approx(2.37105559606)
    assert high.step_barrier_ev == pytest.approx(0.06240864328)
    assert high.diffusion_barrier_ev < low.diffusion_barrier_ev

    with pytest.raises(ValueError, match="N-rich"):
        effective_ga_flux(0.4, 0.28, 1003.15)
    with pytest.raises(ValueError, match="Figure 3 ratio"):
        figure3_parameters(0.8)

    config = figure3_config(0.82, lattice_size=16, seed=9)
    assert config.target_coverage_ml is None
    assert config.target_time_s == 40.0
    assert config.attempt_frequency_hz == 1e13
    assert config.max_isolated_hop_distance == 5
