"""Parameterization used for the paper's Figure 3 GaN homoepitaxy comparison."""

import math
from dataclasses import asdict, dataclass

from mbe_rheed_sim.config import SimulationConfig
from mbe_rheed_sim.rates import BOLTZMANN_EV_PER_K

FIGURE3_TEMPERATURE_K = 730.0 + 273.15
FIGURE3_NITROGEN_FLUX_ML_S = 0.28
FIGURE3_NOMINAL_GA_N_RATIOS = (0.89, 0.82, 0.68)
GAN_DECOMPOSITION_ATTEMPT_FREQUENCY_HZ = 6.3e13
GAN_DECOMPOSITION_BARRIER_EV = 3.1


@dataclass(frozen=True, slots=True)
class Figure3Parameters:
    nominal_ga_n_ratio: float
    temperature_k: float
    nitrogen_flux_ml_s: float
    nominal_ga_flux_ml_s: float
    effective_ga_flux_ml_s: float
    effective_ga_n_ratio: float
    predicted_growth_rate_ml_s: float
    diffusion_barrier_ev: float
    lateral_bond_energy_ev: float
    desorption_barrier_ev: float
    step_barrier_ev: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def gan_decomposition_flux(temperature_k: float) -> float:
    """Equation A1: thermal GaN decomposition rate in ML/s."""
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    return GAN_DECOMPOSITION_ATTEMPT_FREQUENCY_HZ * math.exp(
        -GAN_DECOMPOSITION_BARRIER_EV / (BOLTZMANN_EV_PER_K * temperature_k)
    )


def effective_ga_flux(
    nominal_ga_flux_ml_s: float,
    nitrogen_flux_ml_s: float,
    temperature_k: float,
) -> tuple[float, float]:
    """Equations A3, A4, A8, and A9 in the paper's N-rich flux model."""
    if nominal_ga_flux_ml_s < 0 or nitrogen_flux_ml_s <= 0:
        raise ValueError("nominal Ga flux must be non-negative and N flux must be positive")

    decomposition_flux = gan_decomposition_flux(temperature_k)
    if nominal_ga_flux_ml_s > nitrogen_flux_ml_s + decomposition_flux:
        raise ValueError("effective-flux model is restricted to the paper's N-rich regime")

    a = (nitrogen_flux_ml_s - 2.0 * decomposition_flux) / (
        nitrogen_flux_ml_s + 2.0 * decomposition_flux
    )
    b = nitrogen_flux_ml_s / (
        decomposition_flux * (nitrogen_flux_ml_s + 2.0 * decomposition_flux)
    )
    sticking = a + (1.0 - a) / (
        1.0
        + math.exp(
            b * (nominal_ga_flux_ml_s - nitrogen_flux_ml_s - decomposition_flux)
        )
    )
    flux = sticking * (nominal_ga_flux_ml_s + decomposition_flux)
    return flux, flux - decomposition_flux


def figure3_parameters(nominal_ga_n_ratio: float) -> Figure3Parameters:
    """Return the fitted Equation 8 parameters for one Figure 3 flux ratio."""
    if nominal_ga_n_ratio not in FIGURE3_NOMINAL_GA_N_RATIOS:
        raise ValueError(
            f"Figure 3 ratio must be one of {FIGURE3_NOMINAL_GA_N_RATIOS}"
        )

    nominal_ga_flux = nominal_ga_n_ratio * FIGURE3_NITROGEN_FLUX_ML_S
    effective_flux, growth_rate = effective_ga_flux(
        nominal_ga_flux,
        FIGURE3_NITROGEN_FLUX_ML_S,
        FIGURE3_TEMPERATURE_K,
    )
    eta = effective_flux / FIGURE3_NITROGEN_FLUX_ML_S
    eta_power = eta**14.4
    reference_power = 0.75**14.4
    return Figure3Parameters(
        nominal_ga_n_ratio=nominal_ga_n_ratio,
        temperature_k=FIGURE3_TEMPERATURE_K,
        nitrogen_flux_ml_s=FIGURE3_NITROGEN_FLUX_ML_S,
        nominal_ga_flux_ml_s=nominal_ga_flux,
        effective_ga_flux_ml_s=effective_flux,
        effective_ga_n_ratio=eta,
        predicted_growth_rate_ml_s=growth_rate,
        diffusion_barrier_ev=1.78 - 4.87e-3 * math.exp(4.0 * eta),
        lateral_bond_energy_ev=0.41 - 0.12 * eta,
        desorption_barrier_ev=2.53 - 0.17 * eta,
        step_barrier_ev=0.12 - 0.06 * eta_power / (reference_power + eta_power),
    )


def figure3_config(
    nominal_ga_n_ratio: float,
    *,
    lattice_size: int = 7,
    duration_s: float = 40.0,
    seed: int = 0,
) -> SimulationConfig:
    """Small executable Figure 3 configuration using the paper's rates and time axis."""
    parameters = figure3_parameters(nominal_ga_n_ratio)
    rate_ratio = math.exp(
        parameters.lateral_bond_energy_ev
        / (BOLTZMANN_EV_PER_K * parameters.temperature_k)
    )
    hop_distance = min(
        math.isqrt(math.floor(rate_ratio)), max(1, (lattice_size - 1) // 2)
    )
    return SimulationConfig(
        lattice_size=lattice_size,
        target_coverage_ml=None,
        target_time_s=duration_s,
        temperature_k=parameters.temperature_k,
        deposition_flux_ml_s=parameters.effective_ga_flux_ml_s,
        attempt_frequency_hz=1e13,
        diffusion_barrier_ev=parameters.diffusion_barrier_ev,
        lateral_bond_energy_ev=parameters.lateral_bond_energy_ev,
        step_barrier_ev=parameters.step_barrier_ev,
        desorption_barrier_ev=parameters.desorption_barrier_ev,
        max_isolated_hop_distance=hop_distance,
        sample_every_ml=0.05,
        seed=seed,
        max_events=10_000_000,
    )
