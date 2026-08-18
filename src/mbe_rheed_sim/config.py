from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Inputs for the baseline model.

    Energetic defaults are fast demonstration parameters, not GaN/AlN measurements.
    """

    lattice_size: int = 12
    target_coverage_ml: float | None = 2.0
    target_time_s: float | None = None
    temperature_k: float = 800.0
    deposition_flux_ml_s: float = 0.5
    attempt_frequency_hz: float = 1_000.0
    diffusion_barrier_ev: float = 0.15
    lateral_bond_energy_ev: float = 0.05
    step_barrier_ev: float = 0.05
    desorption_barrier_ev: float = 0.65
    max_isolated_hop_distance: int = 1
    sample_every_ml: float = 0.05
    seed: int = 0
    max_events: int | None = 2_000_000

    def __post_init__(self) -> None:
        if self.lattice_size < 2:
            raise ValueError("lattice_size must be at least 2")
        if self.target_coverage_ml is None and self.target_time_s is None:
            raise ValueError("target coverage or target time is required")
        if self.target_coverage_ml is not None and self.target_coverage_ml <= 0:
            raise ValueError("target_coverage_ml must be positive")
        if self.target_time_s is not None and self.target_time_s <= 0:
            raise ValueError("target_time_s must be positive")
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if self.deposition_flux_ml_s <= 0:
            raise ValueError("deposition_flux_ml_s must be positive")
        if self.attempt_frequency_hz < 0:
            raise ValueError("attempt_frequency_hz cannot be negative")
        if any(
            energy < 0
            for energy in (
                self.diffusion_barrier_ev,
                self.lateral_bond_energy_ev,
                self.step_barrier_ev,
                self.desorption_barrier_ev,
            )
        ):
            raise ValueError("energy barriers cannot be negative")
        if self.sample_every_ml <= 0:
            raise ValueError("sample_every_ml must be positive")
        if not 1 <= self.max_isolated_hop_distance <= max(1, (self.lattice_size - 1) // 2):
            raise ValueError("max_isolated_hop_distance is too large for the periodic lattice")
        if self.max_events is not None and self.max_events < 1:
            raise ValueError("max_events must be positive, or None for no limit")
