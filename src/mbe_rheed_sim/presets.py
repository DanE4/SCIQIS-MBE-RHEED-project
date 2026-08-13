"""Named runtime presets for interactive and publication workflows."""

from mbe_rheed_sim.config import SimulationConfig


def interactive_config(*, seed: int = 7) -> SimulationConfig:
    return SimulationConfig(lattice_size=16, target_coverage_ml=2.0, seed=seed)


def publication_config(*, seed: int = 0) -> SimulationConfig:
    return SimulationConfig(
        lattice_size=64,
        target_coverage_ml=2.0,
        sample_every_ml=0.05,
        seed=seed,
        max_events=20_000_000,
    )
