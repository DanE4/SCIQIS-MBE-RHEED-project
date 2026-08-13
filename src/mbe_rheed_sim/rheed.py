from mbe_rheed_sim.lattice import HeightField
from mbe_rheed_sim.observables import step_density


def step_density_proxy(heights: HeightField) -> float:
    """Dimensionless morphology proxy; this is not a RHEED diffraction calculation."""
    return 1.0 - step_density(heights)
