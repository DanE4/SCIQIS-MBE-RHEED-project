import numpy as np

from mbe_rheed_sim.observables import (
    coverage_ml,
    island_sizes,
    layer_coverages,
    rms_roughness_ml,
    step_density,
)
from mbe_rheed_sim.rheed import step_density_proxy


def test_flat_and_hand_checkable_observables() -> None:
    flat = np.ones((3, 3), dtype=np.int64)
    assert coverage_ml(flat) == 1.0
    assert rms_roughness_ml(flat) == 0.0
    assert step_density(flat) == 0.0
    assert step_density_proxy(flat) == 1.0
    assert layer_coverages(flat).tolist() == [1.0]

    one_island = flat.copy()
    one_island[0, 0] = 2
    assert island_sizes(one_island) == [1]
    assert layer_coverages(one_island).tolist() == [1.0, 1 / 9]
