import numpy as np
import pytest

from mbe_rheed_sim.lattice import deposit, empty_lattice, hop, neighbors


def test_periodic_neighbors_and_valid_transitions() -> None:
    assert set(neighbors(0, 0, 3)) == {(0, 1), (0, 2), (1, 0), (2, 0), (1, 2), (2, 1)}

    heights = empty_lattice(3)
    deposit(heights, 0, 0)
    before = int(heights.sum())
    hop(heights, (0, 0), (0, 2))
    assert int(heights.sum()) == before
    assert np.all(heights >= 0)
    with pytest.raises(ValueError):
        hop(heights, (0, 0), (0, 1))
