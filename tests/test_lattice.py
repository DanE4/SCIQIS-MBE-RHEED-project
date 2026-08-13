import numpy as np
import pytest

from mbe_rheed_sim.lattice import (
    deposit,
    empty_lattice,
    hop,
    hop_allowed,
    long_hop,
    neighbors,
    open_terrace_hop_distance,
)


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

    heights[0, 0] = 1
    heights[0, 1] = 1
    assert hop_allowed(heights, (0, 0), (0, 1))  # one-step upward hop
    heights[0, 1] = 2
    assert not hop_allowed(heights, (0, 0), (0, 1))  # multi-step upward hop


def test_isolated_adatom_long_hop_stays_on_open_terrace() -> None:
    heights = empty_lattice(7)
    heights[3, 3] = 1
    assert open_terrace_hop_distance(heights, 3, 3, 3) == 3

    long_hop(heights, (3, 3), (0, 1), 3)
    assert heights[3, 3] == 0
    assert heights[3, 6] == 1

    heights[3, 1] = 1
    assert open_terrace_hop_distance(heights, 3, 6, 3) == 1
    with pytest.raises(ValueError, match="obstacle"):
        long_hop(heights, (3, 6), (0, 1), 3)
