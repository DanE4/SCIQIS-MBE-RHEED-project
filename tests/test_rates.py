import pytest

from mbe_rheed_sim.rates import arrhenius_rate, desorption_rate, diffusion_rate


def test_arrhenius_rate_sanity() -> None:
    base = arrhenius_rate(1_000.0, 0.3, 800.0)
    assert base > 0
    assert arrhenius_rate(1_000.0, 0.4, 800.0) < base
    assert arrhenius_rate(1_000.0, 0.3, 900.0) > base
    assert diffusion_rate(1_000.0, 0.3, 0.05, 2, 800.0) < base
    assert diffusion_rate(1_000.0, 0.3, 0.05, 0, 800.0, 0.1) < base
    assert desorption_rate(1_000.0, 0.3, 0.05, 2, 800.0) < base
    with pytest.raises(ValueError):
        arrhenius_rate(1_000.0, 0.3, 0.0)
