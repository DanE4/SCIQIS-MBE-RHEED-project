import math

# 2022 CODATA recommended value: https://physics.nist.gov/cgi-bin/cuu/Value?kev
BOLTZMANN_EV_PER_K = 8.617_333_262e-5


def arrhenius_rate(attempt_frequency_hz: float, barrier_ev: float, temperature_k: float) -> float:
    """Scalar reference rate. kmc.py computes the same law vectorized over the lattice."""
    if attempt_frequency_hz < 0 or barrier_ev < 0 or temperature_k <= 0:
        raise ValueError("rate inputs require frequency/barrier >= 0 and temperature > 0")
    return attempt_frequency_hz * math.exp(-barrier_ev / (BOLTZMANN_EV_PER_K * temperature_k))
