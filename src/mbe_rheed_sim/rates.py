import math

# 2022 CODATA recommended value: https://physics.nist.gov/cgi-bin/cuu/Value?kev
BOLTZMANN_EV_PER_K = 8.617_333_262e-5


def arrhenius_rate(attempt_frequency_hz: float, barrier_ev: float, temperature_k: float) -> float:
    if attempt_frequency_hz < 0 or barrier_ev < 0 or temperature_k <= 0:
        raise ValueError("rate inputs require frequency/barrier >= 0 and temperature > 0")
    return attempt_frequency_hz * math.exp(-barrier_ev / (BOLTZMANN_EV_PER_K * temperature_k))


def diffusion_rate(
    attempt_frequency_hz: float,
    diffusion_barrier_ev: float,
    lateral_bond_energy_ev: float,
    bonds: int,
    temperature_k: float,
) -> float:
    if not 0 <= bonds <= 6:
        raise ValueError("a hexagonal site has between zero and six lateral bonds")
    return arrhenius_rate(
        attempt_frequency_hz,
        diffusion_barrier_ev + bonds * lateral_bond_energy_ev,
        temperature_k,
    )
