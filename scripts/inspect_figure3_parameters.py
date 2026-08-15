"""Print the paper-derived Figure 3 inputs and brute-force KMC rate diagnostics."""

import json
from dataclasses import asdict

from mbe_rheed_sim.paper import FIGURE3_NOMINAL_GA_N_RATIOS, figure3_parameters
from mbe_rheed_sim.rates import arrhenius_rate

ATTEMPT_FREQUENCY_HZ = 1e13


def main() -> None:
    rows = []
    for ratio in FIGURE3_NOMINAL_GA_N_RATIOS:
        parameters = figure3_parameters(ratio)
        # Isolated adatom: zero lateral bonds, so the barrier is the bare energy.
        isolated_diffusion_rate = arrhenius_rate(
            ATTEMPT_FREQUENCY_HZ, parameters.diffusion_barrier_ev, parameters.temperature_k
        )
        isolated_desorption_rate = arrhenius_rate(
            ATTEMPT_FREQUENCY_HZ, parameters.desorption_barrier_ev, parameters.temperature_k
        )
        rows.append(
            {
                **asdict(parameters),
                "isolated_diffusion_rate_hz": isolated_diffusion_rate,
                "isolated_desorption_rate_hz": isolated_desorption_rate,
                "isolated_hops_per_deposition_ml": (
                    isolated_diffusion_rate / parameters.effective_ga_flux_ml_s
                ),
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
