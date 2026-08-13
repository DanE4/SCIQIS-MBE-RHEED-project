"""Measure short paper-regime runs before scheduling large ensembles."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from mbe_rheed_sim import run
from mbe_rheed_sim.paper import figure3_config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "runs" / "large_lattice_benchmark.json"
RATIO = 0.82
DURATION_S = 0.1
SIZES = (64, 128, 256)
SEED = 0


def main() -> None:
    measurements = []
    for size in SIZES:
        started = perf_counter()
        result = run(figure3_config(RATIO, lattice_size=size, duration_s=DURATION_S, seed=SEED))
        elapsed = perf_counter() - started
        events = result.deposited_events + result.selected_diffusion_events + result.desorbed_events
        measurements.append(
            {
                "lattice_size": size,
                "elapsed_s": elapsed,
                "simulated_duration_s": DURATION_S,
                "event_throughput_per_wall_s": events / elapsed,
                "events": {
                    "deposited": result.deposited_events,
                    "selected_diffusion": result.selected_diffusion_events,
                    "equivalent_nearest_neighbor_hops": result.diffusion_events,
                    "long_hops": result.long_hop_events,
                    "desorbed": result.desorbed_events,
                },
                "result_array_bytes": sum(
                    array.nbytes
                    for array in (
                        result.final_heights,
                        result.coverage_ml,
                        result.time_s,
                        result.roughness_ml,
                        result.island_density_per_site,
                        result.rheed_proxy,
                        result.snapshots,
                    )
                ),
                "final_rms_roughness_ml": float(result.roughness_ml[-1]),
                "final_step_density": float(1.0 - result.rheed_proxy[-1]),
                "maximum_height_ml": int(result.final_heights.max()),
                "occupied_site_fraction": float(np.mean(result.final_heights > 0)),
            }
        )

    output = {
        "purpose": "short runtime envelope only; not a convergence or physical-duration run",
        "nominal_ga_n_ratio": RATIO,
        "duration_s": DURATION_S,
        "seed": SEED,
        "measurements": measurements,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
