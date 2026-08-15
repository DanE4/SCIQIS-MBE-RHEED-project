"""Small ensemble-analysis helpers for reproducible parameter studies."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mbe_rheed_sim.kmc import SimulationResult


@dataclass(frozen=True, slots=True)
class OscillationMetrics:
    is_oscillatory: bool
    detrended_amplitude: float
    peak_count: int
    trough_count: int
    peak_to_trough_amplitude: float | None
    period_ml: float | None
    period_deviation_ml: float | None
    spectral_power_fraction: float
    peak_phase_ml: float | None
    trough_phase_ml: float | None
    damping_rate_per_ml: float | None


def result_array_bytes(result: SimulationResult) -> int:
    """Return the reproducible storage footprint of arrays retained in one result.

    Derived from the slots so a new array field is counted without editing a name list;
    the config and the scalar event counters are not arrays and drop out.
    """
    values = (getattr(result, name) for name in result.__slots__)
    return sum(value.nbytes for value in values if isinstance(value, np.ndarray))


def successive_size_check(
    smaller_size: int,
    larger_size: int,
    smaller_mean: float,
    larger_mean: float,
    smaller_std: float,
    larger_std: float,
    seed_count: int,
    *,
    relative_tolerance: float = 0.10,
) -> dict[str, int | float | bool]:
    """Apply the project's preliminary uncertainty-aware finite-size rule."""
    if (
        smaller_size <= 0
        or larger_size <= smaller_size
        or seed_count < 2
        or smaller_std < 0
        or larger_std < 0
        or not 0 < relative_tolerance < 1
    ):
        raise ValueError("valid successive sizes, uncertainties, and tolerance are required")
    tolerance = relative_tolerance * abs(larger_mean)
    difference = abs(larger_mean - smaller_mean)
    pooled_se = np.sqrt((smaller_std**2 + larger_std**2) / seed_count)
    upper_difference = difference + 1.96 * pooled_se
    return {
        "smaller_lattice_size": smaller_size,
        "larger_lattice_size": larger_size,
        "absolute_mean_difference": difference,
        "tolerance": tolerance,
        "difference_plus_1.96_pooled_standard_errors": upper_difference,
        "passes": bool(upper_difference <= tolerance),
    }


def oscillation_amplitude(values: NDArray[np.float64]) -> float:
    """Robust half peak-to-trough amplitude using the 5th and 95th percentiles."""
    if values.ndim != 1 or values.size < 2:
        raise ValueError("oscillation amplitude requires a one-dimensional trace")
    low, high = np.quantile(values, (0.05, 0.95))
    return float((high - low) / 2.0)


def rheed_oscillation_metrics(
    coverage_ml: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    expected_period_ml: float = 1.0,
    period_tolerance_ml: float = 0.5,
) -> OscillationMetrics:
    """Measure layer-scale periodicity after linear detrending.

    The classification is diagnostic: it requires at least two separated peaks and troughs
    plus a median peak spacing inside the requested period window. It is not evidence of
    agreement with experimental RHEED intensity.
    """
    coordinate = np.asarray(coverage_ml, dtype=float)
    signal = np.asarray(values, dtype=float)
    if (
        coordinate.ndim != 1
        or signal.ndim != 1
        or coordinate.size != signal.size
        or coordinate.size < 8
        or not np.all(np.isfinite(coordinate))
        or not np.all(np.isfinite(signal))
        or np.any(np.diff(coordinate) <= 0)
        or expected_period_ml <= 0
        or not 0 < period_tolerance_ml < expected_period_ml
    ):
        raise ValueError(
            "oscillation metrics require finite, increasing 1D data and a valid period"
        )

    grid = np.linspace(coordinate[0], coordinate[-1], coordinate.size)
    detrended = np.interp(grid, coordinate, signal)
    detrended -= np.polyval(np.polyfit(grid, detrended, 1), grid)
    detrended_amplitude = float(np.sqrt(2.0) * np.std(detrended))
    numerical_floor = 100.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(signal))))
    if detrended_amplitude <= numerical_floor:
        detrended.fill(0.0)
        detrended_amplitude = 0.0
    spacing = float(grid[1] - grid[0])
    smoothing_points = max(1, round(0.15 * expected_period_ml / spacing))
    if smoothing_points % 2 == 0:
        smoothing_points += 1
    padding = smoothing_points // 2
    smooth = np.convolve(
        np.pad(detrended, padding, mode="edge"),
        np.ones(smoothing_points) / smoothing_points,
        mode="valid",
    )

    peak_candidates = (
        np.flatnonzero((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])) + 1
    )
    trough_candidates = (
        np.flatnonzero((smooth[1:-1] < smooth[:-2]) & (smooth[1:-1] <= smooth[2:])) + 1
    )
    separation = max(1, round(0.5 * expected_period_ml / spacing))

    def separated_extrema(candidates: NDArray[np.int64], *, peaks: bool) -> NDArray[np.int64]:
        order = np.argsort(smooth[candidates])
        if peaks:
            order = order[::-1]
        selected: list[int] = []
        for index in candidates[order]:
            if all(abs(int(index) - previous) >= separation for previous in selected):
                selected.append(int(index))
        return np.asarray(sorted(selected), dtype=np.int64)

    peaks = separated_extrema(peak_candidates, peaks=True)
    troughs = separated_extrema(trough_candidates, peaks=False)
    period_ml = float(np.median(np.diff(grid[peaks]))) if peaks.size >= 2 else None
    peak_to_trough = (
        float(np.median(smooth[peaks]) - np.median(smooth[troughs]))
        if peaks.size and troughs.size
        else None
    )

    def circular_phase(indices: NDArray[np.int64]) -> float | None:
        if not indices.size:
            return None
        phase = np.angle(np.mean(np.exp(2j * np.pi * grid[indices] / expected_period_ml)))
        return float((phase / (2.0 * np.pi)) % 1.0 * expected_period_ml)

    windowed = detrended * np.hanning(detrended.size)
    frequencies = np.fft.rfftfreq(detrended.size, d=spacing)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    positive_power = float(power[frequencies > 0].sum())
    minimum_period = expected_period_ml - period_tolerance_ml
    maximum_period = expected_period_ml + period_tolerance_ml
    expected_band = (frequencies >= 1.0 / maximum_period) & (frequencies <= 1.0 / minimum_period)
    spectral_fraction = (
        float(power[expected_band].sum() / positive_power) if positive_power > 0 else 0.0
    )

    peak_envelope = np.abs(smooth[peaks])
    usable_envelope = peak_envelope > np.finfo(float).eps
    damping_rate = (
        float(
            np.polyfit(
                grid[peaks][usable_envelope],
                np.log(peak_envelope[usable_envelope]),
                1,
            )[0]
        )
        if np.count_nonzero(usable_envelope) >= 3
        else None
    )
    period_deviation = abs(period_ml - expected_period_ml) if period_ml is not None else None
    is_oscillatory = bool(
        peaks.size >= 2
        and troughs.size >= 2
        and period_ml is not None
        and period_deviation <= period_tolerance_ml
        and detrended_amplitude > 10.0 * np.finfo(float).eps
    )
    return OscillationMetrics(
        is_oscillatory=is_oscillatory,
        detrended_amplitude=detrended_amplitude,
        peak_count=int(peaks.size),
        trough_count=int(troughs.size),
        peak_to_trough_amplitude=peak_to_trough,
        period_ml=period_ml,
        period_deviation_ml=period_deviation,
        spectral_power_fraction=spectral_fraction,
        peak_phase_ml=circular_phase(peaks),
        trough_phase_ml=circular_phase(troughs),
        damping_rate_per_ml=damping_rate,
    )


def run_summary(result: SimulationResult, seed: int, elapsed: float) -> dict[str, object]:
    """One per-seed record shared by every convergence and benchmark script."""
    return {
        "seed": seed,
        "elapsed_s": elapsed,
        "result_array_bytes": result_array_bytes(result),
        "events": {
            "deposited": result.deposited_events,
            "selected_diffusion": result.selected_diffusion_events,
            "equivalent_nearest_neighbor_hops": result.diffusion_events,
            "long_hops": result.long_hop_events,
            "desorbed": result.desorbed_events,
        },
        "final": {
            "rms_roughness_ml": float(result.roughness_ml[-1]),
            "step_density": float(1.0 - result.rheed_proxy[-1]),
            "mean_height_ml": float(result.final_heights.mean()),
            "height_std_ml": float(result.final_heights.std()),
            "minimum_height_ml": int(result.final_heights.min()),
            "maximum_height_ml": int(result.final_heights.max()),
            "occupied_site_fraction": float(np.mean(result.final_heights > 0)),
        },
    }
