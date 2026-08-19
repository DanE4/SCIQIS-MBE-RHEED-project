"""Kinematic RHEED from the model surface.

Everything else in this project reports `1 - S_d`, a morphology stand-in. This module
computes the other thing: the electron intensity a detector would collect, in the
**kinematic** (single-scattering) approximation.

Geometry pipeline
-----------------

The calculation follows the exact Ewald construction of Liu, Chang and Zou, J. Vac. Sci.
Technol. B **40**, 054002 (2022), and never approximates the Ewald sphere by a plane:

    (h, k)  ->  G_hk = h b1 + k b2  ->  Ewald intersection  ->  k_f  ->  (exit, deflection)

Every screen pixel is one *outgoing direction*

    k_f = k (cos(exit) cos(defl), cos(exit) sin(defl), sin(exit)),

so `|k_f| = |k_i| = k` holds identically, by construction rather than by tolerance. The
incident beam runs along +x and downwards,

    k_i = k (cos(incidence), 0, -sin(incidence)),

and the momentum transfer is `q = k_f - k_i`. The screen's angular coordinates are the
gnomonic projection of Liu et al.'s flat screen at distance `D`: `x_det = D tan(defl)` and
`y_det = D tan(exit) / cos(defl)`, which is why no separate detector-plane step is needed.

Surface and reciprocal lattice
------------------------------

The KMC height field is stored in axial coordinates `(r, c)`. Those are *not* the physical
positions; the physical primitive vectors of the triangular surface lattice are

    a1 = a (1, 0),        a2 = a (1/2, sqrt(3)/2),

placing the scatterer of column `(r, c)` at

    R = a (c + r/2) x + a (sqrt(3)/2) r y + d h[r, c] z,

and the reciprocal basis satisfying `a_i . b_j = 2 pi delta_ij` is

    b1 = (2 pi / a) (1, -1/sqrt(3)),   b2 = (2 pi / a) (0, 2/sqrt(3)),

both of length `4 pi / (a sqrt(3))`. `reciprocal_basis` returns them and `rod_orders`
reports which `(h, k)` rods actually cut the Ewald sphere at a given beam condition; the
detector shows however many of those fall inside its acceptance, and none are placed by
hand. A first-order rod exists only when `|G| <= k sin(incidence)`, so at shallow grazing
angles the only accessible order is the specular `(00)` one and the side orders appear,
below the specular beam and on the zeroth Laue circle, once the angle is raised.

`azimuth_deg` rotates the sample about its surface normal, which rotates the whole
reciprocal lattice against a fixed beam.

Scattering model
----------------

Each occupied column contributes one **effective scatterer** at the top of its stack. That
matches what the KMC actually simulates -- a generic growth unit, not a resolved Ga/N pair
-- and it is the default. `basis` optionally attaches a unit-cell basis to every column,

    F(q) = sum_m f_m exp(-i q . r_m),      I_hk ~ |F(G_hk)|^2,

so different orders can differ in brightness and systematic absences can appear. No GaN
surface reconstruction is supplied: the KMC is an effective-adatom model and inventing an
atomistic basis for it would be fiction. The screen intensity is

    I(q) = |F(q)|^2 |sum_j W_j exp(-i q . R_j)|^2 / (|F(q_spec)| sum_j W_j)^2,

normalized so a perfectly flat surface reads 1.0 at the specular beam.

Instrument response
-------------------

Three physically distinct broadenings, each applied in its own domain, never merged into
one "transfer width":

1. `coherence_length_nm` -- the FWHM of the Gaussian illumination `W` in **real space**.
   Its transform is the reciprocal-space rod width, `4 sqrt(2) ln2 / L`, so a longer
   coherence length gives narrower rods. The patch is exactly as wide as the beam sees
   whatever the lattice size, so rod width is an instrument property and a 7x7 run and a
   64x64 run of the same physical surface give the same streaks.
2. `divergence_deg` -- FWHM of the **incident** beam's angular spread. Handled where it
   belongs, on `k_i`: the screen is recomputed for a Gauss-Hermite set of incident
   directions and the intensities added.
3. `detector_psf_deg` -- FWHM of the **detector** point spread, a Gaussian blur applied to
   the finished screen in detector angle.

`coherence_patches` repeats the calculation at evenly spaced positions on the periodic
surface and averages intensities, because a real footprint spans many coherence areas and
the detector adds them incoherently.

What this does and does not include
-----------------------------------

- Included: surface roughness, island size and shape, step edges, the layer-phase
  interference that produces RHEED oscillations, the zeroth-Laue-zone geometry, the shadow
  edge, coherence-limited rod width, beam divergence, detector blur, an optional structure
  factor, and the incoherent sum over illuminated patches.
- Not included: multiple/dynamical scattering, refraction at the surface potential,
  inelastic and thermal-diffuse background, Kikuchi lines, absorption, surface
  reconstruction, and Debye-Waller attenuation. Real RHEED is strongly dynamical; absolute
  intensities are not comparable to a measured screen. Relative behaviour over a growth run
  is what this supports.
- A consequence of the periodic model surface: disorder can only scatter into multiples of
  `2 pi / (N a)`, so once the coherence length approaches the box width the diffuse
  background breaks into discrete satellites. `satellite_artifact_ratio` reports how close a
  given run is to that regime; translating the illuminated patch cannot cure it, because the
  satellites sit at positions fixed by the box period.

The phase order `q_z d / pi` at the specular point decides whether oscillations appear at
all: odd order is the anti-phase condition where adjacent terraces cancel, even order is
in-phase where filling a layer changes nothing. `d` is one KMC monolayer of height, taken as
the GaN c/2 bilayer spacing 0.2593 nm. Because the heights are integers, only the order's
parity matters to the specular intensity -- orders 1, 3, 5 give identical traces and differ
only in geometry. `antiphase_grazing_angle_deg` returns the angle that sets a chosen odd
order, which is what an experimenter tunes for.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from mbe_rheed_sim.lattice import HeightField

# Standard wurtzite GaN geometry. These are literature lattice parameters used to place
# scatterers; the primary paper reports no beam geometry, so nothing here is taken from it.
GAN_IN_PLANE_SPACING_NM = 0.3189
GAN_LAYER_HEIGHT_NM = 0.2593
# A common RHEED operating energy. Only the wavelength it implies is used.
DEFAULT_BEAM_ENERGY_KEV = 15.0
DEFAULT_PHASE_ORDER = 3
# Coherence lengths of real RHEED instruments are usually quoted between 10 and 100 nm. This
# is deliberately at the low end: it keeps the illuminated patch small enough to stay
# interactive and the rods wide enough to survive a screen of a few hundred pixels.
DEFAULT_COHERENCE_LENGTH_NM = 4.0
DEFAULT_COHERENCE_PATCHES = 3
DEFAULT_MAX_ORDER = 4
# Reciprocal space is sampled on a zero-padded transform grid and read back by bilinear
# interpolation. With the patch truncated at `_PATCH_CUTOFF_SIGMA` the rod is about
# `TRANSFORM_PADDING * _PATCH_CUTOFF_SIGMA * 2 / (2 pi)` bins wide, which keeps the
# interpolation error near 1e-3 of peak. Nearest-bin lookup instead of interpolation
# quantizes a rod only a few bins wide and was worth tens of percent.
TRANSFORM_PADDING = 8
# The illuminated patch is truncated at this many sigma, on a disk rather than on the
# array's sheared rhombus, so the support shares the lattice's mirror symmetry exactly.
_PATCH_CUTOFF_SIGMA = 4.0
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
# The incident angular spread is integrated on a Gaussian-weighted uniform grid out to
# `_DIVERGENCE_REACH` sigma. Gauss-Hermite would need far fewer nodes for a smooth
# integrand, but here the integrand is a rod narrower than the spread, so what matters is
# resolving the convolution, not the moments: too coarse a grid returns the rod repeated at
# each node instead of one broadened rod.
_DIVERGENCE_REACH = 2.5
# ponytail: hard node ceiling per axis, so a large divergence degrades gracefully instead of
# hanging. Raise it if a run needs divergence much wider than the rod.
_MAX_DIVERGENCE_NODES = 9

#: One entry of a unit-cell basis: position `(x, y, z)` in nm and scattering factor.
Scatterer = tuple[float, float, float, complex]


class RodOrder(NamedTuple):
    """One `(h, k)` reciprocal rod that intersects the Ewald sphere.

    `exit_angle_deg` and `deflection_deg` are the exact Ewald-construction direction of the
    diffracted beam, in the same angular coordinates the screen uses.
    """

    h: int
    k: int
    exit_angle_deg: float
    deflection_deg: float
    relative_intensity: float

    @property
    def label(self) -> str:
        if 0 <= self.h < 10 and 0 <= self.k < 10:
            return f"({self.h}{self.k})"
        return f"({self.h},{self.k})"


@dataclass(frozen=True, slots=True)
class ScreenPattern:
    """One detector image. `intensity` is 1.0 for a flat surface at the same condition."""

    intensity: NDArray[np.float64]
    exit_angle_deg: NDArray[np.float64]
    deflection_deg: NDArray[np.float64]
    grazing_angle_deg: float
    azimuth_deg: float
    beam_energy_kev: float
    coherence_length_nm: float
    divergence_deg: float
    detector_psf_deg: float
    lattice_size: int
    specular_intensity: float
    phase_order: float
    streak_width_deg: float
    rods: tuple[RodOrder, ...]
    satellite_artifact_ratio: float

    @property
    def condition(self) -> str:
        """Whether adjacent terraces cancel (`anti-phase`), add (`in-phase`), or neither."""
        distance_from_odd = abs(self.phase_order % 2.0 - 1.0)
        if distance_from_odd < 0.25:
            return "anti-phase"
        if distance_from_odd > 0.75:
            return "in-phase"
        return "intermediate"


def electron_wavelength_nm(energy_kev: float = DEFAULT_BEAM_ENERGY_KEV) -> float:
    """Relativistically corrected de Broglie wavelength of an accelerated electron."""
    if energy_kev <= 0:
        raise ValueError("beam energy must be positive")
    volts = energy_kev * 1e3
    return 1.226426 / math.sqrt(volts * (1.0 + 0.9784e-6 * volts))


def wavenumber_per_nm(energy_kev: float = DEFAULT_BEAM_ENERGY_KEV) -> float:
    """`k = 2 pi / lambda`, in nm^-1."""
    return 2.0 * math.pi / electron_wavelength_nm(energy_kev)


def surface_basis(
    in_plane_spacing_nm: float = GAN_IN_PLANE_SPACING_NM,
) -> NDArray[np.float64]:
    """Cartesian primitive vectors of the triangular surface lattice, as rows, in nm."""
    if in_plane_spacing_nm <= 0:
        raise ValueError("in-plane spacing must be positive")
    return in_plane_spacing_nm * np.array([[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])


def reciprocal_basis(
    in_plane_spacing_nm: float = GAN_IN_PLANE_SPACING_NM,
) -> NDArray[np.float64]:
    """Reciprocal primitive vectors as rows, in nm^-1, with `a_i . b_j = 2 pi delta_ij`."""
    return 2.0 * math.pi * np.linalg.inv(surface_basis(in_plane_spacing_nm)).T


def incident_wavevector(
    grazing_angle_deg: float, energy_kev: float = DEFAULT_BEAM_ENERGY_KEV
) -> NDArray[np.float64]:
    """`k_i` in nm^-1: along +x, tilted `grazing_angle_deg` below the surface plane."""
    if not 0 < grazing_angle_deg < 90:
        raise ValueError("grazing angle must lie in (0, 90) degrees")
    incidence = math.radians(grazing_angle_deg)
    wavenumber = wavenumber_per_nm(energy_kev)
    return wavenumber * np.array(
        [math.cos(incidence), 0.0, -math.sin(incidence)], dtype=np.float64
    )


@dataclass(frozen=True, slots=True)
class BeamGeometry:
    """Lab-frame directions for one beam condition and sample orientation.

    The lab frame is the one the whole module already uses, stated once here because the 3D
    geometry view and the detector screen have to agree on it:

    * `+x` is downstream along the beam's in-plane projection, `+z` is the surface normal,
      `+y` completes the right-handed set (a positive screen deflection).
    * The **incident beam is fixed** in this frame: `k_i / k = (cos t, 0, -sin t)`.
    * The **specular beam** is `k_i` mirrored in the surface: `(cos t, 0, +sin t)`. It moves
      with the grazing angle and **not** with the azimuth, because rotating a sample about its
      own normal cannot move the mirror direction.
    * The **detector plane is fixed** in this frame too, perpendicular to `+x` downstream, and
      the screen's angular coordinates are its gnomonic projection (`detector_offsets`).
    * `sample_rotation` is what the azimuth does: it turns the **sample** about `+z`, carrying
      its reciprocal lattice with it, which is why the reachable rods change while the beam,
      the specular direction and the screen do not.
    """

    grazing_angle_deg: float
    azimuth_deg: float
    energy_kev: float
    incident_direction: NDArray[np.float64]
    specular_direction: NDArray[np.float64]
    surface_normal: NDArray[np.float64]
    sample_rotation: NDArray[np.float64]


def beam_geometry(
    *,
    grazing_angle_deg: float,
    azimuth_deg: float = 0.0,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
) -> BeamGeometry:
    """Unit directions and the sample rotation for one beam condition. See `BeamGeometry`."""
    incident = incident_wavevector(grazing_angle_deg, energy_kev)
    direction = incident / float(np.linalg.norm(incident))
    return BeamGeometry(
        grazing_angle_deg=grazing_angle_deg,
        azimuth_deg=azimuth_deg,
        energy_kev=energy_kev,
        incident_direction=direction,
        # The mirror direction is the zero-deflection outgoing direction at the same angle, so
        # it comes from the same helper the rods do rather than from a second convention here.
        specular_direction=outgoing_direction(grazing_angle_deg, 0.0),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        sample_rotation=_rotation(azimuth_deg),
    )


def outgoing_direction(
    exit_angle_deg: float, deflection_deg: float
) -> NDArray[np.float64]:
    """Unit `k_f` for one screen direction, in the lab frame of `BeamGeometry`.

    The same parameterization `diffraction_screen` sweeps over its pixels, so every direction
    it returns is elastic by construction: `|k_f| = |k_i|` holds because only the direction is
    built here, never the magnitude.
    """
    exit_angle, deflection = math.radians(exit_angle_deg), math.radians(deflection_deg)
    return np.array(
        [
            math.cos(exit_angle) * math.cos(deflection),
            math.cos(exit_angle) * math.sin(deflection),
            math.sin(exit_angle),
        ]
    )


def detector_intersection(
    direction: NDArray[np.float64], distance: float
) -> tuple[float, float]:
    """Where a ray along `direction` from the sample crosses the detector plane.

    The plane sits perpendicular to `+x` at `distance` downstream, so the ray reaches it after
    `distance / direction_x` and lands at `(horizontal, vertical)` in the plane's own axes.
    Equivalent to `detector_offsets` on the same direction's angles -- a test pins that -- so a
    drawn ray and a painted screen pixel cannot disagree about where an order lands.
    """
    if distance <= 0:
        raise ValueError("detector distance must be positive")
    if direction[0] <= 0:
        raise ValueError("only downstream rays reach the detector plane")
    reach = distance / float(direction[0])
    return float(direction[1]) * reach, float(direction[2]) * reach


def detector_offsets(
    exit_angle_deg: NDArray[np.float64] | float,
    deflection_deg: NDArray[np.float64] | float,
    distance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Where a screen direction lands on a flat detector at `distance`, in the same units.

    The gnomonic projection this module's angular coordinates already are: horizontal
    `d tan(deflection)` across the plane, vertical `d tan(exit) / cos(deflection)` up it,
    measured from where the beam axis pierces the plane. The plane is perpendicular to `+x`,
    so horizontal is lab `+y` and vertical is lab `+z`.
    """
    if distance <= 0:
        raise ValueError("detector distance must be positive")
    exit_angle = np.radians(exit_angle_deg)
    deflection = np.radians(deflection_deg)
    return (
        distance * np.tan(deflection),
        distance * np.tan(exit_angle) / np.cos(deflection),
    )


def phase_order(
    grazing_angle_deg: float,
    *,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
    layer_height_nm: float = GAN_LAYER_HEIGHT_NM,
) -> float:
    """Specular `q_z d / pi`. Odd is anti-phase, even is in-phase.

    On the specular beam `q = (0, 0, 2 k sin(incidence))`, so `q_z d / pi` is
    `4 d sin(incidence) / lambda`: adjacent terraces, one `d` apart, then differ in phase by
    `pi` times this and cancel whenever it is odd.
    """
    if not 0 < grazing_angle_deg < 90:
        raise ValueError("grazing angle must lie in (0, 90) degrees")
    wavelength = electron_wavelength_nm(energy_kev)
    return 4.0 * layer_height_nm * math.sin(math.radians(grazing_angle_deg)) / wavelength


def antiphase_grazing_angle_deg(
    order: int = DEFAULT_PHASE_ORDER,
    *,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
    layer_height_nm: float = GAN_LAYER_HEIGHT_NM,
) -> float:
    """Grazing angle whose specular phase order is `order`; odd orders are anti-phase."""
    if order < 1:
        raise ValueError("phase order must be a positive integer")
    sine = order * electron_wavelength_nm(energy_kev) / (4.0 * layer_height_nm)
    if sine >= 1.0:
        raise ValueError("no grazing angle reaches that order at this beam energy")
    return math.degrees(math.asin(sine))


def _rotation(azimuth_deg: float) -> NDArray[np.float64]:
    """In-plane rotation by the sample azimuth, about the surface normal."""
    angle = math.radians(azimuth_deg)
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )


def structure_factor(
    q_x: NDArray[np.float64] | float,
    q_y: NDArray[np.float64] | float,
    q_z: NDArray[np.float64] | float,
    basis: tuple[Scatterer, ...] | None,
) -> NDArray[np.complex128] | complex:
    """`F(q) = sum_m f_m exp(-i q . r_m)` over one unit cell, in sample coordinates.

    `None` is the effective single scatterer the KMC actually models, for which `F = 1`.
    """
    if not basis:
        return 1.0 + 0.0j
    total = np.zeros(np.broadcast(q_x, q_y, q_z).shape, dtype=np.complex128)
    for x, y, z, factor in basis:
        total += factor * np.exp(-1j * (q_x * x + q_y * y + q_z * z))
    return total


def rod_orders(
    *,
    grazing_angle_deg: float,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
    in_plane_spacing_nm: float = GAN_IN_PLANE_SPACING_NM,
    azimuth_deg: float = 0.0,
    basis: tuple[Scatterer, ...] | None = None,
    max_order: int = DEFAULT_MAX_ORDER,
    span_deg: float | None = None,
) -> tuple[RodOrder, ...]:
    """Exact Ewald intersections of the `(h, k)` rods, in screen angular coordinates.

    A rod is reachable only when its in-plane momentum fits inside the Ewald sphere,
    `|k_i,par + G| <= k`; unreachable orders are simply absent from the result rather than
    projected onto the screen. `span_deg`, if given, keeps only orders inside a screen of
    that half-width centred on the specular beam.
    """
    incident = incident_wavevector(grazing_angle_deg, energy_kev)
    wavenumber = float(np.linalg.norm(incident))
    rotation = _rotation(azimuth_deg)
    basis_vectors = reciprocal_basis(in_plane_spacing_nm) @ rotation.T

    orders: list[RodOrder] = []
    for h in range(-max_order, max_order + 1):
        for k in range(-max_order, max_order + 1):
            reciprocal = h * basis_vectors[0] + k * basis_vectors[1]
            parallel = incident[:2] + reciprocal
            vertical_squared = wavenumber**2 - float(parallel @ parallel)
            if vertical_squared <= 0.0:
                continue
            vertical = math.sqrt(vertical_squared)
            exit_angle = math.degrees(math.asin(vertical / wavenumber))
            deflection = math.degrees(math.atan2(parallel[1], parallel[0]))
            if span_deg is not None and (
                abs(deflection) > span_deg
                or abs(exit_angle - grazing_angle_deg) > span_deg
            ):
                continue
            # The rod is expressed in the sample frame, where the basis positions live.
            sample_q = rotation.T @ reciprocal
            momentum_z = vertical + abs(incident[2])
            amplitude = structure_factor(sample_q[0], sample_q[1], momentum_z, basis)
            reference = structure_factor(0.0, 0.0, 2.0 * abs(incident[2]), basis)
            orders.append(
                RodOrder(
                    h,
                    k,
                    exit_angle,
                    deflection,
                    float(abs(amplitude) ** 2 / abs(reference) ** 2),
                )
            )
    return tuple(orders)


def _illuminated_patches(
    lattice: HeightField,
    coherence_length_nm: float,
    in_plane_spacing_nm: float,
    patches: int,
) -> tuple[list[NDArray[np.int64]], NDArray[np.float64]]:
    """Height patches under the beam, and the Gaussian illumination profile they share.

    The patch is exactly as wide as the beam sees, whatever the lattice size: a lattice
    smaller than the illuminated area is tiled, a larger one is sampled. Patch origins are
    spread over the periodic surface so their intensities average to a detector-like image.

    The profile is truncated on a **disk** centred on a lattice site, not on the array's
    sheared rhombus. A rhombic cut-off does not commute with the lattice's mirror plane and
    left a percent-level left/right asymmetry on a screen that must be symmetric.
    """
    if coherence_length_nm <= 0 or patches < 1:
        raise ValueError("coherence length must be positive and patches at least one")
    sigma_nm = coherence_length_nm / _FWHM_PER_SIGMA
    sigma_sites = sigma_nm / in_plane_spacing_nm
    # Half-width chosen so the inscribed disk of the rhombic patch still reaches the
    # cut-off radius: the rhombus inradius is `half * a * sin(60 deg)`.
    half = math.ceil(_PATCH_CUTOFF_SIGMA * sigma_sites / (math.sqrt(3.0) / 2.0))
    extent = 2 * half + 1

    size = len(lattice)
    origins = np.unique(np.linspace(0, size, patches, endpoint=False).astype(int))
    span = np.arange(extent)
    windows = [
        lattice[np.ix_((span + oy) % size, (span + ox) % size)]
        for oy in origins
        for ox in origins
    ]

    row, column = np.indices((extent, extent))
    # True Cartesian offsets from the centre lattice site: the axial lattice is not square,
    # so weighting on array indices would make the illuminated spot a rhombus.
    x = in_plane_spacing_nm * (column - half + 0.5 * (row - half))
    y = in_plane_spacing_nm * (math.sqrt(3.0) / 2.0) * (row - half)
    radius_squared = x**2 + y**2
    profile = np.where(
        radius_squared <= (_PATCH_CUTOFF_SIGMA * sigma_nm) ** 2,
        np.exp(-radius_squared / (2.0 * sigma_nm**2)),
        0.0,
    )
    return windows, profile


def _centred_transform(weights: NDArray[np.float64], size: int) -> NDArray[np.complex128]:
    """Zero-padded transform of `weights` about its centre site, so no phase ramp remains.

    Leaving the ramp in makes the sampled transform oscillate every few bins, which
    interpolation cannot follow; removing it leaves a smooth envelope. The dropped factor
    `exp(-i m (f_row + f_col))` is common to every height level, so it cancels in `|A|^2`.
    """
    padded = np.zeros((size, size))
    padded[: weights.shape[0], : weights.shape[1]] = weights
    centre = weights.shape[0] // 2
    return np.fft.fft2(np.roll(padded, (-centre, -centre), axis=(0, 1)))


def _sample_transform(
    transform: NDArray[np.complex128],
    row_bin: NDArray[np.float64],
    column_bin: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Bilinear read-back of a `2 pi`-periodic transform at real-valued bin coordinates."""
    size = transform.shape[0]
    row_low = np.floor(row_bin).astype(np.int64)
    column_low = np.floor(column_bin).astype(np.int64)
    row_frac = row_bin - row_low
    column_frac = column_bin - column_low
    row_low %= size
    column_low %= size
    row_high = (row_low + 1) % size
    column_high = (column_low + 1) % size
    return (
        (1.0 - row_frac) * (1.0 - column_frac) * transform[row_low, column_low]
        + (1.0 - row_frac) * column_frac * transform[row_low, column_high]
        + row_frac * (1.0 - column_frac) * transform[row_high, column_low]
        + row_frac * column_frac * transform[row_high, column_high]
    )


def _blur_axis(image: NDArray[np.float64], axis_deg: NDArray[np.float64], fwhm_deg: float):
    """Gaussian smoothing along the first axis, in the axis's own angular units."""
    sigma = fwhm_deg / _FWHM_PER_SIGMA
    separation = axis_deg[:, None] - axis_deg[None, :]
    kernel = np.exp(-0.5 * (separation / sigma) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)
    return kernel @ image


def _divergence_nodes(
    divergence_deg: float, resolution_deg: float
) -> list[tuple[float, float, float]]:
    """Incident-direction offsets and weights for a Gaussian angular spread, in degrees.

    `resolution_deg` is the narrowest detector feature the grid has to resolve; the node
    spacing is kept at or below it, up to `_MAX_DIVERGENCE_NODES` per axis.
    """
    if divergence_deg < 0:
        raise ValueError("beam divergence cannot be negative")
    if divergence_deg == 0:
        return [(0.0, 0.0, 1.0)]
    sigma = divergence_deg / _FWHM_PER_SIGMA
    wanted = 2 * math.ceil(_DIVERGENCE_REACH * sigma / max(resolution_deg, 1e-9)) + 1
    count = int(min(_MAX_DIVERGENCE_NODES, max(3, wanted)))
    offsets = np.linspace(-_DIVERGENCE_REACH * sigma, _DIVERGENCE_REACH * sigma, count)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    return [
        (float(polar), float(azimuthal), float(polar_weight * azimuthal_weight))
        for polar, polar_weight in zip(offsets, weights, strict=True)
        for azimuthal, azimuthal_weight in zip(offsets, weights, strict=True)
    ]


def specular_intensity(
    heights: HeightField,
    *,
    grazing_angle_deg: float,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
    in_plane_spacing_nm: float = GAN_IN_PLANE_SPACING_NM,
    layer_height_nm: float = GAN_LAYER_HEIGHT_NM,
    coherence_length_nm: float = DEFAULT_COHERENCE_LENGTH_NM,
    coherence_patches: int = DEFAULT_COHERENCE_PATCHES,
) -> NDArray[np.float64]:
    """Kinematic (00) intensity for one lattice or a stack of them.

    The in-plane phases all cancel on the specular rod, so only the height distribution
    survives and no transform is needed. Returns a scalar array for a single lattice and one
    value per snapshot for a stack; either way it equals the centre pixel of the matching
    `diffraction_screen` at zero divergence and zero detector blur. The structure factor
    cancels against its own specular value here, so it takes no `basis`.
    """
    stack = np.asarray(heights)
    if stack.ndim not in (2, 3) or min(stack.shape[-2:]) < 2:
        raise ValueError("heights must be one 2D lattice or a stack of them")
    layer_phase = math.pi * phase_order(
        grazing_angle_deg, energy_kev=energy_kev, layer_height_nm=layer_height_nm
    )
    single = stack[None] if stack.ndim == 2 else stack
    intensities = np.empty(len(single))
    for index, lattice in enumerate(single):
        windows, profile = _illuminated_patches(
            lattice, coherence_length_nm, in_plane_spacing_nm, coherence_patches
        )
        amplitudes = [
            np.sum(profile * np.exp(-1j * layer_phase * window)) / profile.sum()
            for window in windows
        ]
        intensities[index] = np.mean(np.abs(amplitudes) ** 2)
    return intensities[0] if stack.ndim == 2 else intensities


# Decades of intensity shown below the flat-surface specular value. A real screen is viewed
# well short of this range; three keeps the rods bright and the background near black while
# still showing the diffuse scattering that roughening produces.
SCREEN_LOG_DECADES = 3.0


def half_max_width(axis: NDArray[np.float64], values: NDArray[np.float64]) -> float:
    """Full width at half maximum, as the extent of the samples above half the peak."""
    above = axis[values > 0.5 * values.max()]
    return float(above.max() - above.min())


def specular_row(pattern: "ScreenPattern") -> NDArray[np.float64]:
    """The screen row through the specular beam. Row counts are odd, so this is the centre."""
    return pattern.intensity[pattern.intensity.shape[0] // 2]


def measured_rod_fwhm_deg(pattern: "ScreenPattern") -> float:
    """Rod width measured off the screen, to compare against the analytic `streak_width_deg`.

    Resolve the screen finely enough before trusting this: a rod a few pixels wide measures its
    own pixel grid. `scripts/validate_rheed.py` is what asserts the two agree.
    """
    return half_max_width(pattern.deflection_deg, specular_row(pattern))


def screen_decades(
    pattern: "ScreenPattern", decades: float = SCREEN_LOG_DECADES
) -> NDArray[np.float64]:
    """Screen intensity in decades relative to a flat surface, floored so log10 stays finite.

    `intensity` is normalized to 1.0 for a flat surface at the same beam condition and never
    exceeds it, so this runs from `-decades` up to 0. Displaying it on that fixed range is what
    keeps two screens comparable: a run whose specular has collapsed stays dark instead of being
    renormalized back to full brightness.
    """
    if decades <= 0:
        raise ValueError("decades must be positive")
    return np.log10(np.maximum(pattern.intensity, 10.0**-decades))


def satellite_artifact_ratio(
    lattice_size: int,
    *,
    in_plane_spacing_nm: float = GAN_IN_PLANE_SPACING_NM,
    coherence_length_nm: float = DEFAULT_COHERENCE_LENGTH_NM,
) -> float:
    """How discrete the diffuse background is: satellite spacing over rod FWHM.

    Disorder on a box of `N` sites can only scatter into multiples of `2 pi / (N a)`. Those
    satellites merge into a continuum while they sit inside one rod width; the ratio here
    exceeding roughly 1 is the point at which a small box starts showing them as separate
    features. Translating the illuminated patch does not help -- the satellite positions are
    fixed by the box period, not by where the beam lands -- so the cure is a bigger box or a
    shorter coherence length, and this number is what says which.
    """
    satellite_spacing = 2.0 * math.pi / (lattice_size * in_plane_spacing_nm)
    rod_width = 4.0 * math.sqrt(2.0) * math.log(2.0) / coherence_length_nm
    return satellite_spacing / rod_width


def diffraction_screen(
    heights: HeightField,
    *,
    grazing_angle_deg: float,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
    in_plane_spacing_nm: float = GAN_IN_PLANE_SPACING_NM,
    layer_height_nm: float = GAN_LAYER_HEIGHT_NM,
    coherence_length_nm: float = DEFAULT_COHERENCE_LENGTH_NM,
    coherence_patches: int = DEFAULT_COHERENCE_PATCHES,
    azimuth_deg: float = 0.0,
    divergence_deg: float = 0.0,
    detector_psf_deg: float = 0.0,
    basis: tuple[Scatterer, ...] | None = None,
    max_order: int = DEFAULT_MAX_ORDER,
    span_deg: float = 3.0,
    shape: tuple[int, int] = (181, 241),
) -> ScreenPattern:
    """Kinematic detector image over a square angular window centred on the specular beam.

    Rows are the exit angle above the surface plane, columns the horizontal deflection, both
    in degrees; the specular beam sits at the centre pixel. Anything below the horizon is the
    shadow of the substrate and is returned as zero.

    The sum over columns is grouped by height level, so the transform cost follows the number
    of occupied levels rather than the number of screen pixels: each level's footprint is
    transformed once, then the levels are recombined per pixel with that pixel's own
    out-of-plane phase.
    """
    lattice = np.asarray(heights)
    if lattice.ndim != 2 or min(lattice.shape) < 2:
        raise ValueError("heights must be a 2D lattice")
    if span_deg <= 0 or min(shape) < 3:
        raise ValueError("the screen needs a positive angular span and at least 3x3 pixels")
    if detector_psf_deg < 0:
        raise ValueError("detector point spread cannot be negative")

    if not 0 < grazing_angle_deg < 90:
        raise ValueError("grazing angle must lie in (0, 90) degrees")
    wavenumber = wavenumber_per_nm(energy_kev)
    incidence = math.radians(grazing_angle_deg)
    # Odd pixel counts put the specular beam exactly on the centre pixel rather than between.
    rows, columns = (side | 1 for side in shape)
    exit_angle = incidence + np.radians(np.linspace(-span_deg, span_deg, rows))[:, None]
    deflection = np.radians(np.linspace(-span_deg, span_deg, columns))[None, :]

    # Outgoing wavevector. |k_f| = k identically, so every pixel sits on the Ewald sphere.
    final_x = wavenumber * np.cos(exit_angle) * np.cos(deflection)
    final_y = wavenumber * np.cos(exit_angle) * np.sin(deflection)
    final_z = wavenumber * np.sin(exit_angle)

    windows, profile = _illuminated_patches(
        lattice, coherence_length_nm, in_plane_spacing_nm, coherence_patches
    )
    transform_size = TRANSFORM_PADDING * len(profile)
    rotation = _rotation(azimuth_deg)

    # A Gaussian illumination of FWHM L transforms to an intensity rod of reciprocal FWHM
    # 4 sqrt(2) ln2 / L; near the specular beam a q_y offset is a deflection of q_y/(k cos).
    streak_width = (
        4.0 * math.sqrt(2.0) * math.log(2.0) / coherence_length_nm
    ) / (wavenumber * math.cos(incidence))
    nodes = _divergence_nodes(divergence_deg, 0.5 * math.degrees(streak_width))
    scale = transform_size / (2.0 * math.pi)

    intensity = np.zeros((rows, columns))
    for window in windows:
        levels = list(range(int(window.min()), int(window.max()) + 1))
        footprints = [
            _centred_transform(np.where(window == level, profile, 0.0), transform_size)
            for level in levels
        ]
        for polar_offset, azimuthal_offset, weight in nodes:
            tilt = incidence + math.radians(polar_offset)
            swing = math.radians(azimuthal_offset)
            q_x = final_x - wavenumber * math.cos(tilt) * math.cos(swing)
            q_y = final_y - wavenumber * math.cos(tilt) * math.sin(swing)
            q_z = np.broadcast_to(final_z + wavenumber * math.sin(tilt), (rows, columns))
            # Rotating the sample by the azimuth is rotating q the other way in sample axes.
            sample_x = rotation[0, 0] * q_x + rotation[1, 0] * q_y
            sample_y = rotation[0, 1] * q_x + rotation[1, 1] * q_y
            # Axial site positions are linear in the array indices, so the in-plane sum is a
            # plain 2D transform once q is expressed in the index basis.
            column_bin = np.broadcast_to(in_plane_spacing_nm * sample_x * scale, (rows, columns))
            row_bin = np.broadcast_to(
                in_plane_spacing_nm * (sample_x + math.sqrt(3.0) * sample_y) / 2.0 * scale,
                (rows, columns),
            )
            amplitude = np.zeros((rows, columns), dtype=np.complex128)
            for level, footprint in zip(levels, footprints, strict=True):
                amplitude += _sample_transform(footprint, row_bin, column_bin) * np.exp(
                    -1j * layer_height_nm * q_z * level
                )
            form = structure_factor(sample_x, sample_y, q_z, basis) / structure_factor(
                0.0, 0.0, 2.0 * wavenumber * math.sin(tilt), basis
            )
            intensity += weight * np.abs(amplitude * form / profile.sum()) ** 2
    intensity /= len(windows)

    exit_grid = np.broadcast_to(exit_angle, intensity.shape)
    intensity[exit_grid < 0.0] = 0.0
    if detector_psf_deg > 0:
        exit_axis = np.degrees(exit_angle).ravel()
        deflection_axis = np.degrees(deflection).ravel()
        intensity = _blur_axis(intensity, exit_axis, detector_psf_deg)
        intensity = _blur_axis(intensity.T, deflection_axis, detector_psf_deg).T
        intensity[exit_grid < 0.0] = 0.0

    return ScreenPattern(
        intensity=intensity,
        exit_angle_deg=np.degrees(exit_angle).ravel(),
        deflection_deg=np.degrees(deflection).ravel(),
        grazing_angle_deg=grazing_angle_deg,
        azimuth_deg=azimuth_deg,
        beam_energy_kev=energy_kev,
        coherence_length_nm=coherence_length_nm,
        divergence_deg=divergence_deg,
        detector_psf_deg=detector_psf_deg,
        lattice_size=len(lattice),
        specular_intensity=float(intensity[rows // 2, columns // 2]),
        phase_order=phase_order(
            grazing_angle_deg, energy_kev=energy_kev, layer_height_nm=layer_height_nm
        ),
        streak_width_deg=math.degrees(streak_width),
        rods=rod_orders(
            grazing_angle_deg=grazing_angle_deg,
            energy_kev=energy_kev,
            in_plane_spacing_nm=in_plane_spacing_nm,
            azimuth_deg=azimuth_deg,
            basis=basis,
            max_order=max_order,
            span_deg=span_deg,
        ),
        satellite_artifact_ratio=satellite_artifact_ratio(
            len(lattice),
            in_plane_spacing_nm=in_plane_spacing_nm,
            coherence_length_nm=coherence_length_nm,
        ),
    )
