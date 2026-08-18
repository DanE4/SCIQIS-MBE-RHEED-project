"""Kinematic RHEED from the model surface.

Everything else in this project reports `1 - S_d`, a morphology stand-in. This module
computes the other thing: the electron intensity a detector would collect, in the
**kinematic** (single-scattering) approximation.

Each occupied column contributes one scatterer at the top of its stack,

    R = a (c + r/2) x + a (sqrt(3)/2) r y + d h[r, c] z,

for axial lattice indices (r, c), in-plane spacing `a` and monolayer height `d`. The
scattered amplitude at momentum transfer q = k_out - k_in is the plain sum

    A(q) = sum_j W_j exp(-i q . R_j),

with `W` a coherence window (below), and the reported intensity is |A|^2 normalized so a
perfectly flat surface at the same beam condition reads 1.0.

What this does and does not include:

- Included: surface roughness, island size and shape, step edges, the layer-phase
  interference that produces RHEED oscillations, Laue-zone arcs, the shadow edge, and
  finite coherence-width broadening.
- Not included: multiple/dynamical scattering, refraction at the surface potential,
  inelastic and thermal-diffuse background, absorption, surface reconstruction, atomic
  form factors, and instrumental point-spread. Real RHEED is strongly dynamical, so
  absolute intensities here are not comparable to a measured screen. Relative behaviour
  over a growth run is what this supports.

The phase order `q_z d / pi` at the specular point decides whether oscillations appear at
all: odd order is the anti-phase condition where adjacent terraces cancel, even order is
in-phase where filling a layer changes nothing. `antiphase_grazing_angle_deg` returns the
angle that sets a chosen odd order, which is what an experimenter tunes for.
"""

import math
from dataclasses import dataclass

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
# Zero-padding sets how finely reciprocal space is sampled between the exact lattice
# frequencies. Beyond this the pattern stops changing but the transform cost keeps growing.
# ponytail: fixed cap; raise it only if a large lattice visibly aliases.
MAX_TRANSFORM_SIZE = 512


@dataclass(frozen=True, slots=True)
class ScreenPattern:
    """One detector image. `intensity` is 1.0 for a flat surface at the same condition."""

    intensity: NDArray[np.float64]
    exit_angle_deg: NDArray[np.float64]
    deflection_deg: NDArray[np.float64]
    grazing_angle_deg: float
    beam_energy_kev: float
    specular_intensity: float
    phase_order: float

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


def phase_order(
    grazing_angle_deg: float,
    *,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
    layer_height_nm: float = GAN_LAYER_HEIGHT_NM,
) -> float:
    """Specular `q_z d / pi`. Odd is anti-phase, even is in-phase."""
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


def _coherence_window(size: int) -> NDArray[np.float64]:
    """Smooth illumination profile over the simulated patch, normalized to unit mean.

    A hard-edged patch would ring: its transform is a sinc whose sidelobes fill the screen
    with speckle that is an artifact of the box, not of the surface. A finite beam
    coherence width is the physical version of the same taper.
    """
    profile = np.hanning(size + 2)[1:-1]
    window = np.outer(profile, profile)
    return window / window.mean()


def specular_intensity(
    heights: HeightField,
    *,
    grazing_angle_deg: float,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
    layer_height_nm: float = GAN_LAYER_HEIGHT_NM,
) -> NDArray[np.float64]:
    """Kinematic (00) intensity for one lattice or a stack of them.

    The in-plane phases all cancel on the specular rod, so only the height distribution
    survives and no transform is needed. Returns a scalar array for a single lattice and
    one value per snapshot for a stack.
    """
    stack = np.asarray(heights)
    if stack.ndim not in (2, 3) or min(stack.shape[-2:]) < 2:
        raise ValueError("heights must be one 2D lattice or a stack of them")
    layer_phase = math.pi * phase_order(
        grazing_angle_deg, energy_kev=energy_kev, layer_height_nm=layer_height_nm
    )
    window = _coherence_window(stack.shape[-1])
    amplitude = np.mean(window * np.exp(-1j * layer_phase * stack), axis=(-2, -1))
    return np.abs(amplitude) ** 2


def diffraction_screen(
    heights: HeightField,
    *,
    grazing_angle_deg: float,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
    in_plane_spacing_nm: float = GAN_IN_PLANE_SPACING_NM,
    layer_height_nm: float = GAN_LAYER_HEIGHT_NM,
    span_deg: float = 3.0,
    shape: tuple[int, int] = (121, 161),
) -> ScreenPattern:
    """Kinematic detector image over a square angular window centred on the specular beam.

    Rows are the exit angle above the surface plane, columns the horizontal deflection, both
    in degrees; the specular beam sits at the centre pixel. Anything below the horizon is
    the shadow of the substrate and is returned as zero.

    The sum over columns is grouped by height level, so the transform cost follows the
    number of occupied levels rather than the number of screen pixels: each level's
    footprint is transformed once, then the levels are recombined per pixel with that
    pixel's own out-of-plane phase.
    """
    lattice = np.asarray(heights)
    if lattice.ndim != 2 or min(lattice.shape) < 2:
        raise ValueError("heights must be a 2D lattice")
    if span_deg <= 0 or min(shape) < 3:
        raise ValueError("the screen needs a positive angular span and at least 3x3 pixels")

    wavenumber = 2.0 * math.pi / electron_wavelength_nm(energy_kev)
    incidence = math.radians(grazing_angle_deg)
    # Odd pixel counts put the specular beam exactly on the centre pixel rather than between.
    rows, columns = (side | 1 for side in shape)
    exit_angle = incidence + np.radians(np.linspace(-span_deg, span_deg, rows))[:, None]
    deflection = np.radians(np.linspace(-span_deg, span_deg, columns))[None, :]

    q_x = wavenumber * (np.cos(exit_angle) * np.cos(deflection) - math.cos(incidence))
    q_y = wavenumber * np.cos(exit_angle) * np.sin(deflection)
    q_z = wavenumber * (np.sin(exit_angle) + math.sin(incidence))
    # Axial site positions are linear in the array indices, so the in-plane sum is a plain
    # 2D transform once the momentum transfer is expressed in the index basis.
    column_frequency = in_plane_spacing_nm * q_x
    row_frequency = in_plane_spacing_nm * (q_x + math.sqrt(3.0) * q_y) / 2.0

    size = len(lattice)
    transform_size = max(size, min(4 * size, MAX_TRANSFORM_SIZE))
    window = _coherence_window(size)
    levels = np.arange(int(lattice.min()), int(lattice.max()) + 1)
    footprints = np.stack(
        [
            np.fft.fft2(np.where(lattice == level, window, 0.0), s=(transform_size,) * 2)
            for level in levels
        ]
    )
    row_index = np.rint(row_frequency * transform_size / (2 * math.pi)).astype(int)
    column_index = np.rint(column_frequency * transform_size / (2 * math.pi)).astype(int)
    row_index, column_index = np.broadcast_arrays(
        row_index % transform_size, column_index % transform_size
    )
    out_of_plane = np.exp(-1j * layer_height_nm * q_z[None] * levels[:, None, None])
    amplitude = (footprints[:, row_index, column_index] * out_of_plane).sum(axis=0)

    intensity = np.abs(amplitude) ** 2 / lattice.size**2
    intensity[np.broadcast_to(exit_angle, intensity.shape) < 0.0] = 0.0
    return ScreenPattern(
        intensity=intensity,
        exit_angle_deg=np.degrees(exit_angle).ravel(),
        deflection_deg=np.degrees(deflection).ravel(),
        grazing_angle_deg=grazing_angle_deg,
        beam_energy_kev=energy_kev,
        specular_intensity=float(intensity[rows // 2, columns // 2]),
        phase_order=phase_order(
            grazing_angle_deg, energy_kev=energy_kev, layer_height_nm=layer_height_nm
        ),
    )
