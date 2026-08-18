"""Kinematic RHEED from the model surface.

Everything else in this project reports `1 - S_d`, a morphology stand-in. This module
computes the other thing: the electron intensity a detector would collect, in the
**kinematic** (single-scattering) approximation.

Each occupied column contributes one scatterer at the top of its stack,

    R = a (c + r/2) x + a (sqrt(3)/2) r y + d h[r, c] z,

for axial lattice indices (r, c), in-plane spacing `a` and monolayer height `d`. Every screen
pixel is one elastic outgoing direction, so its momentum transfer q = k_out - k_in sits on the
Ewald sphere by construction, and the intensity there is

    I(q) = |sum_j W_j exp(-i q . R_j)|^2 / (sum_j W_j)^2,

normalized so a perfectly flat surface reads 1.0 at the specular beam.

**What sets the streak width.** `W` is the illumination profile, a Gaussian whose full width
at half maximum is the beam's *transfer width*: the combined coherence length and instrument
response, a property of the microscope rather than of this simulation. The lattice is periodic,
so the surface it represents extends indefinitely; it is tiled or cropped to fill exactly that
illuminated patch. Diffraction features therefore have the width the stated instrument implies.
Windowing the raw simulation box instead would make a 7x7 run produce degree-wide bands and a
64x64 run produce narrow ones from identical physics, which is a statement about the box.

**Why several patches.** A real beam footprint spans many coherence patches and the detector
adds their intensities, not their amplitudes. `coherence_patches` repeats the calculation at
evenly spaced positions on the periodic surface and averages the intensities, which is what
removes single-patch speckle from the diffuse background.

What this does and does not include:

- Included: surface roughness, island size and shape, step edges, the layer-phase interference
  that produces RHEED oscillations, Laue-zone arcs, the shadow edge, transfer-width broadening,
  and the incoherent sum over illuminated patches.
- Not included: multiple/dynamical scattering, refraction at the surface potential, inelastic
  and thermal-diffuse background, absorption, surface reconstruction, atomic form factors, and
  Debye-Waller attenuation, so every rod order here is equally bright. Real RHEED is strongly
  dynamical; absolute intensities are not comparable to a measured screen. Relative behaviour
  over a growth run is what this supports.
- A consequence of the periodic model surface: disorder can only scatter into multiples of
  1/`lattice_size` of the Brillouin zone, so a small lattice shows discrete diffuse satellites
  where a real surface would show a continuum.

The phase order `q_z d / pi` at the specular point decides whether oscillations appear at all:
odd order is the anti-phase condition where adjacent terraces cancel, even order is in-phase
where filling a layer changes nothing. `antiphase_grazing_angle_deg` returns the angle that
sets a chosen odd order, which is what an experimenter tunes for.
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
# Transfer widths of real RHEED instruments are usually quoted between 10 and 100 nm. This is
# deliberately at the low end: it keeps the illuminated patch small enough to stay interactive
# and the streaks wide enough to survive a screen of a few hundred pixels.
DEFAULT_TRANSFER_WIDTH_NM = 4.0
DEFAULT_COHERENCE_PATCHES = 3
# Zero-padding sets how finely reciprocal space is sampled between the lattice frequencies.
# The illuminated patch is already small, so this stays modest.
TRANSFORM_PADDING = 2
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


@dataclass(frozen=True, slots=True)
class ScreenPattern:
    """One detector image. `intensity` is 1.0 for a flat surface at the same condition."""

    intensity: NDArray[np.float64]
    exit_angle_deg: NDArray[np.float64]
    deflection_deg: NDArray[np.float64]
    grazing_angle_deg: float
    beam_energy_kev: float
    transfer_width_nm: float
    lattice_size: int
    specular_intensity: float
    phase_order: float
    streak_width_deg: float
    rod_spacing_deg: float

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


def _illuminated_patches(
    lattice: HeightField,
    transfer_width_nm: float,
    in_plane_spacing_nm: float,
    patches: int,
) -> tuple[list[NDArray[np.int64]], NDArray[np.float64]]:
    """Height patches under the beam, and the Gaussian illumination profile they share.

    The patch is exactly as wide as the beam sees, whatever the lattice size: a lattice
    smaller than the illuminated area is tiled, a larger one is sampled. Patch origins are
    spread over the periodic surface so their intensities average to a detector-like image.
    """
    if transfer_width_nm <= 0 or patches < 1:
        raise ValueError("transfer width must be positive and patches at least one")
    sigma_sites = transfer_width_nm / _FWHM_PER_SIGMA / in_plane_spacing_nm
    extent = 2 * math.ceil(3.0 * sigma_sites) + 1

    size = len(lattice)
    origins = np.unique(np.linspace(0, size, patches, endpoint=False).astype(int))
    span = np.arange(extent)
    windows = [
        lattice[np.ix_((span + oy) % size, (span + ox) % size)]
        for oy in origins
        for ox in origins
    ]

    row, column = np.indices((extent, extent))
    # True Cartesian offsets from the patch centre: the axial lattice is not square, so
    # weighting on array indices would make the illuminated spot a rhombus.
    x = in_plane_spacing_nm * (column + 0.5 * row)
    y = in_plane_spacing_nm * (math.sqrt(3.0) / 2.0) * row
    radius_squared = (x - x.mean()) ** 2 + (y - y.mean()) ** 2
    profile = np.exp(-radius_squared / (2.0 * (sigma_sites * in_plane_spacing_nm) ** 2))
    return windows, profile


def specular_intensity(
    heights: HeightField,
    *,
    grazing_angle_deg: float,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
    in_plane_spacing_nm: float = GAN_IN_PLANE_SPACING_NM,
    layer_height_nm: float = GAN_LAYER_HEIGHT_NM,
    transfer_width_nm: float = DEFAULT_TRANSFER_WIDTH_NM,
    coherence_patches: int = DEFAULT_COHERENCE_PATCHES,
) -> NDArray[np.float64]:
    """Kinematic (00) intensity for one lattice or a stack of them.

    The in-plane phases all cancel on the specular rod, so only the height distribution
    survives and no transform is needed. Returns a scalar array for a single lattice and one
    value per snapshot for a stack; either way it equals the centre pixel of the matching
    `diffraction_screen`.
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
            lattice, transfer_width_nm, in_plane_spacing_nm, coherence_patches
        )
        amplitudes = [
            np.sum(profile * np.exp(-1j * layer_phase * window)) / profile.sum()
            for window in windows
        ]
        intensities[index] = np.mean(np.abs(amplitudes) ** 2)
    return intensities[0] if stack.ndim == 2 else intensities


def diffraction_screen(
    heights: HeightField,
    *,
    grazing_angle_deg: float,
    energy_kev: float = DEFAULT_BEAM_ENERGY_KEV,
    in_plane_spacing_nm: float = GAN_IN_PLANE_SPACING_NM,
    layer_height_nm: float = GAN_LAYER_HEIGHT_NM,
    transfer_width_nm: float = DEFAULT_TRANSFER_WIDTH_NM,
    coherence_patches: int = DEFAULT_COHERENCE_PATCHES,
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

    wavenumber = 2.0 * math.pi / electron_wavelength_nm(energy_kev)
    incidence = math.radians(grazing_angle_deg)
    # Odd pixel counts put the specular beam exactly on the centre pixel rather than between.
    rows, columns = (side | 1 for side in shape)
    exit_angle = incidence + np.radians(np.linspace(-span_deg, span_deg, rows))[:, None]
    deflection = np.radians(np.linspace(-span_deg, span_deg, columns))[None, :]

    q_x = wavenumber * (np.cos(exit_angle) * np.cos(deflection) - math.cos(incidence))
    q_y = wavenumber * np.cos(exit_angle) * np.sin(deflection)
    q_z = wavenumber * (np.sin(exit_angle) + math.sin(incidence))
    # Axial site positions are linear in the array indices, so the in-plane sum is a plain 2D
    # transform once the momentum transfer is expressed in the index basis.
    column_frequency = in_plane_spacing_nm * q_x
    row_frequency = in_plane_spacing_nm * (q_x + math.sqrt(3.0) * q_y) / 2.0

    windows, profile = _illuminated_patches(
        lattice, transfer_width_nm, in_plane_spacing_nm, coherence_patches
    )
    transform_size = TRANSFORM_PADDING * len(profile)
    # Frequencies outside the first Brillouin zone fold back onto it because the reciprocal
    # lattice repeats; that repetition is what puts the +/-1 rods on the screen.
    row_index = np.rint(row_frequency * transform_size / (2 * math.pi)).astype(int)
    column_index = np.rint(column_frequency * transform_size / (2 * math.pi)).astype(int)
    row_index, column_index = np.broadcast_arrays(
        row_index % transform_size, column_index % transform_size
    )

    intensity = np.zeros((rows, columns))
    for window in windows:
        levels = np.arange(int(window.min()), int(window.max()) + 1)
        footprints = np.stack(
            [
                np.fft.fft2(np.where(window == level, profile, 0.0), s=(transform_size,) * 2)
                for level in levels
            ]
        )
        out_of_plane = np.exp(-1j * layer_height_nm * q_z[None] * levels[:, None, None])
        amplitude = (footprints[:, row_index, column_index] * out_of_plane).sum(axis=0)
        intensity += np.abs(amplitude / profile.sum()) ** 2
    intensity /= len(windows)
    intensity[np.broadcast_to(exit_angle, intensity.shape) < 0.0] = 0.0

    # A Gaussian illumination of this width transforms to a Gaussian rod of this width, and
    # the rods repeat with the hexagonal reciprocal lattice.
    streak_width = 2.0 * _FWHM_PER_SIGMA * math.log(2.0) / (transfer_width_nm * wavenumber)
    rod_spacing = 4.0 * math.pi / (in_plane_spacing_nm * math.sqrt(3.0)) / wavenumber
    return ScreenPattern(
        intensity=intensity,
        exit_angle_deg=np.degrees(exit_angle).ravel(),
        deflection_deg=np.degrees(deflection).ravel(),
        grazing_angle_deg=grazing_angle_deg,
        beam_energy_kev=energy_kev,
        transfer_width_nm=transfer_width_nm,
        lattice_size=len(lattice),
        specular_intensity=float(intensity[rows // 2, columns // 2]),
        phase_order=phase_order(
            grazing_angle_deg, energy_kev=energy_kev, layer_height_nm=layer_height_nm
        ),
        streak_width_deg=math.degrees(streak_width),
        rod_spacing_deg=math.degrees(rod_spacing),
    )
