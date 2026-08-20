"""Rendering conventions for the static Figure 3/4 morphology panels."""

import sys
from pathlib import Path

import numpy as np

# `scripts/` is a directory of entry points, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import matplotlib.pyplot as plt
from figure3_plots import _draw_hex_cells


def test_hex_cells_tile_every_site_exactly_once() -> None:
    size = 12
    heights = np.arange(size * size).reshape(size, size) % 3
    figure, axis = plt.subplots()
    try:
        collection = _draw_hex_cells(axis, heights, int(heights.max()))
        paths = collection.get_paths()
        # One hexagonal cell per lattice site, coloured by that site's column height.
        assert len(paths) == size * size
        assert np.array_equal(collection.get_array(), heights.ravel().astype(float))

        centers = np.asarray([path.vertices[:6].mean(axis=0) for path in paths])
        row, column = np.indices(heights.shape)
        expected = np.column_stack(
            (column.ravel() + 0.5 * row.ravel(), np.sqrt(3.0) / 2.0 * row.ravel())
        )
        assert np.allclose(centers, expected)
        # Cells are the lattice's Voronoi hexagons, so their areas sum to the rhombus area
        # exactly: no overlap to smear the slanted boundaries, no gaps between sites.
        areas = []
        for path in paths:
            corners = path.vertices[:6]
            shifted = np.roll(corners, -1, axis=0)
            areas.append(
                0.5
                * abs(
                    np.sum(corners[:, 0] * shifted[:, 1] - shifted[:, 0] * corners[:, 1])
                )
            )
        assert np.allclose(areas, np.sqrt(3.0) / 2.0)
        assert np.isclose(np.sum(areas), size * size * np.sqrt(3.0) / 2.0)
        assert axis.get_aspect() == 1.0
    finally:
        plt.close(figure)
