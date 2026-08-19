"""Reduce a source animation to the greyscale frames the reconstruction overlay draws.

Run once, by hand, with a local animation:

    uv run python scripts/build_reconstruction_frames.py path/to/animation.gif

It writes `data/reconstruction/template_frames.npz` and a contact-sheet preview beside it. Only
the reduction is committed: 8-bit greyscale, contrast stretched once across the whole animation so
every frame shares one mapping, which is all the notebook needs to turn a frame into a height
field. The source animation stays out of the repository.

Nothing in the simulation reads this. The frames it produces are display-only, and the module that
loads them says so at length.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "reconstruction" / "template_frames.npz"
# Enough to fill the largest lattice the notebook offers without upsampling; a bigger reduction
# would only store detail no lattice can show.
MAX_EDGE = 200
# Clipped off each end before stretching, so a few extreme pixels cannot flatten everything else.
CLIP_PERCENTILE = 2.0


def _frames(path: Path) -> np.ndarray:
    """Every frame as greyscale, square-cropped about the centre and reduced to MAX_EDGE."""
    frames = []
    for frame in ImageSequence.Iterator(Image.open(path)):
        image = frame.convert("L")
        edge = min(image.size)
        left, top = (image.width - edge) // 2, (image.height - edge) // 2
        image = image.crop((left, top, left + edge, top + edge))
        if edge != MAX_EDGE:
            image = image.resize((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
        frames.append(np.asarray(image, dtype=float))
    if not frames:
        raise SystemExit(f"{path} has no frames")
    return np.stack(frames)


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <animation>")
    source = Path(argv[1])
    stack = _frames(source)

    # One stretch for the whole animation, not per frame: a per-frame stretch would rescale the
    # tones every frame, so the surface would pulse instead of the structure moving.
    low, high = np.percentile(stack, [CLIP_PERCENTILE, 100.0 - CLIP_PERCENTILE])
    stretched = np.clip((stack - low) / (high - low), 0.0, 1.0)
    stored = np.rint(stretched * 255.0).astype(np.uint8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, luminance=stored)
    preview = OUTPUT.with_suffix(".png")
    columns = min(8, len(stored))
    rows = -(-len(stored) // columns)
    sheet = np.zeros((rows * MAX_EDGE, columns * MAX_EDGE), dtype=np.uint8)
    for index, frame in enumerate(stored):
        row, column = divmod(index, columns)
        sheet[row * MAX_EDGE : (row + 1) * MAX_EDGE, column * MAX_EDGE : (column + 1) * MAX_EDGE] = (
            frame
        )
    Image.fromarray(sheet).save(preview)

    print(
        f"{source} -> {OUTPUT.relative_to(ROOT)}: {len(stored)} frames of "
        f"{MAX_EDGE}x{MAX_EDGE}, {OUTPUT.stat().st_size / 1024:.0f} KiB "
        f"(stretched from {low:.0f}-{high:.0f}); preview {preview.name}"
    )


if __name__ == "__main__":
    main(sys.argv)
