"""Extract Figure 3 experimental traces from the tracked CC BY paper PDF.

This is an optional provenance tool. Normal publication reproduction reads the committed
JSON and therefore requires only the locked Python environment.
"""

import json
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = ROOT / "nanomaterials-12-03052.pdf"
OUTPUT = ROOT / "data" / "reference" / "figure3_experimental_digitized.json"
RATIOS = (0.89, 0.82, 0.68)
X_LEFT = 195.622907
X_RIGHT = 381.973727
# SVG coordinates increase downwards: each tuple is the panel's top and bottom edge.
PANEL_EDGES = (
    (269.721881, 350.094656),
    (189.324744, 269.721881),
    (108.927606, 189.324744),
)
NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
POINT = re.compile(rf"[ML]\s*({NUMBER})\s+({NUMBER})")
MATRIX = re.compile(r"matrix\(([^)]+)\)")
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


def _transform(path: ET.Element) -> list[tuple[float, float]]:
    points = [(float(x), float(y)) for x, y in POINT.findall(path.attrib.get("d", ""))]
    match = MATRIX.fullmatch(path.attrib.get("transform", ""))
    if match is None:
        raise ValueError("Figure 3 path is missing its affine transform")
    a, b, c, d, e, f = (float(value.strip()) for value in match.group(1).split(","))
    return [(a * x + c * y + e, b * x + d * y + f) for x, y in points]


def extract(svg_path: Path) -> dict[str, object]:
    root = ET.parse(svg_path).getroot()
    paths = []
    for path in root.iter(f"{SVG_NAMESPACE}path"):
        if path.attrib.get("stroke") != "rgb(100%, 0%, 0%)" or path.attrib.get("fill") != "none":
            continue
        points = _transform(path)
        if len(points) > 50:  # Excludes the legend segment and circular point markers.
            paths.append(points)
    paths.sort(key=lambda points: points[0][1], reverse=True)
    if len(paths) != len(RATIOS):
        raise ValueError(f"expected three experimental paths, found {len(paths)}")

    traces = []
    for ratio, points, (top, bottom) in zip(RATIOS, paths, PANEL_EDGES, strict=True):
        clipped = [(x, y) for x, y in points if X_LEFT <= x <= X_RIGHT]
        samples: dict[float, list[float]] = {}
        for x, y in clipped:
            time = round((x - X_LEFT) / (X_RIGHT - X_LEFT) * 40.0, 8)
            samples.setdefault(time, []).append((bottom - y) / (bottom - top))
        time_s = sorted(samples)
        intensity = [sum(samples[time]) / len(samples[time]) for time in time_s]
        if len(time_s) < 50 or any(right <= left for left, right in pairwise(time_s)):
            raise ValueError(f"invalid extracted trace for Ga/N = {ratio:.2f}")
        traces.append(
            {
                "nominal_ga_n_ratio": ratio,
                "time_s": [round(value, 8) for value in time_s],
                "rheed_panel_coordinate": [round(value, 8) for value in intensity],
            }
        )

    return {
        "description": "Figure-derived experimental RHEED reference curves from paper Figure 3",
        "classification": "digitized visual reference; not raw experimental data",
        "source": {
            "paper": "Budagosky and Garcia-Cristobal, Nanomaterials 12, 3052 (2022)",
            "doi": "10.3390/nano12173052",
            "figure": 3,
            "pdf_page": 10,
            "original_experiment": "Adelmann et al., Journal of Applied Physics 91, 9638 (2002)",
            "license": "CC BY 4.0 for the source paper figure",
        },
        "extraction": {
            "method": "red vector path extraction from pdftocairo SVG output",
            "time_mapping": "linear mapping of the plotted x-axis bounds to 0-40 s",
            "normalization": (
                "each panel's plotted vertical bounds map linearly to [0,1]; this preserves "
                "the displayed panel coordinate and is not the unavailable author normalization"
            ),
            "limitations": (
                "figure-derived coordinates include plotting and digitization uncertainty; "
                "no raw values, detector units, or author normalization were available"
            ),
        },
        "traces": traces,
    }


def main() -> None:
    if shutil.which("pdftocairo") is None:
        raise SystemExit("pdftocairo is required only to refresh the committed reference JSON")
    with tempfile.TemporaryDirectory() as directory:
        svg_path = Path(directory) / "figure3.svg"
        subprocess.run(
            ["pdftocairo", "-f", "10", "-l", "10", "-svg", str(SOURCE_PDF), str(svg_path)],
            check=True,
        )
        data = extract(svg_path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
