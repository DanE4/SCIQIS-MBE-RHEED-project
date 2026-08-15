"""Presentation layer for notebooks/mbe_rheed.py.

Kept out of `mbe_rheed_sim` so the simulation package never imports marimo or a plotting
library. marimo notebooks are single-file applications by design, so the notebook itself
stays one file and delegates everything that is not narrative or reactive wiring to these
modules — the split marimo recommends in its own best-practices guide:
https://docs.marimo.io/guides/best_practices/
"""
