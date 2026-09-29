"""
analytics_engine
=================

A small, pluggable machine-learning engine: a single AnalyticEngine takes a
plain dict describing (1) which algorithm to run, (2) which method on that
algorithm to call, and (3) what data to load it with, and dispatches to the
matching class under `algorithms/`.

Trimmed and adapted from Spherity's original football data-analytics engine
(danalyticAPI) for reuse as a template on other projects. See README.md for
what was kept, what was dropped, and why.
"""
