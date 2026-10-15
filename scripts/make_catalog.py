#!/usr/bin/env python3
"""Regenerates references/catalog.md (every analysis, its JMP/SPSS equivalent and its exact --params) from the code's own docstrings."""
import inspect, os, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run
out = ["# Analysis catalog (auto-generated from code - do not edit by hand)", "",
       "Run `python scripts/make_catalog.py` to refresh. Parameters are JSON keys passed with `--params`. `[optional]`/defaults are as written by the function's author.", ""]
groups = {}
for k, (fn, desc) in run.ANALYSES.items(): groups.setdefault(fn.__module__, []).append((k, fn, desc))
titles = {"ss_desc": "Descriptive & multivariate", "ss_linear": "Linear models, GLM, power", "ss_spc": "SPC, capability, Gage R&R", "ss_rel": "Reliability & survival", "ss_doe": "DOE (classical)",
          "ss_stats2": "Classical tests, agreement, reliability of scales, transforms", "ss_models2": "Mixed models, factor/discriminant, ordinal/multinomial, nonlinear, ML, time series",
          "ss_spc2": "Advanced SPC & MSA", "ss_rel2": "Advanced reliability", "ss_doe2": "Advanced DOE", "ss_graph": "Graphs & Fit Y by X"}
for mod in titles:
    if mod not in groups: continue
    out += [f"## {titles[mod]}  (`{mod}.py`)", ""]
    for k, fn, desc in groups[mod]:
        doc = inspect.getdoc(fn) or ""
        nodata = " - **no data file needed**" if k in run.NO_DATA else ""
        out += [f"### `{k}`{nodata}", desc, "", "```", textwrap.fill(doc, 110, replace_whitespace=False) if doc else "(see description)", "```", ""]
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references", "catalog.md")
open(path, "w", encoding="utf-8").write("\n".join(out)); print("wrote", os.path.abspath(path), len(out), "lines")
