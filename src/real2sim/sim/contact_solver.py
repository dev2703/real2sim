"""Explicit simulator contact options for reproducible sensitivity comparisons.

Absent options preserve the pinned model. These settings change simulated dynamics;
they do not calibrate physical friction, compliance or grasp retention.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


def contact_solver_attributes(config):
    requested = config.get("contact_solver")
    if requested is None:
        return {}
    if not isinstance(requested, Mapping):
        raise ValueError("contact_solver must be a mapping")
    if unknown := requested.keys() - {"impratio", "noslip_iterations"}:
        raise ValueError(f"unsupported contact_solver options: {sorted(unknown)}")
    attributes = {}
    if "impratio" in requested:
        ratio = requested["impratio"]
        if (isinstance(ratio, bool) or not isinstance(ratio, Real)
                or not np.isfinite(ratio) or ratio <= 0):
            raise ValueError("contact_solver.impratio must be finite and positive")
        attributes["impratio"] = str(float(ratio))
    if "noslip_iterations" in requested:
        iterations = requested["noslip_iterations"]
        if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 0:
            raise ValueError("contact_solver.noslip_iterations must be a nonnegative integer")
        attributes["noslip_iterations"] = str(int(iterations))
    return attributes


def contact_solver_settings(model):
    """Record compiled settings, including inherited defaults, in run provenance."""
    return {"impratio": float(model.opt.impratio),
            "noslip_iterations": int(model.opt.noslip_iterations),
            "noslip_tolerance": float(model.opt.noslip_tolerance),
            "solver": int(model.opt.solver), "cone": int(model.opt.cone)}
