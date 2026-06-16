"""Power / runs planning for the paired multi-run non-inferiority design.

This is a PLANNING HEURISTIC, not an exact power calculation. It answers: given an estimate of the
per-task run-to-run noise (``sigma_pp``), a fixed task set (``n_tasks``), and a non-inferiority
margin (``margin_pp``), how many runs per (task, arm) cell, ``k``, do we need so the test can
resolve a true delta of 0 within the margin at the requested ``alpha`` and ``power``?

Model: the paired mean delta over ``n_tasks`` tasks, each averaged over ``k`` runs, has standard
error ``SE ~= sigma_pp / sqrt(n_tasks * k)``. A standard one-sided non-inferiority sizing requires
``(z_alpha + z_power) * SE <= margin_pp``, i.e.

    (z_alpha + z_power) * sigma_pp / sqrt(n_tasks * k) <= margin_pp.

Solving for the smallest integer ``k >= 1``. The ``z`` values come from ``scipy.stats.norm.ppf`` —
no hardcoded constants. The approximation ignores task-to-task variance structure and treats
``sigma_pp`` as a single pooled noise scale, so treat the result as a lower-bound planning figure.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from ..logging import get_logger

_log = get_logger("stats.power")


def _z_one_sided(prob: float) -> float:
    """One-sided normal quantile ``Phi^{-1}(prob)`` (e.g. prob=0.95 -> ~1.645)."""

    if not 0.0 < prob < 1.0:
        raise ValueError(f"probability must be in (0, 1); got {prob}")
    return float(norm.ppf(prob))


def recommended_runs(
    *,
    sigma_pp: float,
    margin_pp: float,
    n_tasks: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Smallest ``k >= 1`` runs/cell so the paired non-inferiority test is adequately powered.

    See the module docstring for the sizing model. Returns ``ceil`` of the solved ``k``, floored at
    1. Raises ``ValueError`` on non-positive ``sigma_pp``/``margin_pp``/``n_tasks``.
    """

    if sigma_pp <= 0.0:
        raise ValueError(f"sigma_pp must be > 0; got {sigma_pp}")
    if margin_pp <= 0.0:
        raise ValueError(f"margin_pp must be > 0; got {margin_pp}")
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1; got {n_tasks}")

    z_alpha = _z_one_sided(1.0 - alpha)
    z_power = _z_one_sided(power)
    z_sum = z_alpha + z_power

    # (z_sum * sigma / sqrt(n*k))^2 <= margin^2  =>  k >= (z_sum*sigma)^2 / (margin^2 * n)
    k_real = (z_sum * sigma_pp) ** 2 / (margin_pp**2 * n_tasks)
    k = max(1, math.ceil(k_real))

    _log.info(
        "recommended runs",
        extra={
            "fields": {
                "sigma_pp": sigma_pp,
                "margin_pp": margin_pp,
                "n_tasks": n_tasks,
                "alpha": alpha,
                "power": power,
                "k_real": k_real,
                "k": k,
            }
        },
    )
    return k


def is_underpowered(
    *,
    k_runs: int,
    sigma_pp: float,
    margin_pp: float,
    n_tasks: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> bool:
    """True iff the planned ``k_runs`` is below ``recommended_runs(...)`` for the same design."""

    needed = recommended_runs(
        sigma_pp=sigma_pp,
        margin_pp=margin_pp,
        n_tasks=n_tasks,
        alpha=alpha,
        power=power,
    )
    under = k_runs < needed
    _log.info(
        "underpowered check",
        extra={
            "fields": {
                "k_runs": k_runs,
                "recommended": needed,
                "underpowered": under,
            }
        },
    )
    return under
