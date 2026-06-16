"""Wilson score interval for a single proportion.

Used for per-arm pooled accuracy reporting and for the power planner's half-width sanity checks.
The Wilson interval is well-behaved near 0/1 and for small n, unlike the normal (Wald) interval.
The ``z`` multiplier is derived from the requested confidence via ``scipy.stats.norm.ppf`` — never
hardcoded.
"""

from __future__ import annotations

from scipy.stats import norm

from ..logging import get_logger

_log = get_logger("stats.wilson")


def _z_for(confidence: float) -> float:
    """Two-sided normal quantile for the given confidence (e.g. 0.95 -> ~1.96)."""

    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    alpha = 1.0 - confidence
    return float(norm.ppf(1.0 - alpha / 2.0))


def wilson_interval(successes: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion, returned as a fraction in ``[0, 1]``.

    ``successes`` must satisfy ``0 <= successes <= n`` and ``n >= 1``.
    """

    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, n]; got {successes} of {n}")

    z = _z_for(confidence)
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z / denom) * ((p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) ** 0.5)
    low = max(0.0, center - half)
    high = min(1.0, center + half)

    _log.info(
        "wilson interval",
        extra={
            "fields": {
                "successes": successes,
                "n": n,
                "confidence": confidence,
                "low": low,
                "high": high,
            }
        },
    )
    return (low, high)


def wilson_half_width_pp(p: float, n: int, *, confidence: float = 0.95) -> float:
    """Half-width of the Wilson interval at proportion ``p`` and sample size ``n``, in pp.

    Computed by evaluating the Wilson interval at ``successes = round(p * n)`` and halving the span.
    Sanity: ``n=100, p=0.5, 95%`` gives ~9.6 pp.
    """

    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")

    successes = round(p * n)
    low, high = wilson_interval(successes, n, confidence=confidence)
    return (high - low) / 2.0 * 100.0
