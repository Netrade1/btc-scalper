"""
Ruin probability calculation utilities.

Uses the classical gambler's-ruin formula adapted for trading:

    P(ruin) = ((1 - p) / p) ^ N

where:
  p  = win probability
  N  = number of trade-size units in current capital

When p == 0.5 the formula degenerates; we fall back to a linear approximation.
If p < 0.5 (negative edge) we return 1.0 (certain ruin given infinite time).
All values are clamped to [0, 1].
"""

import math
from typing import List


def ruin_probability(
    wins: int,
    losses: int,
    capital: float,
    unit_size: float,
) -> float:
    """Return the probability of ruin (0–1).

    Args:
        wins:      Number of winning trades in history.
        losses:    Number of losing trades in history.
        capital:   Current account balance.
        unit_size: Size of one trade (stake per trade).

    Returns:
        Float in [0, 1] representing ruin probability.
    """
    if capital <= 0:
        return 1.0
    if unit_size <= 0:
        return 0.0

    total = wins + losses
    if total == 0:
        return 0.0

    p = wins / total          # empirical win rate

    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0

    n = capital / unit_size   # number of betting units

    if math.isclose(p, 0.5, rel_tol=1e-9):
        # Symmetric random walk — for a finite number of betting units N, the
        # probability of ruin before doubling the stake is N/(2N) = 0.5.
        # We approximate with 1/(N+1) as a conservative finite-horizon estimate.
        return max(0.0, min(1.0, 1.0 / (n + 1)))

    if p < 0.5:
        return 1.0

    ratio = (1 - p) / p
    pr = ratio ** n
    return max(0.0, min(1.0, pr))


def stats_from_trades(pnl_values: List[float]) -> dict:
    """Compute win/loss counts and average payoff from a list of P&L values."""
    wins = [v for v in pnl_values if v > 0]
    losses = [v for v in pnl_values if v <= 0]
    return {
        "total_trades": len(pnl_values),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(pnl_values) if pnl_values else 0.0,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "net_pnl": sum(pnl_values),
    }
