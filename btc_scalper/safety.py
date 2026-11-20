"""Deployment safety switches. Changing these is a code change that needs review.

PAPER_TRADING_ONLY: no code path may send a real order. ``LiveRouter`` raises
    PermissionError while this is True. Flip it only after (1) a replay on real
    exchange history and (2) a forward paper run both pass, and the Risk Chair
    has signed off.
FAIL_CLOSED: on missing/invalid data, unhealthy models or errors the system
    takes no new risk. It never substitutes a guess and carries on.
"""
PAPER_TRADING_ONLY = True
FAIL_CLOSED = True


class TradingHalted(RuntimeError):
    """Raised when a fail-closed condition must stop the loop."""
