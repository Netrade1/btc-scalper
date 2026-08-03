from __future__ import annotations

from dataclasses import dataclass


class ConfigurationError(ValueError):
    """Raised when platform safety configuration is inconsistent."""


@dataclass(frozen=True)
class PlatformConfig:
    """Governed research/paper execution configuration.

    Defaults enforce research-only paper-trading operation.
    """

    RESEARCH_ONLY: bool = True
    PAPER_TRADING_ONLY: bool = True
    LIVE_EXECUTION: bool = False
    HUMAN_APPROVAL_REQUIRED: bool = True
    AUTOMATIC_MODEL_PROMOTION: bool = False
    FAIL_CLOSED: bool = True

    PRODUCTION_APPROVED: bool = False
    LIVE_CAPITAL_APPROVED: bool = False

    def __post_init__(self) -> None:
        if self.FAIL_CLOSED and self.LIVE_EXECUTION:
            raise ConfigurationError("FAIL_CLOSED mode forbids LIVE_EXECUTION")
        if self.RESEARCH_ONLY and self.LIVE_CAPITAL_APPROVED:
            raise ConfigurationError("LIVE_CAPITAL_APPROVED is unavailable in research builds")
        if self.PAPER_TRADING_ONLY and self.LIVE_EXECUTION:
            raise ConfigurationError("PAPER_TRADING_ONLY forbids LIVE_EXECUTION")
        if self.AUTOMATIC_MODEL_PROMOTION:
            raise ConfigurationError("AUTOMATIC_MODEL_PROMOTION must remain disabled")

    @property
    def live_execution_enabled(self) -> bool:
        """True only when all live-capital controls are open."""

        return (
            self.LIVE_EXECUTION
            and self.LIVE_CAPITAL_APPROVED
            and self.HUMAN_APPROVAL_REQUIRED
            and not self.FAIL_CLOSED
            and not self.RESEARCH_ONLY
            and not self.PAPER_TRADING_ONLY
        )
