import unittest

from k2think.config import ConfigurationError, PlatformConfig


class PlatformConfigTests(unittest.TestCase):
    def test_defaults_are_research_safe(self) -> None:
        cfg = PlatformConfig()
        self.assertFalse(cfg.live_execution_enabled)

    def test_rejects_live_execution_when_fail_closed(self) -> None:
        with self.assertRaises(ConfigurationError):
            PlatformConfig(LIVE_EXECUTION=True)

    def test_rejects_live_capital_in_research_build(self) -> None:
        with self.assertRaises(ConfigurationError):
            PlatformConfig(LIVE_CAPITAL_APPROVED=True)


if __name__ == "__main__":
    unittest.main()
