import unittest

from k2think.audit import AuditChain, AuditEvent


class AuditChainTests(unittest.TestCase):
    def test_hash_chain_verifies(self) -> None:
        chain = AuditChain()
        chain.append(AuditEvent.create("forecast.recorded", {"symbol": "BTCUSDT", "score": 0.62}))
        chain.append(AuditEvent.create("execution.paper", {"order_id": "paper-1", "status": "filled"}))
        self.assertTrue(chain.verify())

    def test_tamper_detection_fails_verification(self) -> None:
        chain = AuditChain()
        chain.append(AuditEvent.create("forecast.recorded", {"symbol": "BTCUSDT"}))
        record = chain.records[0]
        chain._records[0] = type(record)(
            index=record.index,
            event=record.event,
            previous_hash=record.previous_hash,
            record_hash="deadbeef",
        )
        self.assertFalse(chain.verify())


if __name__ == "__main__":
    unittest.main()
