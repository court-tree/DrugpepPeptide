from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phase3.drugclip.finalize import load_resume_summary


class FinalizeContractTests(unittest.TestCase):
    def test_resume_rejects_v1_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps({"schema_version": "old-v1"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not resume from v1"):
                load_resume_summary(path, "active-v2")


if __name__ == "__main__":
    unittest.main()
