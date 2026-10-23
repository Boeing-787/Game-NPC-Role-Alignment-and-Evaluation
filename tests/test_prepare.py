"""不需要 GPU 或第三方机器学习依赖，仅测试数据校验逻辑。"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_data import validate_record, text_key

class TestData(unittest.TestCase):
    def setUp(self):
        self.r = {"id":"D1", "group_id":"G1", "source_ids":["S1"],
                  "category":"daily", "construction_method":"grounded_adaptation",
                  "messages":[{"role":"user","content":"你好"}, {"role":"assistant","content":"你好。"}]}

    def test_valid(self):
        validate_record(self.r, {"S1"}, 1)

    def test_invalid_source(self):
        with self.assertRaises(ValueError):
            validate_record(self.r, {"S2"}, 1)

    def test_reject_template(self):
        self.r["id"] = "EXAMPLE_1"
        with self.assertRaises(ValueError):
            validate_record(self.r, {"S1"}, 1)

    def test_text_key(self):
        self.assertEqual(text_key("你 好"), text_key("你好"))

if __name__ == "__main__":
    unittest.main()
