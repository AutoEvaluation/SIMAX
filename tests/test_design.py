"""The released study design: personas, targets and the condition bank."""
import collections
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import generate_dialogues as gen

CONFIG = json.loads((ROOT / "configs/generation.json").read_text())
PLAN = json.loads((ROOT / "configs/persona_plan.json").read_text())["blocks"]
BANK = json.loads((ROOT / "configs/condition_bank.json").read_text())["conditions"]


class DesignTest(unittest.TestCase):
    def test_every_label_has_15_persona_blocks(self):
        counts = collections.Counter(b["label"] for b in gen.build_blocks(CONFIG))
        self.assertEqual(len(counts), 21)
        self.assertEqual(set(counts.values()), {15})

    def test_plan_covers_three_specialties(self):
        self.assertEqual(len(PLAN), 315)
        self.assertEqual(set(b["department"] for b in PLAN.values()), {"obstetrics", "orthopedics", "rheumatology"})

    def test_high_count_targets(self):
        design = CONFIG["design"]
        high = {label: design["wiser"]["high_count"].get(label, design["wiser"]["high_count"]["default"])
                for label in design["wiser"]["labels"]}
        for family in design["count_families"].values():
            high.update({label: family["high_count"].get(label, family["high_count"]["default"]) for label in family["labels"]})
        self.assertEqual(high, {"EO": 10, "ER": 10, "ES": 10, "OE": 10, "RS": 10, "S": 5, "EQ": 4, "AQ": 8, "AR": 8, "SD": 5, "W": 5,
                                "ASK": 3, "ADVISED": 3, "ASSIST": 3, "ARRANGE": 3, "ASSESS": 5})

    def test_conditions_and_codes_come_from_the_bank(self):
        bank = {(c["department"], c["condition"]): c["icd10"] for c in BANK}
        self.assertEqual(len(BANK), 450)
        for code in bank.values():
            self.assertRegex(code, re.compile(r"^[A-Z]\d{2}(\.\d)?$"))
        for block in PLAN.values():
            self.assertEqual(block["icd10"], bank[(block["department"], block["case"]["condition"])])

    def test_profile_templates_are_blank_and_refused(self):
        from simax.generation.long import load_profile, load_topic_profile

        for loader, name in ((load_profile, "dialogue_profile.json"), (load_topic_profile, "topic_profile.json")):
            with self.assertRaisesRegex(ValueError, "blank values"):
                loader(ROOT / "configs" / name)


if __name__ == "__main__":
    unittest.main()
