"""Focused unit tests for JUDGEBERT's lightweight legal-risk rules.

Run from the project root:
    python -m unittest discover -s tests -v

These tests validate rule behaviour only; they do not validate legal correctness.
"""
import ast
import pathlib
import re
import unittest
import numpy as np

APP_PATH = pathlib.Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
wanted = {"normalize_legal_text", "extract_terms", "extract_numbers", "legal_risk_check"}
constants = {"NEGATION_TERMS", "MODAL_TERMS", "CONDITION_TERMS"}
nodes = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        nodes.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in constants
        for target in node.targets
    ):
        nodes.append(node)

namespace = {"re": re, "np": np}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP_PATH), "exec"), namespace)
risk_check = namespace["legal_risk_check"]


class LegalRiskRuleTests(unittest.TestCase):
    def test_equivalent_english_paraphrase_no_critical_warning(self):
        warnings = risk_check(
            "The tenant must pay rent by the fifth day.",
            "The tenant is required to pay rent by the fifth day."
        )
        self.assertFalse(any("CHANGE:" in w for w in warnings))

    def test_english_negation_change_detected(self):
        warnings = risk_check("The tenant must pay rent.", "The tenant must not pay rent.")
        self.assertTrue(any("NEGATION CHANGE" in w for w in warnings))

    def test_english_modal_change_detected(self):
        warnings = risk_check("The tenant may terminate.", "The tenant must terminate.")
        self.assertTrue(any("MODAL CHANGE" in w for w in warnings))

    def test_english_condition_change_detected(self):
        warnings = risk_check(
            "The tenant may terminate if notice is given.",
            "The tenant may terminate."
        )
        self.assertTrue(any("CONDITION CHANGE" in w for w in warnings))

    def test_english_number_change_detected(self):
        warnings = risk_check("Pay 5000 rupees.", "Pay 6000 rupees.")
        self.assertTrue(any("NUMBER CHANGE" in w for w in warnings))

    def test_hindi_negation_change_detected(self):
        warnings = risk_check("किरायेदार किराया देगा।", "किरायेदार किराया नहीं देगा।")
        self.assertTrue(any("NEGATION CHANGE" in w for w in warnings))

    def test_hindi_modal_change_detected(self):
        warnings = risk_check("किरायेदार किराया दे सकता है।", "किरायेदार किराया देना चाहिए।")
        self.assertTrue(any("MODAL CHANGE" in w for w in warnings))

    def test_hindi_condition_change_detected(self):
        warnings = risk_check("यदि किराया दिया जाए तो अनुबंध जारी रहेगा।", "अनुबंध जारी रहेगा।")
        self.assertTrue(any("CONDITION CHANGE" in w for w in warnings))

    def test_devanagari_number_change_detected(self):
        warnings = risk_check("राशि ५००० रुपये है।", "राशि ६००० रुपये है।")
        self.assertTrue(any("NUMBER CHANGE" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
