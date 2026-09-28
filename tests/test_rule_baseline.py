import importlib.util
import unittest
from types import SimpleNamespace

class RuleBaselineTests(unittest.TestCase):
    def component(self):
        self.assertIsNotNone(importlib.util.find_spec("evaluate.defense_methods.rule_baseline"),
                             "operational provenance/ACL component is missing")
        from evaluate.defense_methods.rule_baseline import ProvenanceACL
        return ProvenanceACL()

    def test_acl_blocks_unauthorized_reader(self):
        rule = self.component()
        label = rule.label(owner=0, readers=[0], channel="agent_output")
        self.assertFalse(rule.decide(label, "retrieval", actor=1)[0])
        self.assertTrue(rule.decide(label, "retrieval", actor=0)[0])

    def test_taint_survives_summarization_and_parent_union(self):
        rule = self.component()
        external = rule.label(owner=0, readers=[0,1], channel="external")
        child = rule.label(owner=1, readers=[0,1], channel="agent_output", parents=[external])
        self.assertEqual(child["taints"], ["untrusted_external"])
        self.assertIn("external", child["sources"])
        for phase in ("write", "retrieval", "promotion", "transfer"):
            self.assertFalse(rule.decide(child, phase, actor=1, recipients=[1])[0])

    def test_derivation_cannot_widen_parent_acl(self):
        rule = self.component()
        parent = rule.label(owner=0, readers=[0], channel="agent_output")
        child = rule.label(owner=0, readers=[0,1], channel="agent_output", parents=[parent])
        self.assertEqual(child["readers"], [0])
        self.assertFalse(rule.decide(child, "promotion", actor=0, recipients=[0,1])[0])

    def test_unknown_loaded_records_are_not_assumed_clean(self):
        rule = self.component()
        label = rule.label(owner=0, readers=[0], channel="unknown")
        self.assertFalse(rule.decide(label, "retrieval", actor=0)[0])

    def test_labels_do_not_use_evaluator_poison_fields(self):
        rule = self.component()
        args = dict(owner=0, readers=[0], channel="agent_output")
        entry = SimpleNamespace(taint="poisoned", source_type="direct_memory_injection",
                                success_count=99, content_hazard=1.0)
        self.assertEqual(rule.label(**args), rule.label(**args))
        self.assertTrue(rule.decide(rule.label(**args), "write", actor=0)[0])
        self.assertEqual(entry.taint, "poisoned")

    def test_no_implicit_declassification_on_success(self):
        rule = self.component()
        parent = rule.label(owner=0, readers=[0], channel="external")
        child = rule.label(owner=0, readers=[0], channel="agent_output", parents=[parent])
        self.assertFalse(rule.decide(child, "promotion", actor=0, recipients=[0])[0])

    def test_bad_schema_fails_closed(self):
        rule = self.component()
        with self.assertRaises(ValueError):
            rule.decide({"readers":[0]}, "retrieval", actor=0)
        with self.assertRaises(ValueError):
            rule.label(owner=0, readers=[0], channel="magic_trusted_attack")

class OwnershipTests(unittest.TestCase):
    def test_reader_cannot_write_or_promote_owners_record(self):
        from evaluate.defense_methods.rule_baseline import ProvenanceACL
        rules=ProvenanceACL()
        label=rules.label(owner=0,readers=[0,1],channel="agent_output")
        self.assertTrue(rules.decide(label,"retrieval",actor=1)[0])
        for phase in ("write","promotion"):
            self.assertFalse(rules.decide(label,phase,actor=1,recipients=[0,1])[0])

if __name__ == "__main__":
    unittest.main()
