import json
from types import SimpleNamespace
import unittest
from evaluate.defense_methods.comparison_runtime import ComparisonRuntime

class ReceivedProvenanceTests(unittest.TestCase):
    def test_cycles_and_self_messages_preserve_each_label_once(self):
        runtime=ComparisonRuntime(SimpleNamespace(method="no_defense_memrl",agents=3))
        first=runtime.rules.label(owner=0,readers=[0,1],channel="external")
        second=runtime.rules.label(owner=1,readers=[1,2],channel="agent_output")
        runtime.received={0:[first],1:[second]}
        for _ in range(3):
            for sender,recipient in [(0,0),(0,1),(1,2),(2,0)]:
                self.assertEqual(runtime.route("unchanged response",sender,recipient),"unchanged response")
        for labels in runtime.received.values():
            self.assertEqual(len(labels),2)
            self.assertEqual({json.dumps(x,sort_keys=True) for x in labels},
                             {json.dumps(first,sort_keys=True),json.dumps(second,sort_keys=True)})
        first["taints"].append("mutated-outside-runtime")
        self.assertNotIn("mutated-outside-runtime",runtime.received[2][0]["taints"])

    def test_deduplication_preserves_acl_denial(self):
        runtime=ComparisonRuntime(SimpleNamespace(method="provenance_acl",agents=2))
        label=runtime.rules.label(owner=0,readers=[0],channel="agent_output")
        runtime.received={0:[label,label]}
        self.assertIsNone(runtime.route("private data",0,1))
        self.assertEqual(runtime.received.get(1,[]),[])

if __name__=="__main__":unittest.main()
