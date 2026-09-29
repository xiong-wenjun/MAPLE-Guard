import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools import run_appworld_matrix as matrix
from maple_guard import run_appworld, run_mmlu

class AgentSafeMatrixTests(unittest.TestCase):
    def args(self, root, **changes):
        values=dict(profile="strict",phase="smoke",backbone="qwen",tasks=2,
            bundle="/tmp/tasks.json",run_root=str(root/"run"),task_service="qwen",judge_service="judge",
            agentsafe_policy=str(root/"policy.json"),agentsafe_criteria=str(root/"criteria.json"),
            agentsafe_threshold=0.18,agentsafe_profile="paper_v2_adapted",
            agentsafe_calibration_manifest=str(root/"calibration.json"),agentsafe_review_interval=1)
        values.update(changes)
        return SimpleNamespace(**values)
    def services(self):
        return {"qwen":{"base_url":"http://task/v1","model":"Qwen/Qwen3.5-122B-A10B"},
            "judge":{"base_url":"http://judge/v1","model":"Qwen/Qwen3.5-122B-A10B"},
            "embedding":{"base_url":"http://embed/v1","model":"Qwen/Qwen3-Embedding-8B"}}
    def test_adaptation_identity_and_calibration_are_carried_into_real_cli_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ("policy.json","criteria.json","calibration.json"):(root/name).write_text("{}")
            args=self.args(root);job=matrix.build_job(args,"agentsafe_full",42,self.services())
            command=job["command"]
            self.assertIn("--agentsafe-profile",command)
            self.assertEqual(command[command.index("--agentsafe-profile")+1],"paper_v2_adapted")
            self.assertEqual(command[command.index("--agentsafe-calibration-manifest")+1],args.agentsafe_calibration_manifest)
            self.assertEqual(command[command.index("--agentsafe-review-interval")+1],"1")
            offset=command.index("maple_guard.run_appworld")
            with patch("sys.argv",["run_appworld",*command[offset+1:]]):
                actual=run_mmlu.resolve_args(run_appworld.parse_args())
            self.assertEqual(actual.agentsafe_profile,"paper_v2_adapted")
            info=job["baseline_configuration"]
            self.assertEqual(info["reporting_label"],"AgentSafe (full-component adaptation)")
            self.assertEqual(info["calibration_manifest_sha256"],hashlib.sha256(b"{}").hexdigest())
            self.assertFalse(info["official_configuration_recovered"])
    def test_accepted_adaptation_cannot_launch_without_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            job=matrix.build_job(self.args(Path(d)),"agentsafe_full",42,self.services())
            self.assertEqual(job["status"],"blocked")
            self.assertIn("calibration",job["reason"].lower())

if __name__=="__main__":unittest.main()
