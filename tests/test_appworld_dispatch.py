import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "tools/dispatch_appworld_matrix.py"

class MatrixDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODULE.exists():
            cls.dispatch = None
            return
        spec = importlib.util.spec_from_file_location("dispatch_appworld_matrix", MODULE)
        cls.dispatch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.dispatch)

    def api(self):
        self.assertIsNotNone(self.dispatch, "Prepared matrix needs a bounded dispatcher")
        return self.dispatch

    def test_reserved_star42_is_not_reexecuted_even_if_owner_has_not_started(self):
        d=self.api()
        job={"method":"maple_guard","topology":"star","seed":42,"status":"prepared"}
        self.assertEqual(d.disposition("qwen",job,{("qwen","maple_guard","star",42)},True,False,False),"reserved")

    def test_asset_blockers_and_failed_family_override_readiness(self):
        d=self.api()
        job={"method":"infa_guard_full","topology":"chain","seed":43,"status":"blocked"}
        self.assertEqual(d.disposition("qwen",job,set(),True,False,False),"blocked")
        job["status"]="prepared"
        self.assertEqual(d.disposition("qwen",job,set(),True,True,False),"held_after_failure")

    def test_unvalidated_and_existing_cells_never_launch(self):
        d=self.api()
        job={"method":"agentxposed_full_guide","topology":"chain","seed":43,"status":"prepared"}
        self.assertEqual(d.disposition("qwen",job,set(),False,False,False),"waiting_probe")
        self.assertEqual(d.disposition("qwen",job,set(),True,False,True),"existing")
        self.assertEqual(d.disposition("qwen",job,set(),True,False,False),"queued")

    def test_slot_budget_accounts_for_live_pilots(self):
        d=self.api()
        self.assertEqual(d.available_slots(6,4,2,0),2)
        self.assertEqual(d.available_slots(6,6,2,2),0)
        self.assertEqual(d.available_slots(8,10,2,0),0)
        self.assertEqual(d.available_slots(6,5,2,1),1)

    def make_probe(self, root, status="completed", count=2):
        folder=root/"probe";folder.mkdir()
        manifest={"status":status,"exit_code":0,"phase":"smoke","method":"agentxposed_full_guide","completed_tasks":count}
        (folder/"run.json").write_text(json.dumps(manifest))
        (folder/"trace.jsonl").write_text('{}\n'*count)
        (folder/"trace.summary.json").write_text('{}')
        (folder/"api-calls.jsonl").write_text(json.dumps({"http_status":200,"finish_reasons":["stop"],"final_content_present":[True],"invalid_for_benchmark":False})+'\n')
        return folder

    def test_smoke_requires_success_summary_actual_count_and_clean_journal(self):
        d=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            f=self.make_probe(Path(tmp))
            self.assertTrue(d.probe_ready(f/"run.json"))
            (f/"api-calls.jsonl").write_text('{"http_status":500}\n')
            self.assertFalse(d.probe_ready(f/"run.json"))
            (f/"api-calls.jsonl").write_text('{"invalid_for_benchmark":true}\n')
            self.assertFalse(d.probe_ready(f/"run.json"))
            (f/"api-calls.jsonl").write_text('{"http_status":200,"finish_reasons":["length"]}\n')
            self.assertFalse(d.probe_ready(f/"run.json"))
            (f/"api-calls.jsonl").write_text('{"http_status":200}\n')
            (f/"trace.summary.json").unlink()
            self.assertFalse(d.probe_ready(f/"run.json"))

    def test_incomplete_agentxposed_probe_is_not_ready(self):
        d=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            f=self.make_probe(Path(tmp),status="running",count=1)
            self.assertFalse(d.probe_ready(f/"run.json"))

    def test_live_base_pilot_is_readiness_evidence_but_not_a_completed_result(self):
        d=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            f=self.make_probe(Path(tmp),status="running",count=3)
            row=json.loads((f/"run.json").read_text())
            row.update(method="no_defense_memrl",phase="pilot",exit_code=None,completed_tasks=None)
            (f/"run.json").write_text(json.dumps(row))
            self.assertTrue(d.probe_ready(f/"run.json"))
            row["method"]="agentxposed_full_guide"
            (f/"run.json").write_text(json.dumps(row))
            self.assertFalse(d.probe_ready(f/"run.json"))

    def test_duplicate_dispatcher_lock_is_rejected(self):
        d=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"lock"
            with d.acquire_lock(p):
                with self.assertRaises(BlockingIOError):
                    with d.acquire_lock(p): pass

    def test_launch_failure_never_overwrites_a_competing_run_manifest(self):
        d=self.api()
        self.assertTrue(hasattr(d,"record_launch_failure"), "Launch errors need a separate journal")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/"cell";folder.mkdir()
            manifest=folder/"run.json";original='{"status":"running","pid":123}'
            manifest.write_text(original)
            job={"run_id":"cell","method":"maple_guard","directory":str(folder)}
            failures=d.record_launch_failure(root,"inference2","qwen",job,FileExistsError())
            self.assertEqual(manifest.read_text(),original)
            self.assertEqual(failures["cell"]["error_type"],"FileExistsError")

    def test_pid_match_rejects_unrelated_process(self):
        d=self.api()
        self.assertFalse(d.process_matches(1,"a-run-id-that-is-not-in-pid1"))

if __name__=="__main__":
    unittest.main()
