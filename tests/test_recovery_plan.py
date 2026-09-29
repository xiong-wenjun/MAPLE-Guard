import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.run_recovery_plan import rewrite_job, retryable_failure, lane_lock, validate_result, run_lane

class RecoveryPlanTests(unittest.TestCase):
    def job(self):
        return {"run_id":"original","directory":"/old/run/original","method":"agentsafe_full",
                "command":["python","-B","/old/source/tools/run_instrumented.py","maple_guard.infa_memlink_eval",
                 "--methods","agentsafe_full","--output-root","/old/run/original",
                 "--memory-store-dir","/old/run/original/memory","--memory-run-id","original",
                 "--baseline-experiment-id","original","--baseline-state-path","/old/run/original/state.json",
                 "--config","configs/csqa_star.yaml","--max-tokens","1024"]}

    def test_restart_isolates_memory_and_preserves_generation_protocol(self):
        old=self.job()
        new=rewrite_job(old,Path("/fresh/retry"),Path("/new/source"),{"--chat-timeout":"600"})
        args=new["command"]
        self.assertEqual(old["directory"],"/old/run/original")
        self.assertEqual(args[args.index("--max-tokens")+1],"1024")
        self.assertEqual(args[args.index("--chat-timeout")+1],"600")
        self.assertEqual(args[args.index("--memory-store-dir")+1],"/fresh/retry/memory")
        self.assertEqual(args[args.index("--baseline-state-path")+1],"/fresh/retry/state.json")
        self.assertEqual(args[args.index("--memory-run-id")+1],"retry")
        self.assertEqual(args[args.index("--baseline-experiment-id")+1],"retry")
        self.assertIn("/new/source/tools/run_instrumented.py",args)
        self.assertEqual(args[args.index("--config")+1],"/old/source/configs/csqa_star.yaml")
        self.assertFalse(any("/old/run/original" in x for x in args))

    def test_override_cannot_silently_change_method_or_seed(self):
        for flag in ("--methods","--seed","--benchmark-bundle","--strict-comparison"):
            with self.subTest(flag=flag),self.assertRaises(ValueError):
                rewrite_job(self.job(),Path("/fresh/retry"),Path("/new/source"),{flag:"different"})

    def test_lane_lock_prevents_duplicate_controller(self):
        with tempfile.TemporaryDirectory() as folder:
            with lane_lock(Path(folder)/"lane.lock"):
                with self.assertRaises(BlockingIOError):
                    with lane_lock(Path(folder)/"lane.lock"):pass

    def test_retry_requires_transport_failure_and_never_accepts_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            log=root/"run.log";api=root/"api-calls.jsonl"
            log.write_text("Full baseline judge failed (HTTPError); strict comparison cannot use a fallback response.")
            api.write_text(json.dumps({"http_status":500})+"\n")
            self.assertTrue(retryable_failure(root))
            api.write_text(json.dumps({"finish_reasons":["length"],"invalid_for_benchmark":True})+"\n")
            self.assertFalse(retryable_failure(root))
            api.write_text(json.dumps({"error_type":"ReadTimeout"})+"\n")
            log.write_text("AttributeError: missing configuration")
            self.assertFalse(retryable_failure(root))

    def test_completed_result_requires_matching_task_order_and_valid_responses(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"trace.jsonl").write_text('{"task_id":"b"}\n{"task_id":"a"}\n')
            (root/"trace.summary.json").write_text("{}")
            (root/"api-calls.jsonl").write_text("")
            self.assertFalse(validate_result(root,0,2,["a","b"])["valid"])
            self.assertTrue(validate_result(root,0,2,["b","a"])["valid"])
            (root/"api-calls.jsonl").write_text('{"invalid_for_benchmark":true}\n')
            self.assertFalse(validate_result(root,0,2,["b","a"])["valid"])


    def test_successful_seed_does_not_skip_the_next_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            first=self.job();first["seed"]=42
            second=json.loads(json.dumps(first).replace("original","second"));second["seed"]=43
            lane={"id":"lane","jobs":[first,second],"credentials":"unused"}
            plan={"recovery_root":str(root)}
            def run(job, env, lock, ids):
                path=Path(job["directory"]);path.mkdir(parents=True)
                job.update(status="completed",exit_code=0)
                (path/"run.json").write_text(json.dumps(job))
                return job
            with patch("tools.run_recovery_plan.environment",return_value={}), \
                 patch("tools.run_recovery_plan.running_directory",return_value=None), \
                 patch("tools.run_recovery_plan.run_job",side_effect=run) as execute:
                state=run_lane(plan,lane)
            self.assertEqual(state["status"],"completed")
            self.assertEqual(execute.call_count,2)
            self.assertNotEqual(state["completed_jobs"]["original"],state["completed_jobs"]["second"])
