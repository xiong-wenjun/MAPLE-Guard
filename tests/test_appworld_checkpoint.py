import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from maple_guard import run_appworld as runner
from tools import run_appworld_matrix as matrix

@dataclass
class Record:
    task_index: int
    task_id: str
    is_correct: bool = True

class AppWorldCheckpointTests(unittest.TestCase):
    def test_cli_exposes_explicit_checkpoint_and_resume_flags(self):
        with patch("sys.argv", ["runner", "--method", "provenance_acl",
                               "--task-checkpoint-dir", "/tmp/checkpoints", "--resume-task-checkpoint"]):
            args = runner.parse_args()
        self.assertEqual(args.task_checkpoint_dir, "/tmp/checkpoints")
        self.assertTrue(args.resume_task_checkpoint)

    def test_matrix_checkpoint_flag_supports_acl_and_maple(self):
        with patch("sys.argv", ["matrix", "--bundle", "/tmp/tasks", "--services", "/tmp/services",
                               "--run-root", "/tmp/run", "--task-checkpoints"]):
            args = matrix.arguments()
        args.task_service = "q"; args.judge_service = "q"
        services = {"q": {"base_url": "http://q/v1", "model": "Qwen"}, "embedding": {"base_url": "http://e/v1", "model": "embed"}}
        job = matrix.build_job(args, "provenance_acl", 42, services)
        self.assertIn("--task-checkpoint-dir", job["command"])
        self.assertIn("--task-checkpoint-dir", matrix.build_job(args, "maple_guard", 42, services)["command"])
        args.task_checkpoints = False
        self.assertNotIn("--task-checkpoint-dir", matrix.build_job(args, "provenance_acl", 42, services)["command"])

    def run_main(self, folder, *, resume=False, initial=None, fail_at=None):
        folder = Path(folder)
        out = folder / "trace.jsonl"
        args = SimpleNamespace(seed=42, tasks=3, method="provenance_acl",
            task_checkpoint_dir=str(folder/"checkpoints"), resume_task_checkpoint=resume,
            dataset="", benchmark_bundle=str(folder/"bundle.json"), appworld_split="test_normal",
            out=str(out), trace_id="test", attack_variant="explicit", attacker_ids=[0],
            communication_topology="star", communication_sparsity=None, agents=2, log_every=0)
        tasks = [SimpleNamespace(task_id=str(i)) for i in range(3)]
        bundle = SimpleNamespace(private_memories={0: [], 1: []}, shared_memories=[])
        saved, calls, summaries = [], [], []
        checkpoint = types.ModuleType("maple_guard.task_checkpoint")
        checkpoint.recover_trace_id = lambda args: None
        checkpoint.run_lock = lambda actual_args: contextlib.nullcontext()
        checkpoint.load_checkpoint = lambda actual_args: {"stream_state": initial}
        checkpoint.restore_bundle = lambda *a: None
        def save(actual_args, actual_bundle, state, trace_path):
            saved.append((state, Path(trace_path).read_text()))
        checkpoint.checkpoint_summary = lambda args: {}
        checkpoint.save_checkpoint = save
        def task(trace_id, idx, task, poison, *state):
            calls.append(idx)
            if idx == fail_at:
                raise RuntimeError("interrupted task")
            return Record(idx, task.task_id)
        def summarize(records, *a):
            summaries.append([r.task_index for r in records])
            return {}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"maple_guard.task_checkpoint": checkpoint}))
            stack.enter_context(patch.object(runner, "parse_args", return_value=args))
            stack.enter_context(patch.object(runner.stream, "resolve_args", side_effect=lambda a:a))
            stack.enter_context(patch.object(runner, "ensure_appworld_index", return_value={"total_cases":3}))
            stack.enter_context(patch.object(runner, "select_cases", return_value=tasks))
            stack.enter_context(patch.object(runner, "task_from_case", side_effect=lambda c:c))
            stack.enter_context(patch.object(runner.stream, "choose_poison_indices", return_value={2}))
            stack.enter_context(patch.object(runner.ep, "create_memory_backend_bundle", return_value=bundle))
            stack.enter_context(patch.object(runner.stream, "StreamTaskRecord", Record))
            stack.enter_context(patch.object(runner.stream, "run_stream_task", side_effect=task))
            stack.enter_context(patch.object(runner.stream, "update_agent_trust", side_effect=lambda *a:None))
            stack.enter_context(patch.object(runner.stream, "summarize_stream", side_effect=summarize))
            stack.enter_context(patch.object(runner.stream, "dump_text_memory", return_value="memories"))
            stack.enter_context(patch.object(runner.stream, "log_progress", side_effect=lambda *a:None))
            stack.enter_context(patch.object(runner.ep, "baseline_run_provenance", return_value={}))
            stack.enter_context(patch.object(runner, "add_appworld_summary", side_effect=lambda *a:None))
            try:
                runner.main()
            except BaseException as exc:
                return saved, calls, summaries, exc
        return saved, calls, summaries, None

    def initial(self):
        return {"next_task_index":1, "task_ids":["0","1","2"], "poison_indices":[2],
            "records":[asdict(Record(0,"0"))], "poison_targets":{}, "poison_target_texts":{},
            "poison_pattern_texts":{}, "poison_origins":{}, "agent_trust":{"0":0.6,"1":0.6},
            "args_poison_target_cache":{}, "args_poison_target_reason_cache":{}}

    def test_resume_skips_committed_prefix_and_summary_keeps_it(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/"trace.jsonl"
            prefix = json.dumps(asdict(Record(0,"0")))+"\n"
            out.write_text(prefix)
            saved, calls, summaries, error = self.run_main(folder, resume=True, initial=self.initial())
            self.assertIsNone(error)
            self.assertEqual(calls,[1,2])
            self.assertEqual(summaries,[[0,1,2]])
            self.assertTrue(out.read_text().startswith(prefix))
            self.assertEqual([s["next_task_index"] for s,_ in saved],[2,3])

    def test_failed_task_does_not_advance_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            saved,calls,_,error=self.run_main(folder,fail_at=1)
            self.assertIsInstance(error,RuntimeError)
            self.assertEqual(calls,[0,1])
            self.assertEqual([s["next_task_index"] for s,_ in saved],[0,1])
            self.assertEqual([len(text.splitlines()) for _,text in saved],[0,1])

    def test_fresh_checkpoint_run_refuses_existing_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/"trace.jsonl";out.write_text("original evidence\n")
            _,calls,_,error=self.run_main(folder)
            self.assertIsInstance(error,ValueError)
            self.assertEqual(calls,[])
            self.assertEqual(out.read_text(),"original evidence\n")

    def test_mismatched_task_order_stops_before_any_task(self):
        with tempfile.TemporaryDirectory() as folder:
            initial=self.initial();initial["task_ids"]=["wrong","1","2"]
            (Path(folder)/"trace.jsonl").write_text(json.dumps(asdict(Record(0,"0")))+"\n")
            _,calls,_,error=self.run_main(folder,resume=True,initial=initial)
            self.assertIsInstance(error,ValueError)
            self.assertEqual(calls,[])

if __name__=="__main__":unittest.main()
