import contextlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
from dataclasses import dataclass, asdict
import unittest
from unittest.mock import patch

from maple_guard import run_mmlu, run_longmemeval, infa_memlink_eval
from maple_guard.memory_backend import MemoryBackendBundle
from maple_guard import maple_guard_core as ep

@dataclass
class Record:
    task_index: int
    task_id: str
    value: float

class BenchmarkCheckpointTests(unittest.TestCase):
    def test_each_public_cli_exposes_checkpoint_resume_and_budget_policy(self):
        for module in (run_mmlu,run_longmemeval,infa_memlink_eval):
            with self.subTest(module=module.__name__), patch("sys.argv",["runner",
                    "--task-checkpoint-dir","/tmp/checkpoints","--resume-task-checkpoint",
                    "--checkpoint-allow-budget-change"] + (["--attack-mode","PI","--chat-base-url","http://unused/v1"] if module is infa_memlink_eval else [])):
                args=module.parse_args()
                self.assertEqual(args.task_checkpoint_dir,"/tmp/checkpoints")
                self.assertTrue(args.resume_task_checkpoint)
                self.assertTrue(args.checkpoint_allow_budget_change)

    def test_mmlu_interrupted_resume_matches_uninterrupted_state_and_reuses_prefix(self):
        self._equivalent(run_mmlu)

    def test_longmemeval_interrupted_resume_matches_uninterrupted_state(self):
        self._equivalent(run_longmemeval)

    def _equivalent(self,module):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            tasks=[SimpleNamespace(task_id="task-"+str(i)) for i in range(3)]
            outputs=[]
            for interrupted in (False,True):
                folder=root/str(interrupted);folder.mkdir()
                args=SimpleNamespace(seed=42,tasks=3,method="maple_guard",task_mode="qa",
                    memory_backend="memrl",memory_store_dir=str(folder/"store"),memory_run_id="test",
                    out=str(folder/"trace.jsonl"),task_checkpoint_dir=str(folder/"checkpoints"),
                    resume_task_checkpoint=False,checkpoint_allow_budget_change=False,
                    trace_id="",dataset="",agents=2,log_every=0,attack_capability="dmi",
                    attack_variant="explicit",attacker_ids=[0],communication_topology="chain",
                    communication_sparsity=None,stream_protocol="scheduled",warmup_tasks=0,
                    memory_topology="brokered-shared",dry_run=False,prompt_file="",attack_strength="standard")
                for key,value in {"attack_stealth_mode":"none","poison_payload":"","malicious_activation_rate":0.1,"poison_selection_mode":"random","poison_target_population":"all","attacker_id":0,"target_agent_id":1,"poison_target_scope":"all","stream_builder":"random","retrieval_mode":"topk"}.items(): setattr(args,key,value)
                Path(args.memory_store_dir).mkdir()
                called=[]
                def bundle(*unused):
                    return MemoryBackendBundle(args,ep.MemoryEntry)
                def task(trace_id,idx,task,poison,*rest):
                    called.append(idx)
                    if interrupted and not args.resume_task_checkpoint and idx==1:
                        Path(args.memory_store_dir,"partial").write_text("failed task")
                        random.random()
                        raise RuntimeError("injected failure")
                    self.assertFalse(Path(args.memory_store_dir,"partial").exists())
                    return Record(idx,task.task_id,random.random())
                def summary(records,*unused):return {"records":[asdict(r) for r in records]}
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.dict(os.environ,{"MEMOS_BASE_PATH":str(folder/"memos")}))
                    for obj,name,value in [(module,"parse_args",lambda:args),(module,"resolve_args",lambda a:a),
                                          (module,"run_stream_task",task),(module,"log_progress",lambda *a:None),
                                          (module,"dump_text_memory",lambda *a:"unused"),
                                          (module.ep,"create_memory_backend_bundle",bundle),
                                          (module.ep,"baseline_run_provenance",lambda *a:{})]:
                        stack.enter_context(patch.object(obj,name,side_effect=value))
                    if module is run_mmlu:
                        stack.enter_context(patch.object(module.ep,"load_dataset",return_value=tasks))
                        stack.enter_context(patch.object(module.ep,"normalize_example",side_effect=lambda row,i:row))
                        stack.enter_context(patch.object(module,"build_task_stream",return_value=tasks))
                        stack.enter_context(patch.object(module,"choose_poison_indices",return_value={2}))
                        stack.enter_context(patch.object(module,"update_agent_trust",side_effect=lambda *a:None))
                        stack.enter_context(patch.object(module,"StreamTaskRecord",Record))
                        stack.enter_context(patch.object(module,"summarize_stream",side_effect=summary))
                    else:
                        stack.enter_context(patch.object(module,"load_longmemeval",return_value=tasks))
                        stack.enter_context(patch.object(module,"load_prompt_bundle",return_value={}))
                        stack.enter_context(patch.object(module,"LongMemTaskRecord",Record))
                        stack.enter_context(patch.object(module,"summarize",side_effect=summary))
                    random.seed(42)
                    if interrupted:
                        with self.assertRaisesRegex(RuntimeError,"injected failure"):module.main()
                        first=Path(args.out).read_bytes()
                        args.resume_task_checkpoint=True
                        args.trace_id=""
                        module.main()
                        self.assertTrue(Path(args.out).read_bytes().startswith(first))
                        self.assertEqual(called,[0,1,1,2])
                    else:module.main()
                outputs.append(json.loads(Path(args.out.replace(".jsonl",".summary.json")).read_text())["records"])
            self.assertEqual(outputs[0],outputs[1])
