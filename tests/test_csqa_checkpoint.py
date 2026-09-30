
import contextlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from maple_guard import infa_memlink_eval as runner, maple_guard_core as ep
from maple_guard.memory_backend import MemoryBackendBundle

class CsqaCheckpointTests(unittest.TestCase):
    def test_resumes_task_boundary_with_metrics_rng_official_state_and_all_journals(self):
        with tempfile.TemporaryDirectory() as temp:
            results=[]
            for interrupted in (False,True):
                root=Path(temp)/str(interrupted);root.mkdir()
                with patch("sys.argv",["runner","--attack-mode","PI","--chat-base-url","http://unused/v1",
                            "--output-root",str(root),"--baseline-state-path",str(root/"state.json"),"--strict-comparison","--agents","1","--rounds","0",
                            "--memory-backend","memrl","--task-checkpoint-dir",str(root/"checkpoints")]):
                    args=runner.parse_args()
                args.memory_store_dir=str(root/"store");Path(args.memory_store_dir).mkdir()
                args.write_final_json=False;args.progress_every=0
                cases=[dict(source_bundle_id=f"case{i}",question=f"Q{i}?\nA. yes\nB. no",
                    correct_answer="A",wrong_answer=["B"],adj_matrix=[[0]],attacker_idxes=[],system_prompts=["system"]) for i in range(3)]
                calls=[]
                def generate(base,model,messages,**kw):
                    idx=int(messages[-1]["content"].split("TASK:")[1].split(":")[0])
                    calls.append(idx)
                    if interrupted and not args.resume_task_checkpoint and idx==1:
                        Path(args.memory_store_dir,"dirty").write_text("uncommitted")
                        runner._append_jsonl(root/"no_defense_memrl.memories.jsonl",{"failed":True})
                        raise TimeoutError("injected")
                    self.assertFalse(Path(args.memory_store_dir,"dirty").exists())
                    return "Reason "+str(random.random())+"\n<ANSWER>: A"
                def first(d,agent,args,rng,method):
                    return "TASK:"+d["source_bundle_id"][-1]+":"+str(rng.random())
                def defense(method,responses,state,**kw):
                    from evaluate.defense_methods.base import OfficialDefenseState
                    state=state or OfficialDefenseState()
                    state.guardian_history.append(kw["task_id"])
                    self.assertEqual(len(state.guardian_history),int(kw["task_id"][-1])+1)
                    return responses,state,[]
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.dict(os.environ,MEMOS_BASE_PATH=str(root/"memos")))
                    for obj,name,fn in [(ep,"create_memory_backend_bundle",lambda args,entry:MemoryBackendBundle(args,entry)),
                        (ep,"call_chat",generate),(runner,"first_prompt",first),
                        (runner,"add_memory_to_prompt",lambda *a:("",[],[],[])),
                        (runner,"commit_round_memory",lambda *a:([],[])),
                        (ep,"apply_official_communication_defense_to_outputs",defense)]:
                        stack.enter_context(patch.object(obj,name,side_effect=fn))
                    random.seed(42)
                    if interrupted:
                        with self.assertRaisesRegex(TimeoutError,"injected"):
                            runner.run_one_method(ep,cases,"no_defense_memrl",args)
                        args.resume_task_checkpoint=True
                    summary=runner.run_one_method(ep,cases,"no_defense_memrl",args)
                self.assertEqual(calls,[0,1,1,2] if interrupted else [0,1,2])
                self.assertEqual(summary["samples"],3)
                for suffix in ["progress","metrics"]:
                    self.assertEqual(len((root/f"no_defense_memrl.{suffix}.jsonl").read_text().splitlines()),3)
                journal=root/"no_defense_memrl.memories.jsonl"
                self.assertFalse(journal.exists() and "failed" in journal.read_text())
                results.append([json.loads(l)["communication_data"] for l in (root/"no_defense_memrl.trace.jsonl").read_text().splitlines()])
            self.assertEqual(results[0],results[1])
