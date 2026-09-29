import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from maple_guard import infa_memlink_eval as runner, maple_guard_core as ep
from evaluate.defense_methods.full_runtime import runtime_scope

class CsqaBundleWiringTests(unittest.TestCase):
    def args(self, **values):
        defaults=dict(benchmark_bundle="", attack_mode="PI", dataset="csqa", samples=3,
                      agents=8, num_attackers=3, attacker_ids="0,1,2", graph_type="star",
                      num_graphs=1, seed=42, attacker_seed=42, infa_root="/not-used")
        defaults.update(values)
        return SimpleNamespace(**defaults)

    def bundle(self, root):
        rows=[]
        for n,answer in enumerate(("A","E","C")):
            rows.append(dict(benchmark="commonsenseqa",task_id=f"commonsenseqa:dev:{n}",split="dev",
                             data=dict(id=str(n),answerKey=answer,question=dict(stem=f"Question {n}?",
                              choices=[dict(label=lab,text=f"Option {lab}") for lab in "ABCDE"]))))
        path=Path(root)/"bundle.json"
        path.write_text(json.dumps(dict(benchmark="commonsenseqa",task_count=3,tasks=rows)))
        return path,rows

    def prompt_modules(self):
        package=ModuleType("MAS");package.__path__=[]
        prompts=ModuleType("MAS.agent_prompts")
        prompts.PI_SYS_PROMPT="Normal agent {}"
        prompts.PI_ATTACKER_SYS_PROMPT="Attacker agent {}"
        return {"MAS":package,"MAS.agent_prompts":prompts}

    def test_cli_accepts_frozen_bundle_and_released_guardian(self):
        argv=["runner","--attack-mode","PI","--chat-base-url","http://example/v1",
              "--benchmark-bundle","bundle.json","--official-defense-guardian-bert-dir","bert",
              "--official-defense-guardian-profile","released_detector","--official-defense-guardian-epochs","20"]
        with patch.object(sys,"argv",argv),contextlib.redirect_stderr(io.StringIO()):
            try: args=runner.parse_args()
            except SystemExit: self.fail("CSQA CLI does not accept bundle/released GUARDIAN settings")
        self.assertEqual(args.benchmark_bundle,"bundle.json")
        self.assertEqual(args.official_defense_guardian_profile,"released_detector")
        self.assertEqual(args.official_defense_guardian_epochs,20)

    def test_bundle_preserves_five_choices_ids_and_paired_order(self):
        self.assertTrue(hasattr(runner,"load_csqa_bundle_cases"),"frozen CSQA bundle loader missing")
        with tempfile.TemporaryDirectory() as root,patch.dict(sys.modules,self.prompt_modules()):
            path,rows=self.bundle(root)
            orders=[]
            for topo in ("star","chain","tree"):
                args=self.args(benchmark_bundle=str(path),graph_type=topo)
                cases=runner.load_infa_cases(args)
                orders.append([c["source_bundle_id"] for c in cases])
                self.assertEqual(orders[-1], [c["source_bundle_id"] for c in runner.load_infa_cases(args)])
                self.assertEqual(set(orders[-1]), {r["task_id"] for r in rows})
                for case in cases:
                    row=next(r for r in rows if r["task_id"]==case["source_bundle_id"])
                    self.assertEqual(case["question"],row["data"]["question"]["stem"]+"\n"+"\n".join(f"{c['label']}. {c['text']}" for c in row["data"]["question"]["choices"]))
                    self.assertEqual(case["correct_answer"],row["data"]["answerKey"])
                    self.assertEqual(case["wrong_answer"],[x for x in "ABCDE" if x!=case["correct_answer"]])
                    self.assertEqual(case["adj_matrix"],ep.build_adj_matrix(topo,8,42))
                    self.assertEqual(case["attacker_idxes"],[0,1,2])
                    self.assertEqual(case["system_prompts"][3],"Normal agent 3")
            self.assertEqual(orders[0],orders[1]); self.assertEqual(orders[0],orders[2])

    def test_bundle_rejects_wrong_protocol_duplicate_graphs_and_oversampling(self):
        self.assertTrue(hasattr(runner,"load_csqa_bundle_cases"),"frozen CSQA bundle loader missing")
        with tempfile.TemporaryDirectory() as root,patch.dict(sys.modules,self.prompt_modules()):
            path,_=self.bundle(root)
            for fields in ({"attack_mode":"TA"},{"num_graphs":2},{"samples":4},{"attacker_ids":"0,0,1"}):
                with self.subTest(fields=fields),self.assertRaises(ValueError):
                    runner.load_csqa_bundle_cases(self.args(benchmark_bundle=str(path),**fields))

    def test_trace_task_identity_uses_original_bundle_id(self):
        args=self.args()
        case=dict(source_bundle_id="commonsenseqa:dev:native",question="Q?\nA. yes\nB. no\nC. c\nD. d\nE. e",
                  correct_answer="E",wrong_answer=["A","B","C","D"])
        task=runner.task_for_record(ep,case,0,args)
        self.assertEqual(task.task_id,case["source_bundle_id"])
        self.assertEqual(task.answer,"E")

    def test_strict_communication_methods_keep_operational_memory_runtime(self):
        for method in ("gsafeguard","guardian"):
            with self.subTest(method=method),runtime_scope(SimpleNamespace(method=method,agents=8,strict_comparison=True)):
                self.assertEqual(runner.memory_method_for(method),method)
        self.assertEqual(runner.memory_method_for("gsafeguard"),"no_defense_memrl")

    def test_instrumentation_accepts_transfer_runner(self):
        from tools import run_instrumented
        with patch.object(sys,"argv",["run_instrumented.py","maple_guard.infa_memlink_eval","--help"]), \
             patch.dict("os.environ",MAPLE_CALL_LOG="unused"), \
             patch.object(run_instrumented,"install_metrics"), \
             patch.object(run_instrumented.runpy,"run_module") as run:
            try: run_instrumented.main()
            except ValueError: self.fail("PromptInject runner lacks strict response instrumentation")
        run.assert_called_once_with("maple_guard.infa_memlink_eval",run_name="__main__")

    def test_comparison_runner_preserves_actual_conversation_history(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(sys,"argv",["runner","--attack-mode","PI","--chat-base-url","http://unused/v1",
                                         "--output-root",root,"--strict-comparison","--agents","1","--rounds","1"]):
                args=runner.parse_args()
            args.write_final_json=False;args.stream_memories=False;args.progress_every=0
            case=dict(source_bundle_id="commonsenseqa:dev:test",question="Q?\nA. yes\nB. no",
                      correct_answer="A",wrong_answer=["B"],adj_matrix=[[0]],attacker_idxes=[],
                      system_prompts=["system"])
            calls=[]
            def generate(base,model,messages,**kwargs):
                calls.append(list(messages))
                return "Reason: retained evidence\n<ANSWER>: A"
            with patch.object(runner,"first_prompt",return_value="initial question"), \
                 patch.object(runner,"regen_prompt",return_value="follow-up peer context"), \
                 patch.object(ep,"call_chat",side_effect=generate), \
                 patch.object(ep,"remote_embedding",return_value=[1.,0.]):
                summary=runner.run_one_method(ep,[case],"no_defense_memrl",args)
            self.assertEqual(summary["samples"],1)
            self.assertEqual(len(calls),2)
            self.assertTrue(any(m["role"]=="assistant" and "retained evidence" in m["content"] for m in calls[1]),
                            "Strict comparison discarded the previous conversation turn")
