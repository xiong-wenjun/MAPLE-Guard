import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from maple_guard import maple_guard_core as ep, run_mmlu, run_longmemeval
from evaluate.defense_methods.full_runtime import runtime_scope
from evaluate.defense_methods.comparison_runtime import ComparisonRuntime

class StrictRunnerPaths(unittest.TestCase):
    def args(self,method,module=run_mmlu):
        with patch.object(sys,"argv",["runner","--config","","--method",method,"--agents","2","--rounds","2","--strict-comparison"]):
            args=module.parse_args()
        args.memory_backend="memory"
        args.attacker_ids=[1]
        args.enable_round_memory_propagation=False
        args.enable_attacker_system_prompt=False
        args._disable_memory_update=True
        args.peer_communication=True
        return args

    def task(self):
        return ep.TaskExample("q","Which fact?",[("A","yes"),("B","no")],"A","B",raw={})

    def test_public_core_runs_each_comparison_arm_without_role_oracle(self):
        for method in ("no_defense_memrl","maple_guard","maple_guard_retrieval_only",
                       "provenance_acl","piguard_retrieval","piguard_lifecycle"):
            with self.subTest(method=method):
                args=self.args(method)
                scored=[]
                guard=SimpleNamespace(score=lambda text:(scored.append(text) or 0.01)) if method.startswith("piguard") else None
                runtime=ComparisonRuntime(args,guard=guard)
                args._full_baseline_runtime=runtime
                m=ep.create_benign_memory(self.task(),0,"A",True,"test")
                runtime.observe_ingress(m,"agent_output","team",0)
                outputs=[]
                def model(base,name,messages,**kwargs):
                    outputs.append(messages)
                    return "Evidence: a normal fact. Final answer: A"
                with patch.object(ep,"remote_embedding",return_value=[1.,0.]),patch.object(ep,"call_chat",side_effect=model):
                    trace,selected,_,_=ep.run_task("test",self.task(),args,{},[m],False)
                self.assertEqual(len(outputs),4)
                self.assertEqual(trace.final_answer_excluded_agent_ids,[])
                self.assertTrue(any(selected.values()))
                self.assertIn("Previous-round messages",str(outputs[2]))
                if method.startswith("piguard"):self.assertTrue(scored)

    def test_real_amem_component_runs_audit_and_lesson_prompt_in_public_core(self):
        from evaluate.defense_methods.amemguard_full import AMemGuardFull
        args=self.args("amemguard_full")
        args.amemguard_experiment_id="offline-contract"
        calls=[]
        def judge(messages):
            calls.append(messages)
            if "**Build Reasoning Chain**" in str(messages):
                return "Question -> uses -> fact"
            return '[{"id":0,"consistent":true,"safe":true}]'
        guard=AMemGuardFull(args,judge,lambda text:[1.,0.])
        runtime=ComparisonRuntime(args,guard=guard)
        args._full_baseline_runtime=runtime
        m=ep.create_benign_memory(self.task(),0,"A",True,"test")
        runtime.observe_ingress(m,"agent_output","team",0)
        prompts=[]
        with patch.object(ep,"call_chat",side_effect=lambda base,name,messages,**kw:(prompts.append(messages) or "Final answer: A")):
            trace,selected,_,_=ep.run_task("test",self.task(),args,{},[m],False)
        self.assertEqual(trace.final_answer,"A")
        self.assertTrue(calls)
        self.assertTrue(any(selected.values()))
        self.assertIn("CRITICAL WARNING",str(prompts))
        self.assertTrue(any("amemguard_consensus" in d["reason"] for d in trace.defense_decisions))

    def test_longmem_scopes_before_amem_topk_and_audit(self):
        args=self.args("amemguard_full",run_longmemeval)
        args.answer_judge=False
        args.rounds=1
        seen=[]
        guard=SimpleNamespace(
            select=lambda q,entries,aid:(seen.append([m.memory_id for m in entries]) or entries[:1],{}),
            lesson_prompt=lambda *a:"",register_entries=lambda *a:None)
        runtime=ComparisonRuntime(args,guard=guard)
        args._full_baseline_runtime=runtime
        current=ep.create_benign_memory(self.task(),0,"A",True,"test")
        old=copy.deepcopy(current);old.memory_id="old";old.origin_task="old"
        current.memory_id="current"
        for m in (old,current):runtime.observe_ingress(m,"agent_output","private",0)
        with patch.object(ep,"call_chat",return_value="Final answer: yes"),patch.object(run_longmemeval,"adjudicate_final_answer",return_value=("yes",{})),patch.object(run_longmemeval,"judge_correctness",return_value=(True,{})):
            trace,selected,*_=run_longmemeval.run_openqa_task(self.task(),args,{0:[old,current]},[],None,{}, {}, {}, {})
        self.assertEqual(seen[0],["current"])
        self.assertEqual([m.memory_id for m in selected[0]],["current"])

if __name__=="__main__": unittest.main()
