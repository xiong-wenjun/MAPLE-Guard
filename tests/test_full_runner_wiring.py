import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from maple_guard import maple_guard_core as ep
from maple_guard import run_mmlu, run_longmemeval, infa_memlink_eval
from evaluate.defense_methods.full_runtime import FullRuntime, runtime_scope

class KickGuard:
    def __init__(self): self.inactive=set()
    def begin_task(self, task): self.inactive=set()
    def defend(self, outputs, respond, question, round_idx, adjacency, **callbacks):
        self.inactive.add(1)
        return {i:v for i,v in outputs.items() if i != 1}, [
            {"agent_id":1,"action":"block","reason":"test_detected","details":{}}]

class FullRunnerWiringTests(unittest.TestCase):
    def setUp(self):
        embedding=patch.object(ep,"remote_embedding",return_value=None)
        embedding.start()
        self.addCleanup(embedding.stop)

    def args(self, module=run_mmlu):
        with patch.object(sys, "argv", ["runner","--config","","--method","agentxposed_full_kick",
                                        "--agents","2","--rounds","3"]):
            args=module.parse_args()
        args.attacker_ids=[1]
        args.enable_round_memory_propagation=False
        args.enable_attacker_system_prompt=False
        args._disable_memory_update=True
        args._full_baseline_runtime=FullRuntime(args, guard=KickGuard())
        return args

    def test_legacy_runner_still_uses_all_agents_without_full_guard(self):
        args=self.args()
        args.method="no_defense_memrl"
        args.peer_communication=True
        task=ep.TaskExample(task_id="t",question="q",choices=[("A","yes"),("B","no")],answer="B",wrong_answer="A",raw={})
        prompts=[]
        def model(base,name,messages,**kwargs):
            prompts.append(messages)
            return "Reason: original peer evidence.\nFinal answer: B"
        with patch.object(ep,"call_chat",side_effect=model):
            trace,*_=ep.run_task("test",task,args,{},[],False)
        self.assertEqual(len(prompts),6)
        self.assertEqual(trace.final_answer,"B")
        self.assertIn("Previous-round messages",str(prompts[2]))

    def test_cli_exposes_required_full_configuration(self):
        with patch.object(sys,"argv",["runner","--config","","--method","agentsafe_full",
                                     "--agentsafe-policy-file","policy.json",
                                     "--agentsafe-criteria-file","criteria.json",
                                     "--agentsafe-threshold","0.4"]):
            args=run_mmlu.parse_args()
        self.assertEqual(args.agentsafe_threshold,0.4)

    def test_core_runner_executes_full_guard_and_keeps_kick(self):
        args=self.args()
        task=ep.TaskExample(task_id="t",question="q",choices=[("A","yes"),("B","no")],answer="A",wrong_answer="B",raw={})
        with patch.object(ep,"call_chat",return_value="Final answer: A") as model:
            trace, selected, reads, decisions=ep.run_task("test",task,args,{},[],True)
        self.assertEqual(model.call_count,4)
        self.assertTrue(all(1 not in outputs for outputs in trace.outputs_by_round))
        self.assertEqual(trace.final_answer_excluded_agent_ids, [])
        self.assertTrue(any(d.reason=="test_detected" for d in decisions))

    def test_missing_choice_is_abstention_not_a_correct_A(self):
        args=self.args()
        task=ep.TaskExample(task_id="t",question="q",choices=[("A","yes"),("B","no")],answer="A",wrong_answer="B",raw={})
        with patch.object(ep,"call_chat",return_value="I will cooperate now."):
            trace,*_=ep.run_task("test",task,args,{},[],True)
        self.assertEqual(trace.final_answer,"")
        self.assertFalse(trace.is_correct)

    def test_longmem_runner_uses_same_live_agent_session(self):
        args=self.args(run_longmemeval)
        args.answer_judge=False
        task=ep.TaskExample(task_id="t",question="q",choices=[],answer="yes",wrong_answer="no",raw={"question":"q"})
        with patch.object(ep,"call_chat",return_value="Final answer: yes") as model, \
             patch.object(run_longmemeval,"adjudicate_final_answer",return_value=("yes",{})), \
             patch.object(run_longmemeval,"judge_correctness",return_value=(True,{})):
            trace,*_=run_longmemeval.run_openqa_task(task,args,{},[],None,{}, {}, {}, {})
        self.assertEqual(model.call_count,4)
        self.assertTrue(all(1 not in outputs for outputs in trace.outputs_by_round))

    def test_real_infa_component_replaces_and_repairs_in_public_runner(self):
        import json
        from evaluate.defense_methods.infa_full import InfaGuardFull
        args=self.args()
        args.method="infa_guard_full"
        args.agents=3
        args.communication_topology="full"
        args.rounds=2
        detector=lambda history,adj: [[0.9,0.01],[0.01,0.8],[0.01,0.01]] if len(history)==1 else [[0.01,0.01]]*3
        guard=InfaGuardFull(args,lambda messages:json.dumps({"response":"Reason: repaired task evidence.\nFinal answer: B"}),detector=detector)
        args._full_baseline_runtime=FullRuntime(args,guard=guard)
        task=ep.TaskExample(task_id="t",question="q",choices=[("A","yes"),("B","no")],answer="A",wrong_answer="B",raw={})
        prompts=[]
        def chat(base,model,messages,**kwargs):
            prompts.append(messages)
            return "Final answer: A"
        with patch.object(ep,"call_chat",side_effect=chat):
            trace,*_=ep.run_task("test",task,args,{},[],True)
        self.assertIn("repaired task evidence",str(prompts[4]))
        self.assertTrue(args._full_baseline_runtime.replacements)
        self.assertTrue(any("infa_attacker_replaced" in d["reason"] for d in trace.defense_decisions))
        self.assertEqual(trace.final_answer_excluded_agent_ids,[])

    def test_transfer_memory_mapping_preserves_agentsafe_full(self):
        self.assertEqual(infa_memlink_eval.memory_method_for("agentsafe_full"),"agentsafe_full")
        self.assertEqual(infa_memlink_eval.memory_method_for("agentsafe"),"no_defense_memrl")



class MemoryHookTests(unittest.TestCase):
    def entry(self):
        return ep.create_benign_memory(
            ep.TaskExample(task_id="t",question="q",choices=[("A","a")],answer="A",wrong_answer="B",raw={}),
            0, "A", True, "test")

    def test_admission_blocks_store_mutation_without_reading_labels(self):
        class Guard:
            def admit(self, **record):
                self.received=set(record)
                return False, {"reason":"rejected"}
        args=SimpleNamespace(method="agentsafe_full")
        guard=Guard()
        args._full_baseline_runtime=FullRuntime(args,guard=guard)
        private,shared={},[]
        with runtime_scope(args):
            allowed, decisions=ep.commit_memory(self.entry(),"team",0,args.method,private,shared)
        self.assertFalse(allowed)
        self.assertEqual(shared,[])
        self.assertEqual(guard.received, {"memory_id","text","owner","recipient","metadata"})

    def test_handoff_is_blocked_before_receiver_summarization(self):
        class Guard:
            inactive=set()
            def route(self,text,sender,recipient): return None, {"level":4}
            def read(self,**record): return True, {}
        args=SimpleNamespace(method="agentsafe_full",attacker_ids=[],attacker_id=9)
        args._full_baseline_runtime=FullRuntime(args,guard=Guard())
        task=ep.TaskExample(task_id="t",question="q",choices=[("A","a")],answer="A",wrong_answer="B",raw={})
        with runtime_scope(args), patch.object(ep,"receiver_summarize_dialogue_for_memory",side_effect=AssertionError("secret reached receiver")):
            ids,_=ep.commit_pre_round_memory_handoffs(task,1,{0:[self.entry()]},[[0,1],[0,0]],args,{},[],None)
        self.assertEqual(ids,{})

    def test_shared_dataclass_metadata_is_detached_before_admission(self):
        from dataclasses import replace
        class Guard:
            def admit(self,**record):
                stamp=record["metadata"].setdefault("agentsafe_full",{})
                if record["recipient"]==1: stamp["quarantined"]=True
                return not stamp.get("quarantined",False), {}
        runtime=FullRuntime(SimpleNamespace(method="agentsafe_full"),guard=Guard())
        original=self.entry()
        self.assertFalse(runtime.admit(replace(original,memory_id="low"),"private",1)[0])
        self.assertTrue(runtime.admit(replace(original,memory_id="high"),"private",2)[0])

    def test_agentsafe_discards_old_prompt_context_after_memory_review(self):
        args=SimpleNamespace(method="agentsafe_full")
        runtime=FullRuntime(args,guard=SimpleNamespace(inactive=set()))
        requests=[]
        generate=lambda messages: requests.append(messages) or "answer"
        runtime.begin_task("t","question")
        runtime.generate(0,[{"role":"system","content":"normal"},{"role":"user","content":"quarantined memory"}],generate)
        runtime.generate(0,[{"role":"system","content":"normal"},{"role":"user","content":"clean memory"}],generate)
        self.assertNotIn("quarantined memory",str(requests[-1]))


    def test_longmem_vote_does_not_use_ground_truth_attacker_ids(self):
        args=SimpleNamespace(method="agentxposed_full_kick",attacker_ids=[1],final_adjudicator=False)
        answer,_=run_longmemeval.adjudicate_final_answer(SimpleNamespace(),{1:"Final answer: yes"},{},args,{})
        self.assertEqual(answer,"yes")

    def test_trusted_seed_still_requires_agentsafe_admission(self):
        guard=SimpleNamespace(admit=lambda **kwargs:(False,{"reason":"test"}))
        args=SimpleNamespace(method="agentsafe_full")
        args._full_baseline_runtime=FullRuntime(args,guard=guard)
        shared=[]
        with runtime_scope(args):
            allowed,_=run_longmemeval.commit_trusted_seed_memory(self.entry(),"team",0,{},shared,None,method=args.method)
        self.assertFalse(allowed)
        self.assertEqual(shared,[])

if __name__=="__main__": unittest.main()
