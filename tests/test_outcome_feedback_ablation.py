
import copy
import inspect
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from maple_guard import maple_guard_core as ep, run_mmlu
from evaluate.defense_methods.full_runtime import runtime_scope, experiment_identity

class FeedbackAblationTests(unittest.TestCase):
    def args(self, policy="full", **kwargs):
        values=dict(method="maple_guard", agents=1, rounds=1, seed=42,
                    strict_comparison=True, peer_communication=False,
                    communication_topology="star", memory_topology=ep.MEM_BROKERED_SHARED,
                    retrieval_mode="once", top_k_memory=3, min_retrieval_score=-.5,
                    chat_base_url="", chat_model="", embed_base_url="", embed_model="",
                    outcome_feedback_policy=policy, attacker_ids=[], attacker_id=0,
                    disable_private_memory=False, disable_chat_thinking=True,
                    benign_shared_promotion_policy=run_mmlu.PROMOTION_RETRIEVED_PRIVATE,
                    promote_benign_to_shared=True, benign_shared_promotion_agents="one",
                    benign_shared_min_agent_trust=0.)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def task(self, answer="A"):
        return ep.TaskExample("t","Choose an action",[("A","one"),("B","two")],answer,"B" if answer=="A" else "A",raw={})

    def used_memory(self):
        m=ep.create_benign_memory(self.task(),0,"A",True,"test")
        m.embedding=[1.,0.]
        return m

    def run_used_memory(self,policy,success=True):
        args=self.args(policy)
        task=self.task("A" if success else "B")
        m=self.used_memory()
        backend=Mock()
        with runtime_scope(args):
            rt=ep.current_runtime(args.method)
            rt.observe_ingress(m,"agent_output","private",0)
            m.utility_q=.4
            rt.record_feedback(m)
            before=tuple(getattr(m,k) for k in ("utility_q","success_count","failure_count","provenance_trust","content_hazard"))
            with patch.object(ep,"retrieve_for_agent",return_value=([m],[],[])),patch.object(ep,"call_chat",return_value="Reason: a choice.\nFinal answer: A"):
                ep.run_task("test",task,args,{0:[m]},[],False,backend)
        return m,before,backend

    def test_no_q_preserves_positive_and_negative_trust_feedback(self):
        for success in (True,False):
            with self.subTest(success=success):
                m,before,backend=self.run_used_memory("no_q",success)
                self.assertEqual(m.utility_q,before[0],"Q changed in no_q ablation")
                self.assertEqual(m.success_count,1 if success else 0)
                self.assertEqual(m.failure_count,0 if success else 1)
                self.assertNotEqual(m.provenance_trust,before[3])
                backend.update_value.assert_not_called()

    def test_none_prevents_all_outcome_changes_and_backend_rewards(self):
        for success in (True,False):
            with self.subTest(success=success):
                m,before,backend=self.run_used_memory("none",success)
                self.assertEqual(tuple(getattr(m,k) for k in ("utility_q","success_count","failure_count","provenance_trust","content_hazard")),before,
                                 "Outcome reached memory state in no-feedback ablation")
                backend.update_value.assert_not_called()

    def test_full_preserves_existing_feedback_behavior(self):
        m,before,backend=self.run_used_memory("full")
        self.assertGreater(m.utility_q,before[0])
        self.assertEqual(m.success_count,1)
        backend.update_value.assert_called_once_with(m,True)

    def test_no_feedback_memory_does_not_depend_on_reference_or_correctness(self):
        self.assertIn("feedback_policy",inspect.signature(ep.create_benign_memory).parameters,
                      "Memory constructor has no feedback isolation")
        a=ep.create_benign_memory(self.task("A"),0,"A",True,"same",agent_output="Final answer: A",feedback_policy="none")
        b=ep.create_benign_memory(self.task("B"),0,"A",False,"same",agent_output="Final answer: A",feedback_policy="none")
        self.assertEqual(vars(a),vars(b))
        self.assertNotIn("verified correct answer",a.experience.lower())
        self.assertNotIn("Outcome: correct",a.experience)
        self.assertNotIn("Outcome: incorrect",a.experience)
        self.assertEqual(a.utility_q,0.)

    def test_no_feedback_constructor_never_reads_reference(self):
        self.assertIn("feedback_policy",inspect.signature(ep.create_benign_memory).parameters)
        class HiddenReference:
            task_id="hidden";question="Choose";choices=[("A","one"),("B","two")];raw={}
            @property
            def answer(self):raise AssertionError("Reference answer accessed by no-feedback memory")
        ep.create_benign_memory(HiddenReference(),0,"A",False,"same",agent_output="Final answer: A",feedback_policy="none")

    def test_no_feedback_does_not_update_agent_reputation(self):
        args=self.args("none"); trusts={0:.5}
        run_mmlu.update_agent_trust(trusts,SimpleNamespace(is_correct=False),args)
        self.assertEqual(trusts,{0:.5})

    def test_no_q_still_updates_agent_reputation(self):
        trusts={0:.5}
        run_mmlu.update_agent_trust(trusts,SimpleNamespace(is_correct=False),self.args("no_q"))
        self.assertEqual(trusts,{0:.48})

    def promote(self,policy,success):
        args=self.args(policy)
        m=self.used_memory(); captured=[]
        trace=ep.TaskRunTrace(phase="test",task_id="t",final_answer="A",correct_answer="A",wrong_answer="B",is_correct=success,
                             outputs_by_round=[{0:"Final answer: A"}],selected_memory_ids={"0":[m.memory_id]})
        with runtime_scope(args),patch.object(ep,"commit_memory",side_effect=lambda entry,*a,**kw:(captured.append(entry) or (True,[]))):
            run_mmlu.maybe_promote_benign_to_shared(self.task(),"test","A",success,args,{0:[m]},[],None,{0:.5},trace)
        return captured

    def test_no_feedback_promotion_eligibility_does_not_use_gold_success(self):
        a=self.promote("none",True); b=self.promote("none",False)
        self.assertEqual(len(a),1)
        self.assertEqual([vars(x) for x in a],[vars(x) for x in b],
                         "Reference correctness leaked into promotion eligibility")

    def test_full_and_no_q_keep_success_condition_for_promotion(self):
        for policy in ("full","no_q"):
            self.assertEqual(len(self.promote(policy,True)),1)
            self.assertEqual(self.promote(policy,False),[])

    def test_policies_cannot_share_runtime_or_checkpoint_identity(self):
        ids={experiment_identity(self.args(p)) for p in ("full","no_q","none")}
        self.assertEqual(len(ids),3,"Feedback policy missing from state identity")

    def test_cli_exposes_policy_and_default_is_full(self):
        with patch.object(sys,"argv",["run_mmlu","--config",""]):
            args=run_mmlu.parse_args()
        self.assertEqual(getattr(args,"outcome_feedback_policy",None),"full")
        with patch.object(sys,"argv",["run_mmlu","--config","","--outcome-feedback-policy","none"]):
            args=run_mmlu.parse_args()
        self.assertEqual(args.outcome_feedback_policy,"none")


    def test_stream_no_feedback_keeps_metrics_without_leaking_to_memory(self):
        self.assertIn("feedback_policy",inspect.signature(ep.create_benign_memory).parameters)
        memories=[]
        for answer in ("A","B"):
            task=self.task(answer)
            args=self.args("none",attack_capability="dmi",attack_variant="explicit",poison_payload="pep",target_agent_id=0,
                           attack_query_activation="none",poison_target_strategy="first_wrong",
                           promote_benign_to_shared=False,benign_shared_promotion_rate=0.)
            trace=ep.TaskRunTrace(phase="test",task_id="t",final_answer="A",correct_answer=answer,wrong_answer=task.wrong_answer,
                                 is_correct=answer=="A",outputs_by_round=[{0:"Final answer: A"}],selected_memory_ids={})
            private={0:[]}
            with runtime_scope(args),patch.object(ep,"run_task",return_value=(trace,{},[],[])):
                record=run_mmlu.run_stream_task("trace",0,task,False,args,private,[],None,{}, {}, {}, {},{0:.5})
            self.assertEqual(record.is_correct,answer=="A")
            self.assertEqual(record.benign_correct_by_round["1"],int(answer=="A"))
            self.assertEqual(getattr(record,"outcome_feedback_policy",None),"none")
            self.assertEqual(len(private[0]),1)
            memories.append(vars(private[0][0]))
        self.assertEqual(memories[0],memories[1],"Stream consolidation leaked the reference outcome")

    def test_backend_cannot_apply_rewards_in_frozen_policies(self):
        from maple_guard.memory_backend import MemRLMemoryBackend
        for policy in ("no_q","none"):
            backend=MemRLMemoryBackend(name="test",user_id="test",args=self.args(policy),entry_cls=ep.MemoryEntry)
            m=self.used_memory(); backend.backend_ids[m.memory_id]="backend-id"
            backend._service=Mock(); backend._service.update_value.return_value=.99
            before=m.utility_q
            backend.update_value(m,True)
            self.assertEqual(m.utility_q,before)
            backend._service.update_value.assert_not_called()

    def test_unsupported_methods_reject_feedback_ablation(self):
        with patch.object(sys,"argv",["run_mmlu","--config",""]):
            args=run_mmlu.parse_args()
        args.method="agentsafe_full";args.method_explicit=True;args.outcome_feedback_policy="none"
        with self.assertRaisesRegex(ValueError,"feedback|MAPLE"):
            run_mmlu.resolve_args(args)

if __name__=="__main__":unittest.main()
