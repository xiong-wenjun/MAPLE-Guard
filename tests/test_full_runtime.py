import importlib
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from maple_guard import maple_guard_core as ep

class FullRuntimeTests(unittest.TestCase):
    def runtime_module(self):
        # Report an assertion for the missing feature rather than an import traceback.
        spec = importlib.util.find_spec("evaluate.defense_methods.full_runtime")
        self.assertIsNotNone(spec, "full baseline session and routing runtime is missing")
        return importlib.import_module("evaluate.defense_methods.full_runtime")

    def test_full_methods_are_separate_from_legacy_choices(self):
        for name in ("agentsafe_full", "infa_guard_full", "agentxposed_full_guide", "agentxposed_full_kick"):
            self.assertIn(name, ep.METHOD_CHOICES)
        self.assertIn("agentsafe", ep.METHOD_CHOICES)

    def test_kick_prevents_future_model_execution(self):
        rt = self.runtime_module()
        guard = SimpleNamespace(inactive={1})
        session = rt.FullRuntime(SimpleNamespace(method="agentxposed_full_kick"), guard=guard)
        called = []
        result = session.generate(1, [{"role":"user","content":"question"}], lambda msgs: called.append(msgs))
        self.assertIsNone(result)
        self.assertEqual(called, [])

    def test_guidance_reaches_actual_target_and_persists(self):
        rt = self.runtime_module()
        session = rt.FullRuntime(SimpleNamespace(method="agentxposed_full_guide"), guard=SimpleNamespace(inactive=set()))
        requests = []
        def generate(messages):
            requests.append(messages)
            return "corrected" if "Please reconsider" in str(messages) else "original"
        session.begin_task("t", "question")
        session.generate(0, [{"role":"system","content":"target private role"},{"role":"user","content":"question"}], generate)
        self.assertEqual(session.respond(0, "Please reconsider"), "corrected")
        session.generate(0, [{"role":"system","content":"target private role"},{"role":"user","content":"next round"}], generate)
        self.assertIn("Please reconsider", str(requests[-1]))
        self.assertIn("corrected", str(requests[-1]))

    def test_donor_replacement_survives_original_prompt_next_round(self):
        rt = self.runtime_module()
        session = rt.FullRuntime(SimpleNamespace(method="infa_guard_full"), guard=SimpleNamespace(inactive=set()))
        seen = []
        def generate(messages):
            seen.append(messages)
            return "answer"
        session.begin_task("t", "question")
        session.generate(0, [{"role":"system","content":"malicious role"},{"role":"user","content":"poison"}], generate)
        session.generate(1, [{"role":"system","content":"donor role"},{"role":"user","content":"question"}], generate)
        session.replace_agent(0, 1, "donor answer")
        session.generate(0, [{"role":"system","content":"malicious role"},{"role":"user","content":"new task evidence"}], generate)
        self.assertIn("donor role", str(seen[-1]))
        self.assertNotIn("malicious role", str(seen[-1]))
        self.assertNotIn("poison", str(seen[-1]))

    def test_scope_cleanup_after_exception_and_no_cross_experiment_state(self):
        rt = self.runtime_module()
        a = SimpleNamespace(method="agentxposed_full_kick")
        b = SimpleNamespace(method="agentxposed_full_kick")
        factory = lambda args: rt.FullRuntime(args, guard=SimpleNamespace(inactive=set()))
        with self.assertRaises(ValueError):
            with rt.runtime_scope(a, factory=factory) as first:
                first.guard.inactive.add(1)
                self.assertIs(rt.current_runtime(a.method), first)
                raise ValueError("abort")
        with self.assertRaises(RuntimeError):
            rt.current_runtime(a.method)
        with rt.runtime_scope(b, factory=factory) as second:
            self.assertEqual(second.guard.inactive, set())

    def test_missing_full_session_fails_instead_of_bypassing_admission(self):
        self.runtime_module()
        with self.assertRaises(RuntimeError):
            ep.commit_memory(SimpleNamespace(), "team", 0, "agentsafe_full", {}, [])




    def test_replacement_preserves_donor_tools_and_memory_owner(self):
        rt=self.runtime_module()
        session=rt.FullRuntime(SimpleNamespace(method="infa_guard_full"),guard=SimpleNamespace(inactive=set()))
        session.begin_task("t","summary question")
        for aid in (0,1):
            clean=[{"role":"system","content":f"role {aid}"},{"role":"user","content":f"tools and task for {aid}"}]
            session.register_task_context(aid,clean)
            session.generate(aid,clean,lambda messages:"answer")
        session.replace_agent(0,1,"donor answer")
        self.assertIn("tools and task for 1",str(session.contexts[0]))
        self.assertEqual(session.memory_owner(0),1)
        self.assertEqual(session.select_memories({0:["poison"],1:["donor memory"]})[0],["donor memory"])

    def test_peer_visibility_obeys_direction_and_kick(self):
        rt=self.runtime_module()
        session=rt.FullRuntime(SimpleNamespace(method="agentxposed_full_kick"),guard=SimpleNamespace(inactive={2}))
        visible=session.peer_context({0:"own",1:"permitted",2:"kicked",3:"no edge"},[[0]*4,[1,0,0,0],[1,0,0,0],[0]*4],0)
        self.assertIn("permitted",visible)
        self.assertNotIn("kicked",visible)
        self.assertNotIn("no edge",visible)

    def test_persistent_agentsafe_requires_explicit_state_path(self):
        rt=self.runtime_module()
        with self.assertRaisesRegex(ValueError,"baseline-state-path"):
            rt.FullRuntime(SimpleNamespace(method="agentsafe_full",memory_backend="memrl"),guard=SimpleNamespace())


    def test_agentsafe_restart_preserves_quarantine_and_review_registry(self):
        import json
        from evaluate.defense_methods.agentsafe_full import AgentSafeFull
        from test_agentsafe_full import FakeJudge
        with tempfile.TemporaryDirectory() as folder:
            policy=Path(folder)/"policy.json"
            criteria=Path(folder)/"criteria.json"
            policy.write_text(json.dumps({"identities":{"0":"Agent 0"}}))
            criteria.write_text(json.dumps(["valid task facts"]))
            args=SimpleNamespace(method="agentsafe_full",baseline_state_path=str(Path(folder)/"state.json"),
                agentsafe_policy_file=str(policy),agentsafe_criteria_file=str(criteria),agentsafe_threshold=0.5,
                agentsafe_review_interval=1)
            judge=FakeJudge(level=1)
            guard=AgentSafeFull(args,judge,lambda text:[1.,0.])
            runtime=self.runtime_module().FullRuntime(args,guard=guard)
            entry=SimpleNamespace(memory_id="m",experience="bad record",origin_agent=0,baseline_metadata={},status="active")
            self.assertTrue(runtime.admit(entry,"private",0)[0])
            judge.junk=True
            runtime.defend({0:"reply"},0,[[0]])
            self.assertEqual(entry.status,"quarantined")
            restored=self.runtime_module().FullRuntime(args,guard=AgentSafeFull(args,judge,lambda text:[1.,0.]))
            self.assertIn("m",restored.entries)
            fresh=SimpleNamespace(memory_id="m",experience="bad record",origin_agent=0,baseline_metadata={},status="active")
            self.assertEqual(restored.filter_entries([fresh],0)[0],[])
            self.assertEqual(restored.review_clock,1)
            entry.experience="new valid version"
            judge.junk=False
            self.assertEqual(runtime.filter_entries([entry],0)[0],[entry])
            self.assertEqual(entry.status,"active")
            entry.experience="bad record"
            self.assertEqual(runtime.filter_entries([entry],0)[0],[])
            self.assertEqual(entry.status,"quarantined")

    def test_shared_agentsafe_quarantine_is_holder_local_and_persistent(self):
        import json
        from evaluate.defense_methods.agentsafe_full import AgentSafeFull
        from test_agentsafe_full import FakeJudge
        with tempfile.TemporaryDirectory() as folder:
            policy=Path(folder)/"policy.json"; criteria=Path(folder)/"criteria.json"
            policy.write_text(json.dumps({"identities":{"0":"Agent 0","1":"Agent 1"}}))
            criteria.write_text(json.dumps(["valid task facts"]))
            args=SimpleNamespace(method="agentsafe_full",baseline_state_path=str(Path(folder)/"state.json"),
                agentsafe_policy_file=str(policy),agentsafe_criteria_file=str(criteria),agentsafe_threshold=0.5,
                agentsafe_review_interval=1)
            judge=FakeJudge(level=1)
            rt=self.runtime_module().FullRuntime(args,guard=AgentSafeFull(args,judge,lambda t:[1.,0.]))
            m=SimpleNamespace(memory_id="shared",experience="facts",origin_agent=0,
                memory_scope="team",baseline_metadata={},status="active")
            self.assertTrue(rt.admit(m,"team",0)[0])
            self.assertEqual(rt.filter_entries([m],1)[0],[m])
            judge.junk_by_holder[1]=True
            rt.defend({0:"a",1:"b"},0,[[0,1],[1,0]])
            self.assertEqual(m.status,"active")
            self.assertEqual(rt.filter_entries([m],1)[0],[])
            self.assertEqual(rt.filter_entries([m],0)[0],[m])
            restored=self.runtime_module().FullRuntime(args,guard=AgentSafeFull(args,judge,lambda t:[1.,0.]))
            fresh=SimpleNamespace(memory_id="shared",experience="facts",origin_agent=0,
                memory_scope="team",baseline_metadata={},status="active")
            self.assertEqual(restored.filter_entries([fresh],1)[0],[])
            self.assertEqual(restored.filter_entries([fresh],0)[0],[fresh])
            self.assertEqual(fresh.status,"active")

    def test_released_infa_filters_senders_but_keeps_generation(self):
        rt=self.runtime_module()
        guard=SimpleNamespace(inactive=set(),blocked_senders={0},takeover_context="donor_system_only")
        session=rt.FullRuntime(SimpleNamespace(method="infa_guard_full"),guard=guard)
        session.begin_task("t","task")
        for aid in (0,1):
            messages=[{"role":"system","content":f"role {aid}"},{"role":"user","content":f"private tools {aid}"}]
            session.register_task_context(aid,messages)
            session.generate(aid,messages,lambda ms:"answer")
        self.assertEqual(session.peer_context({0:"blocked",1:"allowed"},[[0,1],[1,0]],1),"")
        self.assertEqual(session.generate(0,[{"role":"user","content":"next"}],lambda ms:"still active"),"still active")
        session.replace_agent(0,1,"donor response")
        self.assertEqual(session.contexts[0],[{"role":"system","content":"role 1"},{"role":"assistant","content":"donor response"}])
        self.assertEqual(session.memory_owner(0),0)
        self.assertEqual(session.replacements,{})

    def test_released_agentxposed_gets_public_history_and_guides_same_turn_prompt(self):
        rt=self.runtime_module()
        class Guard:
            inactive=set()
            def prepare_messages(self,aid,messages):
                return messages
            def defend(self,outputs,*a,released_memories,regenerate,**kw):
                self.public=released_memories(0)
                return {0:regenerate(0,"OFFICIAL GUIDE")},[]
        guard=Guard()
        session=rt.FullRuntime(SimpleNamespace(method="agentxposed_full_guide"),guard=guard)
        session.begin_task("t","question")
        seen=[]
        session.generate(0,[{"role":"system","content":"PRIVATE ROLE"},{"role":"user","content":"TASK"}],lambda msgs: seen.append(msgs) or "answer")
        session.defend({0:"answer"},0,[[0]])
        self.assertEqual(guard.public,[{"role":"user","content":"TASK"},{"role":"assistant","content":"answer"}])
        self.assertEqual(len(seen[1]),2)
        self.assertIn("TASK",seen[1][-1]["content"])
        self.assertIn("OFFICIAL GUIDE",seen[1][-1]["content"])

    def test_agentsafe_runtime_uses_retained_holder_history_and_removes_junk(self):
        import json
        from evaluate.defense_methods.agentsafe_full import AgentSafeFull
        from test_agentsafe_full import FakeJudge
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"p.json";c=Path(folder)/"c.json"
            p.write_text(json.dumps({"identities":{"0":"Agent 0","-1":"User"}}))
            c.write_text(json.dumps(["facts"]))
            args=SimpleNamespace(method="agentsafe_full",agentsafe_policy_file=str(p),
                agentsafe_criteria_file=str(c),agentsafe_threshold=.5,agentsafe_review_interval=1)
            judge=FakeJudge(level=1)
            session=self.runtime_module().FullRuntime(args,guard=AgentSafeFull(args,judge,lambda t:[1.,0.]))
            m=SimpleNamespace(memory_id="seed",experience="RETAINED HISTORY",origin_agent=-1,baseline_metadata={},status="active")
            self.assertTrue(session.admit(m,"private",0)[0])
            prompts=[]
            session.generate(0,[{"role":"user","content":"question"}],lambda ms:prompts.append(ms) or "answer")
            self.assertIn("RETAINED HISTORY",str(prompts[-1]))
            judge.junk=True
            session.defend({0:"answer"},0,[[0]])
            self.assertEqual(m.status,"quarantined")
            session.generate(0,[{"role":"user","content":"next"}],lambda ms:prompts.append(ms) or "answer")
            self.assertNotIn("RETAINED HISTORY",str(prompts[-1]))

    def test_conflicting_private_read_id_fails_before_registry_mutation(self):
        rt=self.runtime_module()
        session=rt.FullRuntime(SimpleNamespace(method="agentsafe_full"),
            guard=SimpleNamespace(read=lambda **kw:(True,{})))
        a=SimpleNamespace(memory_id="same",experience="A",origin_agent=0,memory_scope="agent_private",baseline_metadata={},status="active")
        b=SimpleNamespace(memory_id="same",experience="B",origin_agent=0,memory_scope="agent_private",baseline_metadata={},status="active")
        session.filter_entries([a],0)
        with self.assertRaisesRegex(ValueError,"unique"):
            session.filter_entries([b],1)
        self.assertIs(session.entries["same"],a)
        self.assertEqual(session.holders_by_memory["same"],{0})

    def test_full_methods_reject_oracle_filter_settings(self):
        rt=self.runtime_module()
        for setting in ({"exclude_attackers_from_final_vote":True},{"communication_guard":"source_aware"}):
            args=SimpleNamespace(method="infa_guard_full",**setting)
            with self.subTest(setting=setting),self.assertRaises(ValueError):
                with rt.runtime_scope(args): pass




    def test_other_methods_do_not_consume_agentsafe_sidecar(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"as.json"
            path.write_text(json.dumps({"method":"agentsafe_full"}))
            session=self.runtime_module().FullRuntime(SimpleNamespace(method="agentxposed_full_kick",baseline_state_path=str(path)),guard=SimpleNamespace())
            session.save()
            self.assertEqual(json.loads(path.read_text()),{"method":"agentsafe_full"})

if __name__ == "__main__": unittest.main()
