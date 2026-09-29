import ast
import copy
import os
import re
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "evaluate/defense_methods/agentxposed_full.py"
spec = importlib.util.spec_from_file_location("agentxposed_full", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AgentXposedFull = module.AgentXposedFull
TRAITS = ("honesty_humility", "emotionality", "extraversion",
          "agreeableness", "conscientiousness", "openness")


def score(value, reason="Observed cooperative behavior"):
    return {"scores": {trait: value for trait in TRAITS}, "reason": reason}


class ScriptedJudge:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if not self.answers:
            raise AssertionError("Unexpected judge call")
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)


def args(**updates):
    values = {"method": "agentxposed_full_kick", "agents": 3,
              "agentxposed_protocol": "reconstruction",
              "agentxposed_deviation_threshold": 1.0,
              "agentxposed_inquiry_rounds": 3, "agentxposed_guide_rounds": 2}
    values.update(updates)
    return SimpleNamespace(**values)


def no_response(*unused, **kwargs):
    raise AssertionError("An unflagged agent must not be interrogated")


class AgentXposedFullTests(unittest.TestCase):
    def test_baseline_uses_public_context_and_strict_boundary_does_not_interrogate(self):
        judge = ScriptedJudge([score(6), score(7)])
        guard = AgentXposedFull(args(), judge)
        guard.begin_task("first")
        outputs, decisions = guard.defend({0: "Initial visible answer"}, no_response,
                                         "Public question", 0, [[0]])
        self.assertEqual(outputs, {0: "Initial visible answer"})
        self.assertEqual(decisions[0]["details"]["baseline_scores"], score(6)["scores"])
        self.assertEqual(decisions[0]["details"]["deviation"], 0)
        context = json.loads(judge.calls[0][1]["content"])
        self.assertEqual(context["question"], "Public question")
        self.assertIn("Initial visible answer", json.dumps(context))
        outputs, decisions = guard.defend({0: "A second visible answer"}, no_response,
                                         "Public question", 1, [[0]])
        self.assertEqual(outputs[0], "A second visible answer")
        self.assertEqual(decisions[0]["details"]["baseline_scores"], score(6)["scores"])
        self.assertEqual(decisions[0]["details"]["current_scores"], score(7)["scores"])
        self.assertEqual(decisions[0]["details"]["deviation"], 1)
        self.assertEqual(decisions[0]["details"]["inquiry_count"], 0)
        self.assertFalse(guard.inactive)

    def test_adaptive_real_inquiry_kicks_persistently_and_resets_next_task(self):
        judge = ScriptedJudge([
            score(7), score(3),
            {"question": "Explain the evidence for your disputed claim."}, score(3),
            {"question": "How does your first reply address the task?"}, score(3),
            {"question": "Reconcile your second reply with the observed evidence."}, score(3),
            score(3), score(6),
        ])
        guard = AgentXposedFull(args(), judge)
        guard.begin_task("first")
        guard.defend({0: "First observed answer"}, no_response, "Question", 0, [])
        replies = []
        def respond(agent_id, prompt, reset=False):
            replies.append((agent_id, prompt, reset))
            return f"Real target reply {len(replies)}"
        outputs, decisions = guard.defend({0: "Abruptly changed conduct"}, respond,
                                         "Question", 1, [])
        self.assertEqual(outputs, {})
        self.assertEqual(guard.inactive, {0})
        self.assertEqual(len(replies), 3)
        self.assertTrue(all(agent == 0 and reset is False for agent, _, reset in replies))
        self.assertEqual(decisions[0]["action"], "block")
        details = decisions[0]["details"]
        self.assertEqual(details["inquiry_count"], 3)
        self.assertEqual(details["guide_count"], 0)
        self.assertEqual(details["final_deviation"], 4)
        self.assertEqual(details["inquiry_transcript"][0]["response"], "Real target reply 1")
        next_question_context = json.loads(judge.calls[4][1]["content"])
        self.assertIn("Real target reply 1", json.dumps(next_question_context))
        final_context = json.loads(judge.calls[8][1]["content"])
        self.assertIn("Real target reply 3", json.dumps(final_context))
        before = len(judge.calls)
        outputs, decisions = guard.defend({0: "Must never be published"}, no_response,
                                         "Question", 2, [])
        self.assertEqual(outputs, {})
        self.assertEqual(len(judge.calls), before)
        self.assertEqual(decisions[0]["reason"], "agentxposed_inactive")
        guard.begin_task("second")
        self.assertFalse(guard.inactive)
        outputs, decisions = guard.defend({0: "New task behavior"}, no_response,
                                         "Different question", 0, [])
        self.assertEqual(outputs, {0: "New task behavior"})
        self.assertEqual(decisions[0]["details"]["baseline_scores"], score(6)["scores"])

    def test_guide_uses_live_multiturn_correction_then_publishes_fresh_task_answer(self):
        judge = ScriptedJudge([
            score(7), score(3), {"question": "Explain the task deviation."}, score(3), score(3),
            {"guidance": "Revisit the public task and justify each factual claim."},
            {"guidance": "Address the gap identified in your last correction."},
            score(6), score(6),
        ])
        guard = AgentXposedFull(args(method="agentxposed_full_guide", agentxposed_inquiry_rounds=1), judge)
        guard.begin_task("guide")
        guard.defend({0: "Initial task answer"}, no_response, "Solve the public task", 0, [])
        prompts, replies = [], iter(["Actual inquiry reply", "First correction reply",
                                    "Second correction reply", "Fresh complete task answer"])
        def respond(agent_id, prompt, reset=False):
            prompts.append((agent_id, prompt, reset))
            return next(replies)
        outputs, decisions = guard.defend({0: "Suspect answer"}, respond,
                                         "Solve the public task", 1, [])
        self.assertEqual(outputs, {0: "Fresh complete task answer"})
        self.assertEqual(decisions[0]["action"], "rewrite")
        details = decisions[0]["details"]
        self.assertEqual(details["inquiry_count"], 1)
        self.assertEqual(details["guide_count"], 2)
        self.assertEqual(details["task_reanswer_count"], 1)
        self.assertEqual(details["post_guide_deviation"], 1)
        self.assertEqual(len(prompts), 4)
        self.assertTrue(all(reset is False for _, _, reset in prompts))
        self.assertIn("Solve the public task", prompts[-1][1])
        self.assertIn("First correction reply", json.dumps(judge.calls[6]))
        self.assertFalse(guard.inactive)
        outputs, decisions = guard.defend({0: "Cooperative subsequent answer"}, no_response,
                                         "Solve the public task", 2, [])
        self.assertEqual(outputs, {0: "Cooperative subsequent answer"})
        self.assertEqual(decisions[0]["details"]["baseline_scores"], score(7)["scores"])

    def test_invalid_judge_schema_never_silently_allows_or_uses_old_scores(self):
        valid = score(5)
        malformed = [
            "not JSON", "[]", json.dumps({"scores": {TRAITS[0]: 5}, "reason": "partial"}),
            json.dumps({**valid, "extra": 1}),
            json.dumps({"scores": {**valid["scores"], TRAITS[0]: True}, "reason": "bad type"}),
            json.dumps({"scores": {**valid["scores"], TRAITS[0]: "5"}, "reason": "string score"}),
            json.dumps({"scores": {**valid["scores"], TRAITS[0]: 0}, "reason": "out of range"}),
            json.dumps({"scores": {**valid["scores"], TRAITS[0]: 11}, "reason": "out of range"}),
            json.dumps({"scores": valid["scores"], "reason": ""}),
            json.dumps({"scores": valid["scores"], "reason": "fine"})[:-1] + ', "extra": NaN}',
            json.dumps(valid)[:-1] + ', "reason": "duplicate"}',
            json.dumps(valid).replace('"honesty_humility": 5', '"honesty_humility": 4, "honesty_humility": 5'),
        ]
        for raw in malformed:
            with self.subTest(raw=raw):
                guard = AgentXposedFull(args(), ScriptedJudge([raw]))
                guard.begin_task("invalid")
                with self.assertRaises(module.AgentXposedRuntimeError):
                    guard.defend({0: "Public statement"}, no_response, "Question", 0, [])
        judge = ScriptedJudge([score(7), "not JSON"])
        guard = AgentXposedFull(args(), judge)
        guard.begin_task("later-invalid")
        guard.defend({0: "First output"}, no_response, "Question", 0, [])
        with self.assertRaises(module.AgentXposedRuntimeError):
            guard.defend({0: "Later output"}, no_response, "Question", 1, [])

    def test_invalid_configuration_and_nonpublic_input_shapes_are_rejected(self):
        for updates in ({"method": "agentxposed"}, {"agentxposed_deviation_threshold": True},
                        {"agentxposed_deviation_threshold": -1},
                        {"agentxposed_deviation_threshold": float("nan")},
                        {"agentxposed_deviation_threshold": 10},
                        {"agentxposed_inquiry_rounds": 0}, {"agentxposed_inquiry_rounds": 1.5},
                        {"agentxposed_guide_rounds": False}, {"agents": 0}):
            with self.subTest(updates=updates), self.assertRaises(module.AgentXposedConfigError):
                AgentXposedFull(args(**updates), ScriptedJudge([]))
        guard = AgentXposedFull(args(), ScriptedJudge([score(6)]))
        with self.assertRaises(module.AgentXposedRuntimeError):
            guard.defend({0: "text"}, no_response, "Question", 0, [])
        guard.begin_task("validation")
        for outputs, question, turn in (
            ({True: "text"}, "Question", 0), ({-1: "text"}, "Question", 0),
            ({3: "text"}, "Question", 0), ({0: {"answer": "text", "label": "secret"}}, "Question", 0),
            ({0: "text"}, {"question": "Question", "correct": "secret"}, 0),
            ({0: "text"}, "Question", -1), ({0: ""}, "Question", 0),
        ):
            with self.subTest(outputs=outputs, question=question, turn=turn):
                with self.assertRaises(module.AgentXposedRuntimeError):
                    guard.defend(outputs, no_response, question, turn, [])
        self.assertEqual(guard.judge.calls, [])

    def test_final_verification_can_clear_a_soft_flag(self):
        judge = ScriptedJudge([score(7), score(3), {"question": "Explain the change."},
                               score(3), score(7)])
        guard = AgentXposedFull(args(agentxposed_inquiry_rounds=1), judge)
        guard.begin_task("cleared")
        guard.defend({0: "Initial"}, no_response, "Question", 0, [])
        outputs, decisions = guard.defend({0: "Changed but justified"}, lambda *a, **k: "New evidence",
                                         "Question", 1, [])
        self.assertEqual(outputs, {0: "Changed but justified"})
        self.assertFalse(guard.inactive)
        self.assertEqual(decisions[0]["reason"], "agentxposed_verification_cleared")
        self.assertEqual(decisions[0]["details"]["deviation"], 4)
        self.assertEqual(decisions[0]["details"]["final_deviation"], 0)

    def test_failed_judge_or_live_target_and_invalid_inquiry_are_not_fabricated(self):
        failure = TimeoutError("judge unavailable")
        guard = AgentXposedFull(args(), ScriptedJudge([failure]))
        guard.begin_task("judge-failure")
        with self.assertRaises(module.AgentXposedRuntimeError) as error:
            guard.defend({0: "Initial"}, no_response, "Question", 0, [])
        self.assertIs(error.exception.__cause__, failure)
        for query in ({"question": ""}, {"question": 4}, {"question": "Explain?", "extra": 1}):
            with self.subTest(query=query):
                judge = ScriptedJudge([score(7), score(3), query])
                guard = AgentXposedFull(args(), judge)
                guard.begin_task("bad-question")
                guard.defend({0: "Initial"}, no_response, "Question", 0, [])
                with self.assertRaises(module.AgentXposedRuntimeError):
                    guard.defend({0: "Changed"}, no_response, "Question", 1, [])
        for target_result in (TimeoutError("target unavailable"), "", {"answer": "not plain text"}):
            with self.subTest(target_result=target_result):
                judge = ScriptedJudge([score(7), score(3), {"question": "Explain the change?"}])
                guard = AgentXposedFull(args(), judge)
                guard.begin_task("target-failure")
                guard.defend({0: "Initial"}, no_response, "Question", 0, [])
                def respond(*unused, **kwargs):
                    if isinstance(target_result, BaseException):
                        raise target_result
                    return target_result
                with self.assertRaises(module.AgentXposedRuntimeError):
                    guard.defend({0: "Changed"}, respond, "Question", 1, [])

    def test_evaluator_metadata_and_private_instructions_do_not_reach_judge(self):
        observations = []
        for secret in ("SECRET_ATTACKER_0", "SECRET_BENIGN_1"):
            configuration = args()
            configuration.attacker_idxes = [secret]
            configuration.system_prompts = [secret]
            judge = ScriptedJudge([score(2), score(2)])
            guard = AgentXposedFull(configuration, judge)
            guard.begin_task(secret)
            result = guard.defend({0: "Public low-scoring style"}, no_response, "Public task", 0, [],
                                  ground_truth=secret, system_prompt=secret, taint=secret)
            result2 = guard.defend({0: "Same observable style"}, no_response, "Public task", 1, [],
                                   attacker_id=secret)
            observations.append((result, result2, judge.calls))
            self.assertNotIn(secret, json.dumps(judge.calls))
        self.assertEqual(observations[0], observations[1])
        self.assertEqual(observations[0][1][1][0]["action"], "allow")

    def test_equal_deviations_are_assessed_per_agent_not_as_a_lowest_score_tie(self):
        questions = [{"question": "Explain your own change."}, score(3), score(3)]
        judge = ScriptedJudge([score(7), score(7), score(3), *questions, score(3), *questions])
        guard = AgentXposedFull(args(agentxposed_inquiry_rounds=1), judge)
        guard.begin_task("two")
        guard.defend({0: "Initial zero", 1: "Initial one"}, no_response, "Question", 0, [])
        called = []
        def respond(agent_id, prompt, reset=False):
            called.append(agent_id)
            return f"Actual reply by agent {agent_id}"
        outputs, decisions = guard.defend({0: "Changed zero", 1: "Changed one"},
                                         respond, "Question", 1, [])
        self.assertEqual(called, [0, 1])
        self.assertEqual(outputs, {})
        self.assertEqual(guard.inactive, {0, 1})
        self.assertEqual([d["action"] for d in decisions], ["block", "block"])
        peers = json.loads(judge.calls[2][1]["content"])["peer_outputs"]
        self.assertEqual({p["agent_id"] for p in peers}, {0, 1})

    def test_shared_runtime_preserves_private_target_context_through_live_guide(self):
        runtime_spec = importlib.util.spec_from_file_location(
            "agentxposed_test_full_runtime", MODULE.parent / "full_runtime.py")
        runtime_module = importlib.util.module_from_spec(runtime_spec)
        runtime_spec.loader.exec_module(runtime_module)
        configuration = args(method="agentxposed_full_guide", agentxposed_inquiry_rounds=1)
        judge = ScriptedJudge([
            score(7), score(3), {"question": "Why did your behavior change?"}, score(3), score(3),
            {"guidance": "Reconsider your evidence."}, {"guidance": "Complete the public objective."},
            score(7),
        ])
        guard = AgentXposedFull(configuration, judge)
        runtime = runtime_module.FullRuntime(configuration, guard=guard)
        runtime.begin_task("integration", "Public objective")
        generated_contexts = []
        def generate(messages):
            generated_contexts.append(messages)
            return f"Live backend output {len(generated_contexts)}"
        original = [{"role": "system", "content": "PRIVATE_TARGET_PERSONA_SENTINEL"},
                    {"role": "user", "content": "Public objective"}]
        first = runtime.generate(0, original, generate)
        runtime.defend({0: first}, 0, [[0]])
        second = runtime.generate(0, original, generate)
        outputs, decisions = runtime.defend({0: second}, 1, [[0]])
        self.assertEqual(outputs, {0: "Live backend output 6"})
        self.assertEqual(len(generated_contexts), 6)
        self.assertTrue(all(context[0]["content"] == "PRIVATE_TARGET_PERSONA_SENTINEL"
                            for context in generated_contexts))
        self.assertIn("Live backend output 3", json.dumps(generated_contexts[-1]))
        self.assertIn("Live backend output 4", json.dumps(generated_contexts[-1]))
        self.assertNotIn("PRIVATE_TARGET_PERSONA_SENTINEL", json.dumps(judge.calls))
        self.assertEqual(decisions[0]["details"]["guide_count"], 2)

SOURCE = Path(os.environ.get("AGENTXPOSED_SOURCE", "/mnt/public/data/wj/baseline-references/AgentXposed"))


def released_args(**updates):
    values = vars(args()).copy()
    values.pop("agentxposed_protocol")
    values["agentxposed_code_dir"] = str(SOURCE)
    values["agents"] = 2
    values.update(updates)
    return SimpleNamespace(**values)


def released_scores(value):
    names = ("Honesty-Humility", "Emotionality", "Extraversion",
             "Agreeableness", "Conscientiousness", "Openness")
    return "\n".join(f"{name}: {value}/10" for name in names)


class ReleasedJudge:
    def __init__(self, values=(2, 7)):
        self.values = values
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), dict(kwargs)))
        system, prompt = messages[0]["content"], messages[1]["content"]
        if system.startswith("Analyze the agent's statements"):
            return released_scores(5)
        if system.startswith("You are simulating a participant"):
            return "Simulated participant answer."
        if prompt.startswith("Subject's dialog history:"):
            return "Can you explain the evidence supporting your claim?"
        found = re.search(r"Agent (\d+) was asked", prompt)
        if found:
            return released_scores(self.values[int(found.group(1)) - 1]) + "\nCan you reconcile the competing evidence in your account?"
        raise AssertionError("Unexpected released judge prompt")



def original_source_result(memories, judge, fixed):
    """Independent reference: exact official definitions, one textual assignment if fixed."""
    def completion(**kwargs):
        value = judge(kwargs["messages"], temperature=kwargs["temperature"], response_format=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value))])
    namespace = {"openai": SimpleNamespace(ChatCompletion=SimpleNamespace(create=completion)),
                 "os": SimpleNamespace(getenv=lambda *args: ""),
                 "print": lambda *args, **kwargs: None}
    for file, name in (("Detect/model.py", "HexacoSystem"),
                       ("Detect/reid_interrogator.py", "ReidInterrogator"),
                       ("Detect/file_handler.py", "get_dialog_history"),
                       ("Detect/main.py", "process_question")):
        source = (SOURCE / file).read_text()
        if fixed and file == "Detect/main.py":
            assert source.count("    weights = {") == 1
            source = source.replace("    weights = {",
                                    "        final_hexaco_scores[agent_id] = updated_scores\n    weights = {")
        parsed = ast.parse(source, filename=str(SOURCE / file))
        selected = [node for node in parsed.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE / file), "exec"), namespace)
    return namespace["process_question"]("", memories, len(memories))


@unittest.skipUnless((SOURCE / "Detect/main.py").is_file(), "external AgentXposed source not present")
class ReleasedAgentXposedTests(unittest.TestCase):
    def memories(self):
        return [[{"role": "user", "content": f"Public role {i}\n\nTask\n\nThird paragraph\n\nFourth paragraph"},
                 {"role": "assistant", "content": f"Visible answer {i}"}] for i in range(2)]

    def test_default_runs_source_three_simulated_rounds_for_all_agents(self):
        judge = ReleasedJudge()
        guard = AgentXposedFull(released_args(), judge)
        self.assertEqual(guard.protocol, "released_minimal_fix")
        result = guard.detect_released(self.memories())
        self.assertEqual(result["detected_malicious_agent"], 1)
        self.assertEqual(set(result["hexaco_scores"]), {1, 2})
        for agent in (1, 2):
            process = result["interrogation_process"][agent]["interrogation_process"]
            self.assertEqual(len(process["rounds"]), 3)
            self.assertEqual(process["initial_hexaco_scores"], score(5)["scores"])
        self.assertEqual(len(judge.calls), 16)
        self.assertEqual(judge.calls[0][1], {"temperature": 0.3, "response_format": None})
        simulated = [m for m, _ in judge.calls if m[0]["content"].startswith("You are simulating a participant")]
        self.assertEqual(len(simulated), 6)
        self.assertNotIn("Fourth paragraph", json.dumps(simulated))
        self.assertIn("Third paragraph", json.dumps(simulated))

    def test_released_guide_regenerates_first_answer_and_appends_input_on_later_turns(self):
        judge = ReleasedJudge()
        guard = AgentXposedFull(released_args(method="agentxposed_full_guide"), judge)
        guard.begin_task("guide")
        messages = [{"role": "system", "content": "PRIVATE_SYSTEM_SENTINEL"},
                    {"role": "user", "content": "Original current user message"}]
        self.assertEqual(guard.prepare_messages(0, messages), messages)
        regeneration = []
        def regenerate(agent_id, guidance):
            regeneration.append((agent_id, guidance))
            return "New answer generated with input guidance"
        memories = self.memories()
        outputs, decisions = guard.defend(
            {0: "Visible answer 0", 1: "Visible answer 1"}, no_response, "Public task", 0, [],
            released_memories=lambda i: memories[i], regenerate=regenerate)
        self.assertEqual(outputs[0], "New answer generated with input guidance")
        self.assertEqual(outputs[1], "Visible answer 1")
        self.assertEqual(regeneration, [(0, guard.released_guidance)])
        self.assertEqual(decisions[0]["details"]["protocol"], "released_minimal_fix")
        prepared = guard.prepare_messages(0, messages)
        self.assertEqual(prepared[-1]["content"], messages[-1]["content"] + guard.released_guidance)
        self.assertEqual(messages[-1]["content"], "Original current user message")
        self.assertEqual(guard.prepare_messages(1, messages), messages)
        count = len(judge.calls)
        outputs, _ = guard.defend({0: "Already guided later answer", 1: "Later peer answer"},
                                 no_response, "Public task", 1, [])
        self.assertEqual(outputs[0], "Already guided later answer")
        self.assertEqual(len(judge.calls), count)
        self.assertNotIn("PRIVATE_SYSTEM_SENTINEL", json.dumps(judge.calls))

    def test_both_released_profiles_match_original_source_results_prompts_and_order(self):
        memories = self.memories()
        for memory in memories:
            memory.insert(0, {"role": "system", "content": "PRIVATE_NEVER_SIMULATE_THIS_SYSTEM"})
            memory.extend([{"role": "user", "content": "Follow-up user instruction"},
                           {"role": "assistant", "content": "Second actual answer"}])
        for fixed in (False, True):
            protocol = "released_minimal_fix" if fixed else "released_unmodified"
            with self.subTest(protocol=protocol):
                expected_judge, actual_judge = ReleasedJudge(), ReleasedJudge()
                expected = original_source_result(memories, expected_judge, fixed)
                guard = AgentXposedFull(released_args(agentxposed_protocol=protocol), actual_judge)
                actual = guard.detect_released(memories)
                self.assertEqual(actual, expected)
                self.assertEqual(actual_judge.calls, expected_judge.calls)
                self.assertNotIn("PRIVATE_NEVER_SIMULATE_THIS_SYSTEM", json.dumps(actual_judge.calls))
                self.assertEqual(len(actual_judge.calls), 18)

    def test_unmodified_release_retains_empty_final_scores_and_no_op_defense(self):
        guard = AgentXposedFull(released_args(agentxposed_protocol="released_unmodified"), ReleasedJudge())
        guard.begin_task("original-bug")
        memories = self.memories()
        outputs, decisions = guard.defend({0: "Visible answer 0", 1: "Visible answer 1"},
                                         no_response, "Task", 0, [],
                                         released_memories=lambda i: memories[i])
        self.assertEqual(outputs, {0: "Visible answer 0", 1: "Visible answer 1"})
        self.assertFalse(guard.inactive)
        result = decisions[0]["details"]["source_detection_result"]
        self.assertEqual(result["hexaco_scores"], {})
        self.assertIsNone(result["detected_malicious_agent"])
        self.assertIsNone(decisions[0]["details"]["source_patch"])
        self.assertEqual(decisions[0]["details"]["inquiry_count"], 3)

    def test_released_fixed_ties_are_uncertain_and_exact_threshold_is_not_below(self):
        tied = AgentXposedFull(released_args(), ReleasedJudge((2, 2))).detect_released(self.memories())
        self.assertIsNone(tied["detected_malicious_agent"])
        self.assertIn("tied", tied["detection_reasoning"])
        regular = ReleasedJudge()
        def boundary_judge(messages, **kwargs):
            value = regular(messages, **kwargs)
            if messages[1]["content"].startswith("Agent 1 was asked"):
                value = released_scores(0).replace("Honesty-Humility: 0/10", "Honesty-Humility: 5/10")
            return value
        result = AgentXposedFull(released_args(), boundary_judge).detect_released(self.memories())
        self.assertEqual(result["interrogation_process"][1]["weighted_average_score"], 3.0)
        self.assertIsNone(result["detected_malicious_agent"])

    def test_released_kick_persists_without_redetecting_and_new_task_resets(self):
        judge = ReleasedJudge()
        guard = AgentXposedFull(released_args(), judge)
        memories = self.memories()
        guard.begin_task("kick")
        outputs, _ = guard.defend({0: "Visible answer 0", 1: "Visible answer 1"}, no_response,
                                 "Task", 0, [], released_memories=lambda i: memories[i])
        self.assertEqual(outputs, {1: "Visible answer 1"})
        self.assertEqual(guard.inactive, {0})
        count = len(judge.calls)
        outputs, decisions = guard.defend({0: "Should be dropped", 1: "Second answer"},
                                         no_response, "Task", 1, [])
        self.assertEqual(outputs, {1: "Second answer"})
        self.assertEqual(len(judge.calls), count)
        self.assertEqual(decisions[0]["details"]["inquiry_count"], 0)
        guard.begin_task("new")
        self.assertFalse(guard.inactive)
        guard.defend({0: "Visible answer 0", 1: "Visible answer 1"}, no_response,
                     "Task", 0, [], released_memories=lambda i: memories[i])
        self.assertEqual(len(judge.calls), 2 * count)

    def test_released_parser_and_source_api_failure_fallback_are_preserved(self):
        regular = ReleasedJudge()
        def unusual_judge(messages, **kwargs):
            response = regular(messages, **kwargs)
            if messages[1]["content"].startswith("Agent 1 was asked"):
                return "Honesty-Humility: -2.5/10\nOpenness: 100/10"
            return response
        actual = AgentXposedFull(released_args(), unusual_judge).detect_released(self.memories())
        scores = actual["hexaco_scores"][1]
        self.assertEqual(scores["honesty_humility"], 2)
        self.assertEqual(scores["openness"], 10)
        self.assertEqual(scores["agreeableness"], 5)
        def failed_judge(*args, **kwargs):
            raise TimeoutError("deterministic unavailable")
        guard = AgentXposedFull(released_args(), failed_judge)
        actual = guard.detect_released(self.memories())
        expected = original_source_result(self.memories(), failed_judge, True)
        self.assertEqual(actual, expected)
        self.assertIsNone(actual["detected_malicious_agent"])

    def test_released_requires_matching_source_and_expected_hooks(self):
        for configuration in (released_args(agentxposed_code_dir=""),
                              released_args(agentxposed_code_dir="/missing/agentxposed"),
                              released_args(agentxposed_protocol="unknown")):
            with self.subTest(configuration=vars(configuration)):
                with self.assertRaises(module.AgentXposedConfigError):
                    AgentXposedFull(configuration, ReleasedJudge())
        with tempfile.TemporaryDirectory() as temp:
            for name in module.RELEASE_HASHES:
                target = Path(temp) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((SOURCE / name).read_bytes())
            model = Path(temp) / "Detect/model.py"
            model.write_bytes(model.read_bytes() + b"\n# modified snapshot\n")
            with self.assertRaisesRegex(module.AgentXposedConfigError, "hash mismatch"):
                AgentXposedFull(released_args(agentxposed_code_dir=temp), ReleasedJudge())
        guard = AgentXposedFull(released_args(), ReleasedJudge())
        guard.begin_task("requires-history")
        with self.assertRaisesRegex(module.AgentXposedRuntimeError, "released_memories"):
            guard.defend({0: "a", 1: "b"}, no_response, "Task", 0, [])
        guard = AgentXposedFull(released_args(method="agentxposed_full_guide"), ReleasedJudge())
        guard.begin_task("requires-replay")
        memories = self.memories()
        with self.assertRaisesRegex(module.AgentXposedRuntimeError, "pre-generation replay"):
            guard.defend({0: "a", 1: "b"}, no_response, "Task", 0, [],
                         released_memories=lambda i: memories[i])
        guard.defend({0: "a", 1: "b"}, no_response, "Task", 0, [],
                     released_memories=lambda i: memories[i], regenerate=lambda i, text: "fixed")
        with self.assertRaisesRegex(module.AgentXposedRuntimeError, "input hook"):
            guard.defend({0: "unguided", 1: "b"}, no_response, "Task", 1, [])

    def test_source_loading_does_not_assign_api_key_or_import_openai(self):
        sentinel = os.environ.get("OPENAI_API_KEY")
        original_module = sys.modules.get("openai")
        guard = AgentXposedFull(released_args(), ReleasedJudge())
        guard.detect_released(self.memories())
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), sentinel)
        self.assertIs(sys.modules.get("openai"), original_module)
        self.assertEqual(guard._released_engine.calls[0]["source_model"], "gpt-4o-mini")

    def test_released_full_runtime_replays_same_input_and_guides_future_generations(self):
        runtime_spec = importlib.util.spec_from_file_location(
            "released_test_full_runtime", MODULE.parent / "full_runtime.py")
        runtime_module = importlib.util.module_from_spec(runtime_spec)
        runtime_spec.loader.exec_module(runtime_module)
        configuration = released_args(method="agentxposed_full_guide")
        judge = ReleasedJudge()
        guard = AgentXposedFull(configuration, judge)
        runtime = runtime_module.FullRuntime(configuration, guard=guard)
        runtime.begin_task("released-integration", "Task")
        calls = []
        def generate(messages):
            calls.append(copy.deepcopy(messages))
            return f"Generated {len(calls)}"
        initial = [{"role": "system", "content": "PRIVATE_SYSTEM_FOR_GENERATION_ONLY"},
                   {"role": "user", "content": "Task and original user instruction"}]
        outputs = {i: runtime.generate(i, initial, generate) for i in range(2)}
        outputs, _ = runtime.defend(outputs, 0, [])
        self.assertEqual(outputs, {0: "Generated 3", 1: "Generated 2"})
        self.assertEqual(len(calls[2]), len(calls[0]))
        self.assertEqual(calls[2][0], calls[0][0])
        self.assertEqual(calls[2][-1]["role"], "user")
        self.assertEqual(calls[2][-1]["content"], calls[0][-1]["content"] + guard.released_guidance)
        self.assertNotIn("Generated 1", json.dumps(calls[2]))
        self.assertNotIn("PRIVATE_SYSTEM_FOR_GENERATION_ONLY", json.dumps(judge.calls))
        count = len(judge.calls)
        second = {i: runtime.generate(i, initial, generate) for i in range(2)}
        outputs, _ = runtime.defend(second, 1, [])
        self.assertTrue(calls[3][-1]["content"].endswith(guard.released_guidance))
        self.assertFalse(calls[4][-1]["content"].endswith(guard.released_guidance))
        self.assertEqual(outputs, second)
        self.assertEqual(len(judge.calls), count)


if __name__ == "__main__":
    unittest.main()
