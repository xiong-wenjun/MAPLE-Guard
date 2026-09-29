import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch_geometric.data import Data
from evaluate.defense_methods import guardian_defense as guardian
from evaluate.defense_methods.base import DefenseContext, OfficialDefenseState
from evaluate.defense_methods import reproduction as repro

class ReproductionTests(unittest.TestCase):
    def test_lock_mismatch_never_executes_external_source(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.py"; p.write_text("x = 1")
            with self.assertRaisesRegex(ValueError, "hash"):
                repro.verify_files(d, {"a.py": "0"*64})

    def test_extracted_function_does_not_execute_module_side_effects(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "source.py"
            p.write_text("raise RuntimeError('top-level must not execute')\ndef f(x):\n    return x + 1\n")
            self.assertEqual(repro.source_function(p, "f", {})(2), 3)

    def test_infa_recipe_has_published_grid_and_effective_training_flags(self):
        recipe = repro.infa_recipe("/source", "/job", "/dataset", "gpt-4o-mini", 42)
        jobs = recipe["generation"]
        self.assertEqual(len(jobs), 20)
        self.assertEqual(recipe["expected_dialogues"], 800)
        self.assertEqual(recipe["observed_turns"], 4)
        self.assertEqual({j["sparsity"] for j in jobs}, {.2, .4, .6, .8, 1.})
        self.assertEqual({j["attackers"] for j in jobs}, {1, 2, 3, 4})
        self.assertEqual(len({j["seed"] for j in jobs}), 20)
        train = recipe["training"]["argv"]
        self.assertEqual(train[train.index("--epochs")+1], "50")
        self.assertIn("--selective_training", train)
        self.assertFalse(recipe["author_checkpoint"])
        self.assertFalse(recipe["original_random_seed_known"])
        self.assertFalse(recipe["training_completed"])

    def test_heldout_understands_all_five_bundle_question_fields(self):
        rows = [
            {"tasks":[{"data":{"question":{"stem":"CSQA question","choices":[]}}}]},
            {"tasks":[{"data":{"instruction":"AppWorld task"}}]},
            {"tasks":[{"data":{"User Instruction":"InjecAgent task"}}]},
            {"tasks":[{"data":{"question":"MMLU question"}}]},
            {"tasks":[{"data":{"question":"LongMem question"}}]},
        ]
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for i,row in enumerate(rows):
                path=Path(d)/f"{i}.json";path.write_text(json.dumps(row));paths.append(path)
            self.assertEqual(repro.heldout_questions(paths), {
                "csqa question","appworld task","injecagent task","mmlu question","longmem question"})

    def test_generation_validation_rejects_missing_temporal_labels_and_test_overlap(self):
        row = {"question":"test question", "communication_data":[[]]*4,
               "infected_idxes_per_turn":[[]]*4, "adj_matrix":[[0]*8 for _ in range(8)],
               "attacker_idxes":[0], "system_prompts":["x"]*8}
        with self.assertRaisesRegex(ValueError, "held-out"):
            repro.validate_infa_dialogues([row], {"test question"})
        del row["infected_idxes_per_turn"]
        with self.assertRaisesRegex(ValueError, "per-turn"):
            repro.validate_infa_dialogues([row], set())

    def test_guardian_released_graph_is_complete_even_for_chain_host(self):
        args = SimpleNamespace(official_defense_guardian_profile="released_detector")
        ctx = DefenseContext("guardian", "q", "question", 0, [[0,1,0],[0,0,1],[0,0,0]], args)
        graphs = guardian._build_graph_data({"torch":torch,"Data":Data}, ctx, [{0:"A",1:"B",2:"C"}], [0,1,2])
        self.assertEqual(graphs[0].edge_index.shape[1], 6)
        self.assertEqual(graphs[0].text, ["A","B","C"])
        smaller = guardian._build_graph_data({"torch":torch,"Data":Data}, ctx, [{0:"A",1:"B",2:"C"}], [0,2])
        self.assertEqual(smaller[0].edge_index.tolist(), [[0,1],[1,0]])
        self.assertEqual(smaller[0].text, ["A","C"])

    def test_guardian_host_graph_stays_unchanged(self):
        ctx = DefenseContext("guardian", "q", "question", 0, [[0,1],[0,0]], SimpleNamespace())
        graphs = guardian._build_graph_data({"torch":torch,"Data":Data}, ctx, [{0:"A",1:"B"}], [0,1])
        self.assertEqual(graphs[0].edge_index.tolist(), [[0],[1]])

    def test_released_profile_rejects_nonofficial_epochs_and_device(self):
        for epochs, device in ((1,"cpu"), (20,"cuda")):
            args = SimpleNamespace(official_defense_guardian_profile="released_detector",
                                   official_defense_guardian_epochs=epochs,
                                   official_defense_gnn_device=device)
            ctx = DefenseContext("guardian", "q", "q", 0, [], args)
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(ValueError, "20 epochs|CPU"):
                    guardian._validate_released_profile(ctx, "unused")

    def test_matrix_released_guardian_carries_assets_and_twenty_epochs(self):
        from tools.run_appworld_matrix import extras
        args=SimpleNamespace(guardian_source="/source/code",guardian_bert_dir="/bert-base-uncased",
                             guardian_profile="released_detector")
        command,error=extras(args,"guardian","new-run")
        self.assertIsNone(error)
        self.assertEqual(command[command.index("--official-defense-guardian-profile")+1],"released_detector")
        self.assertEqual(command[command.index("--official-defense-guardian-epochs")+1],"20")
        self.assertIn("/bert-base-uncased",command)
        args.guardian_bert_dir=""
        command,error=extras(args,"guardian","new-run")
        self.assertEqual(command,[])
        self.assertIn("no pretrained detector checkpoint",error)

    def test_infa_api_audit_keeps_request_unchanged_and_rejects_truncation(self):
        import asyncio
        from openai.resources.chat.completions import AsyncCompletions
        from tools.run_infa_release_stage import install_api_audit
        for reason, text, succeeds in (("stop","A",True),("length","partial",False),("stop","",False)):
            calls=[]
            async def fake(_self, *args, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason=reason,message=SimpleNamespace(content=text))],
                    usage=SimpleNamespace(completion_tokens=5,prompt_tokens=10))
            with tempfile.TemporaryDirectory() as d, patch.object(AsyncCompletions,"create",fake):
                path=Path(d)/"api.jsonl"
                install_api_audit(path)
                kwargs={"model":"test","max_tokens":1024,"temperature":0,"messages":[]}
                if succeeds:
                    asyncio.run(AsyncCompletions.create(None,**kwargs))
                else:
                    with self.assertRaisesRegex(RuntimeError,"Incomplete"):
                        asyncio.run(AsyncCompletions.create(None,**kwargs))
                self.assertEqual(calls,[kwargs])
                self.assertEqual(json.loads(path.read_text())["finish_reason"],reason)

    def test_infa_native_imports_cannot_reuse_host_evaluate_package(self):
        import sys
        from tools.run_infa_release_stage import prepare_native_imports
        with patch.dict(sys.modules,{"evaluate.foreign_fixture":SimpleNamespace(__file__="/wrong.py")}), \
             patch.object(sys,"path",list(sys.path)):
            prepare_native_imports("/pinned/source")
            self.assertNotIn("evaluate",sys.modules)
            self.assertNotIn("evaluate.foreign_fixture",sys.modules)
            self.assertEqual(sys.path[0],"/pinned/source")

    def test_infa_protocol_checks_cannot_enter_training(self):
        from tools.run_infa_release_stage import run_stage
        with tempfile.TemporaryDirectory() as d:
            recipe={"working_directory":d,"merge":{"argv":["/unused"]}}
            record={"recipe":recipe,"minilm_path":"/unused","minilm_sha256":{}}
            (Path(d)/"maple-reproduction.json").write_text(json.dumps(record))
            with patch.object(repro,"verify_source"),patch.object(repro,"verify_files"), \
                 patch("tools.run_infa_release_stage.require_training_data"):
                with self.assertRaisesRegex(ValueError,"cannot enter"):
                    run_stage(recipe,"train",0,[],protocol_check=True)

    def test_explicit_bad_bert_directory_never_falls_back(self):
        ctx = DefenseContext("guardian", "q", "q", 0, [], SimpleNamespace(
            official_defense_guardian_bert_dir="/missing/encoder"))
        with self.assertRaisesRegex(ValueError, "BERT"):
            guardian._guardian_bert_parent(ctx)

if __name__ == "__main__":
    unittest.main()
