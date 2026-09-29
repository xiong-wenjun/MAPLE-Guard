"""Regression coverage for concurrent grids, interrupted publication and source migration."""
import asyncio
import copy
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_infa_release_stage as stage
from tools import infa_generation_recovery as recovery
from test_infa_released_budget import POLICY, REQUEST, response


def rows():
    return [{"question":f"training question {i}","adj_matrix":[[0]*8 for _ in range(8)],
             "system_prompts":["prompt"]*8,"attacker_idxes":[0],"infected_idxes":[],
             "infected_idxes_per_turn":[[] for _ in range(4)],
             "communication_data":[[[n,"reply"] for n in range(8)] for _ in range(4)]}
            for i in range(40)]


def fixture(root):
    root=Path(root);job=root/"job";job.mkdir(exist_ok=True)
    formal=job/"graphs/PI/csqa/train"
    recipe={"working_directory":str(job),"source_revision":"native-pinned",
            "generator_model":"Qwen/test","model_substitution":True,
            "generation_profile":"qwen_no_thinking_released_budget","generation_recovery":POLICY,
            "request_overrides":{},"merge":{"argv":[str(formal.parent)]},
            "generation":[{"seed":42+i,"attackers":1,"sparsity":v,
               "argv":["generate.py","--save_dir",str(job/"graphs"),"--samples","40"]}
               for i,v in enumerate((.2,.4))]}
    (job/"maple-reproduction.json").write_text(json.dumps({"recipe":recipe,"minilm_path":"unused","minilm_sha256":{}}))
    services=root/"services.json";services.write_text(json.dumps({"fixture":{"model":"Qwen/test","api_key":"test","base_url":"https://invalid"}}))
    return recipe,formal,services


def name(index):return f"fixture-num_attackers_1-sparsity_{(.2,.4)[index]}.json"


def run_fixture(recipe,services,index=0,changed=False,other=False,barrier=None):
    formal=Path(recipe["merge"]["argv"][-1])/"train"
    def native(*args,**kwargs):
        save=Path(sys.argv[sys.argv.index("--save_dir")+1])/"PI/csqa/train"
        assert save!=formal, "Parallel native writers must use isolated directories"
        assert save.is_dir(), "Precreate output directory to remove native check/mkdir race"
        if barrier:barrier.wait(timeout=20)
        data=rows()
        if changed:data[0]["infected_idxes_per_turn"][0]=[1]
        (save/name(index)).write_text(json.dumps(data))
        if other:
            formal.mkdir(parents=True,exist_ok=True)
            (formal/name(1-index)).write_text(json.dumps(rows()))
    with patch.object(stage.r,"verify_source"),patch.object(stage,"require_training_data"),\
         patch.object(stage.r,"verify_files"),patch.object(stage.r,"heldout_questions",return_value=[]),\
         patch.object(stage,"prepare_native_imports"),patch.object(stage,"install_api_audit") as audit,\
         patch.object(stage.runpy,"run_path",side_effect=native),\
         patch.object(recovery,"summarize_journal",return_value={"accepted_responses":1280,"journal_sha256":"fixture"}):
        result=stage.run_stage(recipe,"generate",index,[],str(services),"fixture")
        return result,audit.call_args


def concurrent_worker(recipe,services,index,barrier,queue):
    try:queue.put((index,run_fixture(recipe,services,index,barrier=barrier)[0]["status"]))
    except BaseException as exc:queue.put((index,type(exc).__name__+": "+str(exc)))


class GridRecoveryTests(unittest.TestCase):
    def test_other_grid_completion_does_not_contaminate_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe,formal,services=fixture(directory)
            result,_=run_fixture(recipe,services,other=True)
            self.assertEqual(result["status"],"completed")
            self.assertEqual(Path(result["output"]).name,name(0))
            self.assertEqual(len(list(formal.glob("*.json"))),2)

    def test_existing_output_is_verified_by_cache_only_native_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe,formal,services=fixture(directory);formal.mkdir(parents=True)
            path=formal/name(0);path.write_text(json.dumps(rows()));before=path.read_bytes()
            result,audit=run_fixture(recipe,services)
            self.assertEqual(path.read_bytes(),before)
            self.assertTrue(audit.kwargs["replay_only"])
            self.assertEqual(result["output_recovery"]["verification"],"cache-only native replay matched all dialogue fields")

    def test_existing_output_must_match_replayed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe,formal,services=fixture(directory);formal.mkdir(parents=True)
            path=formal/name(0);path.write_text(json.dumps(rows()));before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,"replay|differ"):
                run_fixture(recipe,services,changed=True)
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse((Path(recipe["working_directory"])/"stage-results/generate-00.json").exists())

    def test_concurrent_first_grids_have_no_shared_native_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe,formal,services=fixture(directory)
            context=multiprocessing.get_context("fork");barrier=context.Barrier(2);queue=context.Queue()
            processes=[context.Process(target=concurrent_worker,args=(recipe,services,i,barrier,queue)) for i in range(2)]
            for process in processes:process.start()
            for process in processes:process.join(30)
            for process in processes:
                if process.is_alive():process.terminate();process.join();self.fail("Concurrent generation hung")
            results=sorted(queue.get(timeout=5) for _ in processes)
            self.assertEqual(results,[(0,"completed"),(1,"completed")])
            self.assertEqual(len(list(formal.glob("*.json"))),2)

    def test_cache_only_recovery_never_calls_api_for_missing_entry(self):
        async def forbidden(*args,**kwargs):self.fail("Recovery must not spend an API call")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertIn("replay_only",__import__("inspect").signature(recovery.make_audited_create).parameters,
                          "Missing fail-closed cache-only mode")
            call=recovery.make_audited_create(forbidden,root/"audit",{},POLICY,root/"cache",replay_only=True)
            with self.assertRaisesRegex(ValueError,"missing|Missing"):
                asyncio.run(call(None,**REQUEST))


class SourceMigrationTests(unittest.TestCase):
    def test_explicit_migration_preserves_verified_stage_hash_and_old_history(self):
        from test_infa_pipeline import Fixture,write
        from tools import run_infa_training_pipeline as pipeline
        with tempfile.TemporaryDirectory() as directory,patch.object(pipeline,"source_fingerprint",return_value="fixture-source"),\
             patch.object(recovery,"summarize_journal",return_value={"accepted_responses":1280,"journal_sha256":"fixture-journal"}):
            f=Fixture(directory);old=f.pipeline();old._preflight();f.marker("generate-00");old._record_stage("generate-00")
            old._update(status="failed",attempts={"generate-01":3})
            before=old.state_path.read_bytes();args=copy.copy(f.args)
            args.resume_from=str(old.run_root);args.run_root=str(Path(directory)/"recovery")
            new=pipeline.Pipeline(args,runner=f.execute);new._preflight()
            self.assertEqual(old.state_path.read_bytes(),before)
            self.assertEqual(new.state["stages"],old.state["stages"])
            self.assertEqual(new.state["attempts"],{})
            self.assertTrue((new.run_root/"source-transition.json").is_file())

    def test_migration_rejects_changed_old_marker(self):
        from test_infa_pipeline import Fixture,write
        from tools import run_infa_training_pipeline as pipeline
        with tempfile.TemporaryDirectory() as directory,patch.object(pipeline,"source_fingerprint",return_value="fixture-source"),\
             patch.object(recovery,"summarize_journal",return_value={"accepted_responses":1280,"journal_sha256":"fixture-journal"}):
            f=Fixture(directory);old=f.pipeline();old._preflight();f.marker("generate-00");old._record_stage("generate-00");old._update(status="failed")
            marker=old.results/"generate-00.json";value=json.loads(marker.read_text());value["extra"]="changed";write(marker,value)
            args=copy.copy(f.args);args.resume_from=str(old.run_root);args.run_root=str(Path(directory)/"recovery")
            with self.assertRaisesRegex(ValueError,"hash|changed"):
                pipeline.Pipeline(args,runner=f.execute)._preflight()

if __name__=="__main__":unittest.main()
