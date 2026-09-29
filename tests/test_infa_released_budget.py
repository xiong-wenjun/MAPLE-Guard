"""Released 1024-token content consumption is separate from strict recovery/evaluation."""
import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from openai.types.chat import ChatCompletion
from evaluate.defense_methods import reproduction as repro
from tools import infa_generation_recovery as recovery

PROFILE="qwen_no_thinking_released_budget"
POLICY={"token_budgets":[1024],"accepted_finish_reasons":["stop","length"],
        "transport_attempts":3,"transport_backoff_seconds":0}
REQUEST={"model":"Qwen/test","messages":[{"role":"user","content":"Question"}],"temperature":0,"max_tokens":1024}

def response(reason="stop",text="answer"):
    return ChatCompletion.model_validate({"id":"test","object":"chat.completion","created":0,"model":"Qwen/test",
        "choices":[{"index":0,"finish_reason":reason,"message":{"role":"assistant","content":text}}],
        "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})

def invoke(original,audit,cache,policy=POLICY):
    try:return recovery.make_audited_create(original,audit,{},policy,cache)
    except ValueError as exc:raise AssertionError("Released-budget policy is not implemented") from exc

def cache_row(directory,ordinal,budget=1024,reason="stop"):
    payload=response(reason).model_dump(mode="json")
    row={"ordinal":ordinal,"request_sha256":recovery.payload_hash(REQUEST),"model":"Qwen/test",
        "accepted_max_tokens":budget,"response":payload,"response_sha256":recovery.payload_hash(payload)}
    directory.mkdir(exist_ok=True)
    (directory/f"{ordinal:06d}.json").write_text(json.dumps(row))
    return row

class ReleasedBudgetTests(unittest.TestCase):
    def test_profile_preserves_native_generation_and_training_argv(self):
        old=repro.infa_recipe("/source","/job","/data","Qwen/test",42,"qwen_no_thinking")
        try:new=repro.infa_recipe("/source","/job","/data","Qwen/test",42,PROFILE)
        except ValueError as exc:self.fail(str(exc))
        for key in ("generation","training","source_files","expected_dialogues"):
            self.assertEqual(old[key],new[key])
        self.assertEqual(new["generation_recovery"]["token_budgets"],[1024])
        self.assertEqual(new["generation_recovery"]["accepted_finish_reasons"],["stop","length"])
        legacy=repro.infa_recipe("/source","/job","/data","Qwen/test",42,"qwen_no_thinking_recover")
        self.assertEqual(legacy["generation_recovery"]["token_budgets"],[1024,2048,4096,8192])
        self.assertNotIn("accepted_finish_reasons",legacy["generation_recovery"])

    def test_nonempty_length_is_returned_and_cached_without_extra_generation(self):
        calls=[]
        async def upstream(_self,**kwargs):calls.append(kwargs);return response("length","nonempty capped content")
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);call=invoke(upstream,root/"audit.jsonl",root/"cache")
            result=asyncio.run(call(None,**REQUEST))
            self.assertEqual(result.choices[0].finish_reason,"length")
            self.assertEqual(calls,[REQUEST])
            row=json.loads((root/"cache/000000.json").read_text())
            self.assertEqual(row["accepted_max_tokens"],1024)
            replay=asyncio.run(invoke(upstream,root/"audit.jsonl",root/"cache")(None,**REQUEST))
            self.assertEqual(replay.choices[0].message.content,"nonempty capped content")
            self.assertEqual(len(calls),1)
            events=[json.loads(line) for line in (root/"audit.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["finish_reason"],"length")

    def test_empty_stop_or_length_never_becomes_a_training_reply(self):
        for reason in ("stop","length"):
            calls=[]
            async def upstream(_self,**kwargs):calls.append(kwargs);return response(reason," ")
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);call=invoke(upstream,root/"audit.jsonl",root/"cache")
                with self.assertRaisesRegex(RuntimeError,"Incomplete|empty"):
                    asyncio.run(call(None,**REQUEST))
                self.assertEqual(len(calls),1)
                self.assertEqual(list((root/"cache").glob("*.json")),[])

    def test_transport_retries_are_still_bounded_with_fixed_budget(self):
        import httpx
        from openai import APIConnectionError
        calls=[]
        async def upstream(_self,**kwargs):
            calls.append(kwargs)
            if len(calls)<3:raise APIConnectionError(request=httpx.Request("POST","https://example.invalid"))
            return response("length")
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);asyncio.run(invoke(upstream,root/"audit.jsonl",root/"cache")(None,**REQUEST))
            self.assertEqual([c["max_tokens"] for c in calls],[1024]*3)

    def test_summary_counts_finish_reasons_and_rejects_expanded_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache_row(root,0);cache_row(root,1,reason="length")
            try:report=recovery.summarize_journal(root,2,POLICY)
            except TypeError as exc:self.fail("Summary does not support declared profile policy: "+str(exc))
            self.assertEqual(report["accepted_finish_reasons"],{"stop":1,"length":1})
            with self.assertRaisesRegex(ValueError,"incomplete"):
                recovery.summarize_journal(root,2)
            cache_row(root,1,budget=2048)
            with self.assertRaisesRegex(ValueError,"budget"):
                recovery.summarize_journal(root,2,POLICY)

    def test_import_only_copies_contiguous_1024_stop_prefix_and_records_source_hashes(self):
        self.assertTrue(hasattr(recovery,"import_released_prefix"),"Missing explicit prefix importer")
        for stop_kind in ("expanded","missing","length"):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);source=root/"old";destination=root/"new"
                cache_row(source,0);cache_row(source,1)
                if stop_kind!="missing":cache_row(source,2,budget=2048 if stop_kind=="expanded" else 1024,reason="length" if stop_kind=="length" else "stop")
                cache_row(source,3)
                report=recovery.import_released_prefix(source,destination)
                self.assertEqual(report["imported_responses"],2)
                self.assertEqual(sorted(p.name for p in destination.glob("*.json")),["000000.json","000001.json"])
                self.assertEqual(report["source_journal"],str(source.resolve()))
                self.assertEqual(report["source_files_sha256"]["000000.json"],hashlib.sha256((source/"000000.json").read_bytes()).hexdigest())
                self.assertEqual((source/"000000.json").read_bytes(),(destination/"000000.json").read_bytes())
                self.assertTrue((destination/"import-provenance/provenance.json").is_file())
                with self.assertRaises(FileExistsError):recovery.import_released_prefix(source,destination)

    def test_imported_prefix_replay_checks_new_request_hash_and_local_cache_hash(self):
        self.assertTrue(hasattr(recovery,"import_released_prefix"),"Missing explicit prefix importer")
        async def never(_self,**kwargs):raise AssertionError("Must not call API on bad replay")
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/"old";destination=root/"new";cache_row(source,0)
            recovery.import_released_prefix(source,destination)
            call=invoke(never,root/"audit.jsonl",destination)
            with self.assertRaisesRegex(ValueError,"request"):
                asyncio.run(call(None,**dict(REQUEST,messages=[])))
            row=json.loads((destination/"000000.json").read_text());row["response"]["choices"][0]["message"]["content"]="tampered"
            (destination/"000000.json").write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,"hash"):
                asyncio.run(invoke(never,root/"audit.jsonl",destination)(None,**REQUEST))

    def test_new_profile_cannot_be_paired_with_escalating_budget_policy(self):
        from test_infa_pipeline import Fixture,write
        from tools.run_infa_training_pipeline import Pipeline
        with tempfile.TemporaryDirectory() as d:
            fixture=Fixture(d);fixture.recipe["generation_profile"]=PROFILE
            fixture.recipe["generation_recovery"]=dict(POLICY,token_budgets=[1024,2048,4096,8192])
            write(Path(fixture.args.recipe),fixture.recipe)
            write(fixture.job/"maple-reproduction.json",{"recipe":fixture.recipe})
            with self.assertRaisesRegex(ValueError,"1024|policy"):
                Pipeline(fixture.args,runner=fixture.execute)._preflight()

    def test_import_provenance_detects_locally_rehashed_content(self):
        self.assertTrue(hasattr(recovery,"import_released_prefix"),"Missing explicit prefix importer")
        async def never(_self,**kwargs):raise AssertionError("Must not call API")
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/"old";destination=root/"new";cache_row(source,0)
            recovery.import_released_prefix(source,destination)
            summary=recovery.summarize_journal(destination,1,POLICY)
            self.assertEqual(len(summary["import_provenance_sha256"]),64)
            path=destination/"000000.json";row=json.loads(path.read_text())
            row["response"]["choices"][0]["message"]["content"]="locally changed"
            row["response_sha256"]=recovery.payload_hash(row["response"]);path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,"source hash"):
                asyncio.run(invoke(never,root/"audit.jsonl",destination)(None,**REQUEST))
            with self.assertRaisesRegex(ValueError,"source hash"):
                recovery.summarize_journal(destination,1,POLICY)

    def test_stage_audit_dispatches_released_budget_without_changing_legacy_audit(self):
        from openai.resources.chat.completions import AsyncCompletions
        from tools.run_infa_release_stage import install_api_audit
        async def upstream(_self,**kwargs):return response("length")
        for policy,succeeds in ((POLICY,True),(None,False)):
            with tempfile.TemporaryDirectory() as d,patch.object(AsyncCompletions,"create",upstream):
                root=Path(d);install_api_audit(root/"audit.jsonl",{},policy,root/"cache")
                if succeeds:
                    result=asyncio.run(AsyncCompletions.create(None,**REQUEST))
                    self.assertEqual(result.choices[0].finish_reason,"length")
                else:
                    with self.assertRaisesRegex(RuntimeError,"Incomplete"):
                        asyncio.run(AsyncCompletions.create(None,**REQUEST))

    def test_controller_allows_explicit_released_budget_profile(self):
        from test_infa_pipeline import Fixture,write
        from tools.run_infa_training_pipeline import Pipeline
        with tempfile.TemporaryDirectory() as d:
            fixture=Fixture(d);fixture.recipe["generation_profile"]=PROFILE;fixture.recipe["generation_recovery"]=POLICY
            write(Path(fixture.args.recipe),fixture.recipe)
            write(fixture.job/"maple-reproduction.json",{"recipe":fixture.recipe})
            try:Pipeline(fixture.args,runner=fixture.execute)._preflight()
            except ValueError as exc:self.fail("Controller rejected declared profile: "+str(exc))

if __name__=="__main__":unittest.main()
