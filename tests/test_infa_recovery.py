import asyncio
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from openai.types.chat import ChatCompletion
from evaluate.defense_methods import reproduction as repro

POLICY = {"token_budgets": [1024,2048,4096,8192], "transport_attempts": 3, "transport_backoff_seconds": 0}
REQUEST = {"model":"Qwen/test","messages":[{"role":"user","content":"Question"}],"temperature":0,"max_tokens":1024}
def response(reason="stop", text="answer"):
    return ChatCompletion.model_validate({"id":"test","object":"chat.completion","created":0,"model":"Qwen/test",
        "choices":[{"index":0,"finish_reason":reason,"message":{"role":"assistant","content":text}}],
        "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
def factory():
    spec=importlib.util.find_spec("tools.infa_generation_recovery")
    assert spec is not None, "Resumable complete-response journal is not implemented"
    return importlib.import_module("tools.infa_generation_recovery").make_audited_create

class InfaRecoveryTests(unittest.TestCase):
    def test_evaluation_uses_the_same_pinned_source_as_training(self):
        from tools.run_appworld_matrix import extras
        from types import SimpleNamespace
        args=SimpleNamespace(infa_source="/pinned/training-source",infa_checkpoint="/trained/best.pth",minilm_model="/minilm")
        command,error=extras(args,"infa_guard_full","run")
        self.assertIsNone(error)
        self.assertEqual(command[command.index("--infa-code-dir")+1],args.infa_source)

    def test_stage_journal_requires_exact_complete_reply_count(self):
        async def upstream(_self,**kwargs):return response()
        with tempfile.TemporaryDirectory() as d:
            call=factory()(upstream,Path(d)/"api.jsonl",{},POLICY,Path(d)/"cache")
            asyncio.run(call(None,**REQUEST))
            module=importlib.import_module("tools.infa_generation_recovery")
            self.assertTrue(hasattr(module,"summarize_journal"),"Missing stage journal integrity gate")
            report=module.summarize_journal(Path(d)/"cache",1)
            self.assertEqual(report["accepted_responses"],1)
            self.assertEqual(len(report["journal_sha256"]),64)
            with self.assertRaisesRegex(ValueError,"count"):
                module.summarize_journal(Path(d)/"cache",64)

    def test_profile_records_recovery_without_changing_native_training(self):
        old=repro.infa_recipe("/source","/job","/data","Qwen/test",42,"qwen_no_thinking")
        try:new=repro.infa_recipe("/source","/job","/data","Qwen/test",42,"qwen_no_thinking_recover")
        except ValueError as e:self.fail(str(e))
        self.assertEqual(old["training"],new["training"])
        self.assertNotIn("generation_recovery",old)
        self.assertEqual(new["generation_recovery"]["token_budgets"],[1024,2048,4096,8192])
        self.assertTrue(new["model_substitution"])

    def test_truncation_retries_same_prompt_and_never_returns_partial(self):
        calls=[]
        async def upstream(_self,**kwargs):
            calls.append(kwargs)
            return response("length","partial") if len(calls)==1 else response()
        with tempfile.TemporaryDirectory() as d:
            call=factory()(upstream,Path(d)/"api.jsonl",{},POLICY,Path(d)/"cache")
            result=asyncio.run(call(None,**REQUEST))
            self.assertEqual(result.choices[0].message.content,"answer")
            self.assertEqual([c["max_tokens"] for c in calls],[1024,2048])
            self.assertEqual(calls[0]["messages"],calls[1]["messages"])
            entries=list((Path(d)/"cache").glob("*.json"))
            self.assertEqual(len(entries),1)
            self.assertNotIn("partial",entries[0].read_text())

    def test_budget_exhaustion_saves_no_invalid_response(self):
        calls=[]
        async def upstream(_self,**kwargs):
            calls.append(kwargs)
            return response("length","partial")
        with tempfile.TemporaryDirectory() as d:
            call=factory()(upstream,Path(d)/"api.jsonl",{},POLICY,Path(d)/"cache")
            with self.assertRaisesRegex(RuntimeError,"Incomplete"):
                asyncio.run(call(None,**REQUEST))
            self.assertEqual(len(calls),4)
            self.assertEqual(list((Path(d)/"cache").glob("*.json")),[])

    def test_restart_replays_complete_responses_and_rejects_request_drift(self):
        calls=[]
        async def upstream(_self,**kwargs):
            calls.append(kwargs);return response()
        with tempfile.TemporaryDirectory() as d:
            def fresh():return factory()(upstream,Path(d)/"api.jsonl",{},POLICY,Path(d)/"cache")
            asyncio.run(fresh()(None,**REQUEST))
            replay=asyncio.run(fresh()(None,**REQUEST))
            self.assertEqual(replay.choices[0].message.content,"answer")
            self.assertEqual(len(calls),1)
            changed=dict(REQUEST,messages=[{"role":"user","content":"different"}])
            with self.assertRaisesRegex(ValueError,"request"):
                asyncio.run(fresh()(None,**changed))

    def test_corrupted_response_cache_is_not_reused(self):
        async def upstream(_self,**kwargs):return response()
        with tempfile.TemporaryDirectory() as d:
            def fresh():return factory()(upstream,Path(d)/"api.jsonl",{},POLICY,Path(d)/"cache")
            asyncio.run(fresh()(None,**REQUEST))
            p=next((Path(d)/"cache").glob("*.json"));row=json.loads(p.read_text())
            row["response"]["choices"][0]["message"]["content"]="tampered"
            p.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,"hash"):
                asyncio.run(fresh()(None,**REQUEST))

    def test_concurrent_responses_replay_in_request_order(self):
        calls=[]
        async def upstream(_self,**kwargs):
            name=kwargs["messages"][0]["content"];calls.append(name)
            await asyncio.sleep(0.01 if name=="slow" else 0)
            return response(text=name)
        async def batch(call):
            return await asyncio.gather(*(call(None,**dict(REQUEST,messages=[{"role":"user","content":x}]))
                for x in ("slow","fast")))
        with tempfile.TemporaryDirectory() as d:
            def fresh():return factory()(upstream,Path(d)/"api.jsonl",{},POLICY,Path(d)/"cache")
            asyncio.run(batch(fresh()));results=asyncio.run(batch(fresh()))
            self.assertEqual([r.choices[0].message.content for r in results],["slow","fast"])
            self.assertEqual(calls,["slow","fast"])

    def test_transient_api_exception_is_bounded_and_auth_failure_not_retried(self):
        import httpx
        from openai import APIConnectionError,AuthenticationError
        for auth in (False,True):
            attempts=[]
            async def upstream(_self,**kwargs):
                attempts.append(1)
                request=httpx.Request("POST","https://example.invalid")
                if auth:raise AuthenticationError("invalid",response=httpx.Response(401,request=request),body={})
                if len(attempts)<3:raise APIConnectionError(request=request)
                return response()
            with tempfile.TemporaryDirectory() as d:
                call=factory()(upstream,Path(d)/"api.jsonl",{},POLICY,Path(d)/"cache")
                if auth:
                    with self.assertRaises(AuthenticationError):asyncio.run(call(None,**REQUEST))
                    self.assertEqual(len(attempts),1)
                else:
                    self.assertEqual(asyncio.run(call(None,**REQUEST)).choices[0].finish_reason,"stop")
                    self.assertEqual(len(attempts),3)

if __name__=="__main__":unittest.main()
