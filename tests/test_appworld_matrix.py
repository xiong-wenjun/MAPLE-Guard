import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from tools.run_appworld_matrix import MAIN, MECHANISM, GATES, IDENTITY_AUDIT, build_job, execute_job
from maple_guard import maple_guard_core as ep
from maple_guard import run_appworld, run_mmlu
from unittest.mock import patch

class MatrixTests(unittest.TestCase):
    def args(self,**changes):
        options=dict(profile="strict",phase="pilot",backbone="qwen",tasks=200,
            bundle="/tmp/tasks.json",run_root="/tmp/new-run",task_service="qwen",judge_service="judge",
            agentsafe_policy="",agentsafe_criteria="",agentsafe_threshold=None,
            infa_checkpoint="",minilm_model="",guardian_source="")
        options.update(changes)
        return SimpleNamespace(**options)
    def services(self):
        return {"qwen":{"base_url":"http://task/v1","model":"Qwen/Qwen3.5-122B-A10B","api_key":"SECRET"},
                "judge":{"base_url":"http://judge/v1","model":"Qwen/Qwen3.5-122B-A10B","api_key":"JUDGESECRET"},
                "embedding":{"base_url":"http://embedding/v1","model":"Qwen/Qwen3-Embedding-8B","api_key":"EMBEDSECRET"}}
    def test_14_arms_preserve_original_main_baselines(self):
        self.assertEqual(len(MAIN+MECHANISM),14)
        self.assertTrue({"challenger","gsafeguard","guardian"}.issubset(MAIN))
        for method in MAIN+MECHANISM+GATES+IDENTITY_AUDIT:self.assertIn(method,ep.METHOD_CHOICES)
    def test_inspector_identity_audit_kept_separate_from_main_row(self):
        job=build_job(self.args(),"inspector",42,self.services())
        self.assertEqual(job["table_role"],"identity_audit")
    def test_real_cli_resolves_every_prepared_arm(self):
        for method in MAIN+MECHANISM+GATES+IDENTITY_AUDIT:
            job=build_job(self.args(),method,42,self.services())
            if job["status"]=="blocked":continue
            command=job["command"]
            i=command.index("maple_guard.run_appworld")
            with patch("sys.argv",["run_appworld",*command[i+1:]]):
                args=run_mmlu.resolve_args(run_appworld.parse_args())
            self.assertEqual(args.method,method)
            self.assertEqual(args.memory_topology,"brokered-shared")
            self.assertFalse(args.exclude_attackers_from_final_vote)
            self.assertEqual(args.full_judge_base_url,"http://judge/v1")
            self.assertTrue(args.benchmark_bundle)
            self.assertEqual(args.chat_max_tokens,512)
    def test_no_credentials_in_command(self):
        job=build_job(self.args(),"maple_guard",42,self.services())
        self.assertNotIn("SECRET"," ".join(job["command"]))
    def test_paper_profile_does_not_quietly_use_corrected_flags(self):
        cmd=build_job(self.args(profile="paper-code"),"maple_guard",42,self.services())["command"]
        for flag in ("--strict-comparison","--peer-communication","--memory-topology","--asr-metric","--disable-chat-thinking"):
            self.assertNotIn(flag,cmd)
    def test_missing_full_assets_are_blocked_not_substituted(self):
        for method in ("agentsafe_full","infa_guard_full","guardian","gsafeguard"):
            job=build_job(self.args(),method,42,self.services())
            self.assertEqual(job["status"],"blocked")
            self.assertTrue(job["reason"])
    def test_smoke_only_changes_are_not_applied_to_200_task_pilot(self):
        smoke=build_job(self.args(phase="smoke",tasks=4),"maple_guard",42,self.services())["command"]
        pilot=build_job(self.args(),"maple_guard",42,self.services())["command"]
        self.assertIn("--warmup-tasks",smoke)
        self.assertNotIn("--warmup-tasks",pilot)

    def test_each_process_receives_isolated_memos_registry(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder:
            dirs=[]
            for method in ("maple_guard","no_defense_memrl"):
                job=build_job(self.args(run_root=folder),method,42,self.services())
                proc=Mock();proc.pid=123;proc.wait.return_value=1
                with patch("tools.run_appworld_matrix.subprocess.Popen",return_value=proc) as popen:
                    execute_job(job,{},200)
                env=popen.call_args.kwargs["env"]
                self.assertEqual(env["MEMOS_BASE_PATH"],str(Path(job["directory"])/"memos-runtime"))
                dirs.append(env["MEMOS_BASE_PATH"])
            self.assertNotEqual(*dirs)
