import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.run_recovery_plan import rewrite_job, retryable_failure, lane_lock, validate_result, run_lane

class RecoveryPlanTests(unittest.TestCase):
    def job(self):
        return {"run_id":"original","directory":"/old/run/original","method":"agentsafe_full",
                "command":["python","-B","/old/source/tools/run_instrumented.py","maple_guard.infa_memlink_eval",
                 "--methods","agentsafe_full","--output-root","/old/run/original",
                 "--memory-store-dir","/old/run/original/memory","--memory-run-id","original",
                 "--baseline-experiment-id","original","--baseline-state-path","/old/run/original/state.json",
                 "--config","configs/csqa_star.yaml","--max-tokens","1024"]}

    def test_restart_isolates_memory_and_preserves_generation_protocol(self):
        old=self.job()
        new=rewrite_job(old,Path("/fresh/retry"),Path("/new/source"),{"--chat-timeout":"600"})
        args=new["command"]
        self.assertEqual(old["directory"],"/old/run/original")
        self.assertEqual(args[args.index("--max-tokens")+1],"1024")
        self.assertEqual(args[args.index("--chat-timeout")+1],"600")
        self.assertEqual(args[args.index("--memory-store-dir")+1],"/fresh/retry/memory")
        self.assertEqual(args[args.index("--baseline-state-path")+1],"/fresh/retry/state.json")
        self.assertEqual(args[args.index("--memory-run-id")+1],"retry")
        self.assertEqual(args[args.index("--baseline-experiment-id")+1],"retry")
        self.assertIn("/new/source/tools/run_instrumented.py",args)
        self.assertEqual(args[args.index("--config")+1],"/old/source/configs/csqa_star.yaml")
        self.assertFalse(any("/old/run/original" in x for x in args))

    def test_override_cannot_silently_change_method_or_seed(self):
        for flag in ("--methods","--seed","--benchmark-bundle","--strict-comparison"):
            with self.subTest(flag=flag),self.assertRaises(ValueError):
                rewrite_job(self.job(),Path("/fresh/retry"),Path("/new/source"),{flag:"different"})

    def test_fresh_rewrite_accepts_only_explicit_string_response_policy(self):
        job=self.job();job['resolved_args']={'response_budget_policy':'strict'}
        new=rewrite_job(job,Path('/fresh/retry'),Path('/new/source'),{'--response-budget-policy':'fail_task'})
        self.assertEqual(new['resolved_args']['response_budget_policy'],'fail_task')
        self.assertEqual(new['command'][new['command'].index('--response-budget-policy')+1],'fail_task')
        self.assertNotIn('--resume-task-checkpoint',new['command'])
        self.assertEqual(job['resolved_args']['response_budget_policy'],'strict')
        with self.assertRaises(ValueError):
            rewrite_job(job,Path('/fresh/retry'),Path('/new/source'),{'--response-budget-policy':'relaxed'})

    def test_lane_lock_prevents_duplicate_controller(self):
        with tempfile.TemporaryDirectory() as folder:
            with lane_lock(Path(folder)/"lane.lock"):
                with self.assertRaises(BlockingIOError):
                    with lane_lock(Path(folder)/"lane.lock"):pass

    def test_retry_requires_transport_failure_and_never_accepts_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            log=root/"run.log";api=root/"api-calls.jsonl"
            log.write_text("Full baseline judge failed (HTTPError); strict comparison cannot use a fallback response.")
            api.write_text(json.dumps({"http_status":500})+"\n")
            self.assertTrue(retryable_failure(root))
            api.write_text(json.dumps({"finish_reasons":["length"],"invalid_for_benchmark":True})+"\n")
            self.assertFalse(retryable_failure(root))
            api.write_text(json.dumps({"error_type":"ReadTimeout"})+"\n")
            log.write_text("AttributeError: missing configuration")
            self.assertFalse(retryable_failure(root))

    def test_transport_after_committed_budget_failure_remains_recoverable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            event={'response_budget_policy':'fail_task','budget_outcomes_version':1,'request_id':'r','task_id':'t','role':'task','request_scope':'primary','invalid_response_type':'length','finish_reasons':['length']}
            row={'response_budget_policy':'fail_task','budget_outcomes_version':1,'task_id':'t','outcome':'budget_exhausted','is_correct':False,'budget_failures':[event]}
            (root/'trace.jsonl').write_text(json.dumps(row)+'\n')
            (root/'run.log').write_text('ReadTimeout: recover last task checkpoint')
            (root/'api-calls.jsonl').write_text(json.dumps({**event,'invalid_for_benchmark':True})+'\n'+json.dumps({'error_type':'ReadTimeout'})+'\n')
            self.assertTrue(retryable_failure(root))
            row['budget_failures'][0]['request_id']='forged'
            (root/'trace.jsonl').write_text(json.dumps(row)+'\n')
            self.assertFalse(retryable_failure(root))

    def test_completed_result_requires_matching_task_order_and_valid_responses(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"trace.jsonl").write_text('{"task_id":"b"}\n{"task_id":"a"}\n')
            (root/"trace.summary.json").write_text("{}")
            (root/"api-calls.jsonl").write_text("")
            self.assertFalse(validate_result(root,0,2,["a","b"])["valid"])
            self.assertTrue(validate_result(root,0,2,["b","a"])["valid"])
            (root/"api-calls.jsonl").write_text('{"invalid_for_benchmark":true}\n')
            self.assertFalse(validate_result(root,0,2,["b","a"])["valid"])


    def test_successful_seed_does_not_skip_the_next_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            first=self.job();first["seed"]=42
            second=json.loads(json.dumps(first).replace("original","second"));second["seed"]=43
            lane={"id":"lane","jobs":[first,second],"credentials":"unused"}
            plan={"recovery_root":str(root)}
            def run(job, env, lock, ids):
                path=Path(job["directory"]);path.mkdir(parents=True)
                job.update(status="completed",exit_code=0)
                (path/"run.json").write_text(json.dumps(job))
                return job
            with patch("tools.run_recovery_plan.environment",return_value={}), \
                 patch("tools.run_recovery_plan.running_directory",return_value=None), \
                 patch("tools.run_recovery_plan.run_job",side_effect=run) as execute:
                state=run_lane(plan,lane)
            self.assertEqual(state["status"],"completed")
            self.assertEqual(execute.call_count,2)
            self.assertNotEqual(state["completed_jobs"]["original"],state["completed_jobs"]["second"])

    def test_mmlu_task_budget_uses_its_public_cli_flag(self):
        job=self.job()
        job["command"]=[x.replace("maple_guard.infa_memlink_eval","maple_guard.run_mmlu") for x in job["command"]]
        new=rewrite_job(job,Path("/fresh/retry"),Path("/new/source"),{"--chat-max-tokens":"1024"})
        args=new["command"]
        self.assertEqual(args[args.index("--chat-max-tokens")+1],"1024")

    def test_audited_full_judge_budget_does_not_change_protocol_identity(self):
        job=self.job()
        job["method"]="amemguard_full"
        new=rewrite_job(job,Path("/fresh/retry"),Path("/new/source"),{"--full-judge-max-tokens":"8192"})
        command=new["command"]
        self.assertEqual(command[command.index("--full-judge-max-tokens")+1],"8192")
        self.assertEqual(new["method"],"amemguard_full")
        self.assertEqual(new["recovery_overrides"],{"--full-judge-max-tokens":"8192"})

    def test_recorded_resolved_budget_matches_restarted_command(self):
        job=self.job();job["resolved_args"]={"full_judge_max_tokens":4096,"chat_timeout":180}
        new=rewrite_job(job,Path("/fresh/retry"),Path("/new/source"),
                        {"--full-judge-max-tokens":"8192","--chat-timeout":"600"})
        self.assertEqual(new["resolved_args"]["full_judge_max_tokens"],8192)
        self.assertEqual(new["resolved_args"]["chat_timeout"],600.0)
        self.assertEqual(job["resolved_args"]["full_judge_max_tokens"],4096)

    def test_longmemeval_final_adjudicator_budget_uses_answer_judge_flag(self):
        job=self.job()
        job["command"]=[x.replace("maple_guard.infa_memlink_eval","maple_guard.run_longmemeval") for x in job["command"]]
        new=rewrite_job(job,Path("/fresh/retry"),Path("/new/source"),{"--answer-judge-max-tokens":"1024"})
        command=new["command"]
        self.assertEqual(command[command.index("--answer-judge-max-tokens")+1],"1024")

    def test_new_attempts_enable_durable_checkpoints(self):
        new=rewrite_job(self.job(),Path("/fresh/retry"),Path("/new/source"),{})
        self.assertIn("--task-checkpoint-dir",new["command"])
        self.assertNotIn("--resume-task-checkpoint",new["command"])

    def test_transport_retry_resumes_same_directory_instead_of_replaying_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            original=self.job();original["seed"]=42
            lane={"id":"lane","jobs":[original],"credentials":"unused"}
            launched=[]
            def run(job,env,lock,ids):
                launched.append(json.loads(json.dumps(job)))
                path=Path(job["directory"]);path.mkdir(parents=True,exist_ok=True)
                cp=path/"task-checkpoints";cp.mkdir(exist_ok=True)
                (cp/"latest.json").write_text("{}")
                job.update(status="failed" if len(launched)==1 else "completed",exit_code=1 if len(launched)==1 else 0)
                (path/"run.json").write_text(json.dumps(job))
                return job
            with patch("tools.run_recovery_plan.environment",return_value={}), \
                 patch("tools.run_recovery_plan.running_directory",return_value=None), \
                 patch("tools.run_recovery_plan.retryable_failure",return_value=True), \
                 patch("tools.run_recovery_plan.time.sleep"), \
                 patch("tools.run_recovery_plan.run_job",side_effect=run):
                state=run_lane({"recovery_root":str(root)},lane)
            self.assertEqual(state["status"],"completed")
            self.assertEqual(len(launched),2)
            self.assertEqual(launched[0]["directory"],launched[1]["directory"])
            self.assertIn("--resume-task-checkpoint",launched[1]["command"])
            self.assertNotIn("--checkpoint-allow-budget-change",launched[1]["command"])

    def test_rewrite_never_carries_resume_flag_to_a_new_directory(self):
        old=self.job()
        old["command"]+=["--task-checkpoint-dir","/old/run/original/task-checkpoints","--resume-task-checkpoint","--checkpoint-allow-budget-change"]
        new=rewrite_job(old,Path("/fresh/retry"),Path("/new/source"),{})
        self.assertNotIn("--resume-task-checkpoint",new["command"])
        self.assertNotIn("--checkpoint-allow-budget-change",new["command"])

    def test_new_controller_resumes_an_original_checkpoint_instead_of_allocating_fresh_run(self):
        from tools.run_recovery_plan import ROOT
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            original=rewrite_job(self.job(),root/"existing",ROOT,{})
            original["seed"]=42;original["status"]="running"
            Path(original["directory"]).mkdir()
            cp=Path(original["directory"])/"task-checkpoints";cp.mkdir()
            (cp/"latest.json").write_text("{}")
            (Path(original["directory"])/"run.json").write_text(json.dumps(original))
            lane={"id":"lane","jobs":[original],"credentials":"unused"}
            launched=[]
            def run(job,env,lock,ids):
                launched.append(job)
                job.update(status="completed",exit_code=0)
                return job
            with patch("tools.run_recovery_plan.environment",return_value={}), \
                 patch("tools.run_recovery_plan.running_directory",return_value=None), \
                 patch("tools.run_recovery_plan.run_job",side_effect=run):
                state=run_lane({"recovery_root":str(root/"new-controller")},lane)
            self.assertEqual(state["status"],"completed")
            self.assertEqual(launched[0]["directory"],original["directory"])
            self.assertIn("--resume-task-checkpoint",launched[0]["command"])
