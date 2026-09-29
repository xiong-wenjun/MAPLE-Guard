import contextlib
import hashlib
import io
import json
import tempfile

import yaml
import unittest
from pathlib import Path
from types import SimpleNamespace
from tools import run_appworld_matrix as matrix
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


class TopologyMatrixTests(unittest.TestCase):
    services = MatrixTests.services
    def parse_matrix_args(self, *extra):
        with patch("sys.argv", ["matrix", "--bundle", "/tmp/bundle.json",
                               "--services", "/tmp/services.json", "--run-root", "/tmp/fresh", *extra]):
            return matrix.arguments()

    def prepare(self, folder, *extra):
        folder = Path(folder)
        bundle = folder / "bundle.json"
        rows = [{"benchmark": "appworld", "task_id": task_id, "data": {}}
                for task_id in ("second", "first")]
        bundle.write_text(json.dumps({"benchmark": "appworld", "task_count": 2, "tasks": rows}))
        services = folder / "services.json"
        services.write_text(json.dumps(self.services()))
        services.chmod(0o600)
        root = folder / "matrix"
        argv = ["matrix", "--bundle", str(bundle), "--services", str(services),
                "--run-root", str(root), "--task-service", "qwen", "--judge-service", "judge",
                "--tasks", "2", "--methods", "maple_guard", "amemguard_full", *extra]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(matrix.main(), 0)
        return root, json.loads((root / "matrix.json").read_text())

    def resolve(self, job):
        command = job["command"]
        index = command.index("maple_guard.run_appworld")
        with patch("sys.argv", ["run_appworld", *command[index + 1:]]):
            return run_mmlu.resolve_args(run_appworld.parse_args())

    def test_default_and_full_aliases(self):
        args = self.parse_matrix_args()
        self.assertEqual(args.seeds, [42])
        self.assertEqual(getattr(args, "topologies", None), ["star"])
        for alias in ("full", "fully-connected", "complete"):
            self.assertEqual(self.parse_matrix_args("--topologies", alias).topologies, ["full"])

    def test_duplicate_axes_and_unknown_topology_fail_before_writing(self):
        for options in (("--seeds", "42", "42"), ("--topologies", "star", "star"),
                        ("--topologies", "full", "complete"), ("--topologies", "mesh"),
                        ("--methods", "maple_guard", "maple_guard")):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse_matrix_args(*options)

    def test_fifteen_seeds_four_topologies_have_unique_identity_and_state(self):
        seeds = [42, *range(14)]
        topologies = ["star", "chain", "tree", "full"]
        with tempfile.TemporaryDirectory() as folder:
            _, manifest = self.prepare(folder, "--seeds", *map(str, seeds), "--topologies", *topologies)
            jobs = manifest["jobs"]
            self.assertEqual(len(jobs), 15 * 4 * 2)
            self.assertEqual(manifest["seeds"], seeds)
            self.assertEqual(manifest["topologies"], topologies)
            self.assertEqual([(j["seed"], j["topology"], j["method"]) for j in jobs],
                             [(s, t, m) for s in seeds for t in topologies
                              for m in ("maple_guard", "amemguard_full")])
            for key in ("run_id", "directory"):
                self.assertEqual(len({j[key] for j in jobs}), len(jobs))
            for flag in ("--trace-id", "--memory-run-id", "--baseline-experiment-id",
                         "--memory-store-dir", "--baseline-state-path", "--out"):
                self.assertEqual(len({j["command"][j["command"].index(flag) + 1] for j in jobs}), len(jobs))
            amem = [j for j in jobs if j["method"] == "amemguard_full"]
            self.assertEqual(len({self.resolve(j).amemguard_experiment_id for j in amem}), len(amem))
            self.assertEqual(manifest["dataset"]["ordered_task_ids"], ["second", "first"])

    def test_frozen_configs_match_manifest_and_real_cli_topology(self):
        topologies = ("star", "chain", "tree", "full", "random")
        with tempfile.TemporaryDirectory() as folder:
            root, manifest = self.prepare(folder, "--topologies", *topologies)
            self.assertEqual(set(manifest["config_snapshots"]), set(topologies))
            self.assertEqual((root / "appworld_star.snapshot.yaml").read_bytes(),
                             (matrix.ROOT / "configs/appworld_star.yaml").read_bytes())
            reference_args = {}
            runtime_fields = {"config", "trace_id", "memory_run_id", "baseline_experiment_id",
                              "memory_store_dir", "baseline_state_path", "out", "amemguard_experiment_id",
                              "communication_topology"}
            for job in manifest["jobs"]:
                config = manifest["config_snapshots"][job["topology"]]
                path = Path(config["path"])
                self.assertEqual(job["config_snapshot"], config)
                self.assertEqual(job["command"][job["command"].index("--config") + 1], str(path))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), config["sha256"])
                source = matrix.ROOT / config["source"]
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), config["source_sha256"])
                self.assertEqual(yaml.safe_load(path.read_text())["communication"]["topology"], job["topology"])
                resolved = self.resolve(job)
                self.assertEqual(resolved.communication_topology, job["topology"])
                self.assertEqual(resolved.memory_topology, "brokered-shared")
                algorithm_args = {k: v for k, v in vars(resolved).items() if k not in runtime_fields}
                reference_args.setdefault(job["method"], algorithm_args)
                self.assertEqual(algorithm_args, reference_args[job["method"]])
                self.assertEqual(len(ep.build_adj_matrix(resolved.communication_topology, 8, resolved.seed)), 8)
                self.assertNotIn("SECRET", json.dumps(job))
            before = {path: path.read_bytes() for path in root.iterdir() if path.suffix == ".yaml"}
            with patch("sys.argv", ["matrix", "--bundle", str(Path(folder) / "bundle.json"),
                                   "--services", str(Path(folder) / "services.json"), "--run-root", str(root),
                                   "--task-service", "qwen", "--judge-service", "judge"]):
                with self.assertRaisesRegex(ValueError, "Fresh run root"):
                    matrix.main()
            self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_star_resolved_algorithm_settings_match_legacy_command(self):
        for phase in ("smoke", "pilot"):
            for profile in ("strict", "paper-code"):
                with self.subTest(phase=phase, profile=profile), tempfile.TemporaryDirectory() as folder:
                    _, manifest = self.prepare(folder, "--phase", phase, "--profile", profile, "--tasks", "200",
                                               "--methods", "maple_guard", "no_defense_memrl")
                    for job in manifest["jobs"]:
                        command = list(job["command"])
                        legacy_id = f"{phase}_{profile}_qwen_{job['method']}_s42"
                        self.assertEqual(job["legacy_run_id"], legacy_id)
                        command[command.index("--config") + 1] = str(matrix.ROOT / "configs/appworld_star.yaml")
                        index = command.index("--communication-topology")
                        del command[index:index + 2]
                        legacy = dict(job, command=[arg.replace(job["run_id"], legacy_id) for arg in command])
                        new_args, old_args = vars(self.resolve(job)), vars(self.resolve(legacy))
                        runtime_fields = {"config", "trace_id", "memory_run_id", "baseline_experiment_id",
                                          "memory_store_dir", "baseline_state_path", "out", "amemguard_experiment_id"}
                        self.assertEqual({k: v for k, v in new_args.items() if k not in runtime_fields},
                                         {k: v for k, v in old_args.items() if k not in runtime_fields})
