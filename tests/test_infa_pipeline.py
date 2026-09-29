"""Control-flow tests use small files and a fake subprocess boundary, never models/APIs."""
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def implementation():
    spec = importlib.util.find_spec("tools.run_infa_training_pipeline")
    assert spec is not None, "INFA training-to-test controller is not implemented"
    return importlib.import_module("tools.run_infa_training_pipeline")


class Fixture:
    def __init__(self, directory, workers=2, real_journal=False):
        self.base = Path(directory)
        self.real_journal = real_journal
        self.job = self.base / "job"
        self.results = self.job / "stage-results"
        self.results.mkdir(parents=True)
        self.formal = self.job / "graphs/PI/csqa/train"
        (self.job/"native.py").write_text("# pinned source fixture\n")
        self.recipe = {"working_directory": str(self.job), "source_revision": "pinned",
            "generation_profile": "qwen_no_thinking_recover", "generator_model": "Qwen/test",
            "source_files":{"native.py":digest(self.job/"native.py")},
            "expected_dialogues": 800,
            "generation": [{"seed":42+i,"attackers":i//5+1,"sparsity":[.2,.4,.6,.8,1.][i%5],
                            "argv":["generate_data/gen_graph.py","--samples","40"]} for i in range(20)],
            "merge":{"argv":["merge.py", str(self.formal.parent)]},
            "training":{"argv":["train/trainer.py","--epochs","50","--dataset_path",str(self.job/"features.pkl")]}}
        write(self.base / "recipe.json", self.recipe)
        write(self.job / "maple-reproduction.json", {"recipe":self.recipe})
        self.heldout = []
        for i in range(5):
            path = self.base / f"heldout{i}.json"
            write(path, {"question":f"heldout{i}"})
            self.heldout.append(str(path))
        write(self.base / "bundle.json", {"tasks":list(range(200))})
        write(self.base / "services.json", {"inference1":{"model":"Qwen/test","api_key":"SECRET"}})
        self.args = SimpleNamespace(recipe=str(self.base/"recipe.json"),heldout=self.heldout,
            services=str(self.base/"services.json"), generation_services=["inference1","inference2"],
            bundle=str(self.base/"bundle.json"),minilm_model=str(self.base/"minilm"),
            run_root=str(self.base/"pipeline"),workers=workers)
        self.calls = []
        self.failures = Counter()
        self.active = 0
        self.maximum = 0
        self.generation_active = 0
        self.generation_maximum = 0
        self.pilots_active = 0
        self.pilots_maximum = 0
        self.lock = threading.Lock()
        self.bad_epoch = False
        self.warning = False

    def marker(self, label):
        stage = "generate" if label.startswith("generate") else label
        value = {"status":"completed","stage":stage,"source_revision":"pinned",
            "generation_profile":"qwen_no_thinking_recover","generator_model":"Qwen/test"}
        if stage == "generate":
            index = int(label.split("-")[1]); grid = self.recipe["generation"][index]
            protocol = label.endswith("protocol")
            folder = self.job/"protocol-check/PI/csqa/train" if protocol else self.formal
            output = folder / f"data-num_attackers_{grid['attackers']}-sparsity_{grid['sparsity']}.json"
            count = 2 if protocol else 40
            write(output, [{"question":f"train{n}"} for n in range(count)])
            value.update(protocol_check=protocol,seed=grid["seed"],validation={"dialogues":count},
                         output=str(output),sha256=digest(output))
            from tools.infa_generation_recovery import payload_hash, summarize_journal
            journal=self.results/(label+"-responses")
            shared=self.results/f"fixture-responses-{count}"
            if self.real_journal:
                with self.lock:
                    self.prepare_journal(shared,count)
                if not journal.exists():journal.symlink_to(shared,target_is_directory=True)
            else:
                journal.mkdir(exist_ok=True)
            value.update(recovery_journal=str(journal),generation_integrity=summarize_journal(journal,count*8*4))
        elif stage in ("merge","embed"):
            output = self.formal/"dataset.json" if stage == "merge" else self.job/"features.pkl"
            write(output, [1]*800 if stage == "merge" else ["embedding"])
            value.update(output=str(output),sha256=digest(output))
        else:
            return self.training_marker(label,value)
        write(self.results/(label+".json"), value)
        return value

    def prepare_journal(self, shared, count):
        from tools.infa_generation_recovery import payload_hash
        if not shared.exists():
            shared.mkdir()
            response={"id":"fixture","object":"chat.completion","created":0,"model":"Qwen/test",
                "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"reply"}}]}
            for ordinal in range(count*8*4):
                write(shared/f"{ordinal:06d}.json",{"ordinal":ordinal,"request_sha256":"0"*64,
                    "response":response,"response_sha256":payload_hash(response),"accepted_max_tokens":1024})

    def training_marker(self,label,value):
        output = self.job/"checkpoints/csqa/ours/best-epochs_50.pth"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"checkpoint fixture")
        (output.parent/"latest_model_path.txt").write_text(str(output))
        lines = ["Training parameters: epochs=50, lr=0.001, batch_size=32"]
        lines += [f"Epoch {i}/50 || Train Loss: 0.1, Acc(both/mal/inf): 1.00/2.00/3.00% || Test Loss: 0.2, Acc(both/mal/inf): 1.00/2.00/3.00%" + (" || Save!" if i == 0 else "") for i in range(49 if self.bad_epoch else 50)]
        lines += ["Training completed. Best test accuracy: 2.00%",f"Model saved to: {output}"]
        (self.results/"train.log").write_text("\n".join(lines))
        value["checkpoints"] = {str(output):digest(output)}
        write(self.results/(label+".json"), value)
        return value

    def execute(self, label, command, log_path, cpu=True):
        with self.lock:
            self.calls.append((label, command, cpu)); self.active += 1
            self.maximum = max(self.maximum, self.active)
            if label.startswith("generate"):
                self.generation_active += 1
                self.generation_maximum = max(self.generation_maximum,self.generation_active)
            if label.startswith("pilot"):
                self.pilots_active += 1
                self.pilots_maximum = max(self.pilots_maximum,self.pilots_active)
        try:
            time.sleep(.03 if label.startswith("pilot") else .004)
            if self.failures[label]:
                self.failures[label] -= 1
                return 1
            if label.startswith("generate") or label in ("merge","embed","train"):
                self.marker(label)
            elif label == "validate-checkpoint":
                checkpoint = Path(command[command.index("--checkpoint")+1])
                output=Path(command[command.index("--output")+1]) if "--output" in command else log_path
                write(output, {"strict_load":True,"forward_shape":[2,2],"checkpoint_sha256":digest(checkpoint)})
                if "--output" in command:
                    log_path.parent.mkdir(parents=True,exist_ok=True)
                    log_path.write_text("FutureWarning: fixture diagnostic on stderr\n")
            else:
                phase = command[command.index("--phase")+1]
                count = int(command[command.index("--tasks")+1])
                backbone = command[command.index("--backbone")+1]
                root = Path(command[command.index("--run-root")+1])
                run_id = f"{phase}_strict_{backbone}_infa_guard_full_star_s42"
                run_dir = root/run_id; run_dir.mkdir(parents=True)
                (run_dir/"trace.jsonl").write_text("{}\n"*count)
                write(run_dir/"trace.summary.json", {"tasks":count})
                job = {"status":"completed_with_response_warnings" if self.warning else "completed",
                    "run_id":run_id,"method":"infa_guard_full","phase":phase,"profile":"strict",
                    "seed":42,"topology":"star","exit_code":0,"completed_tasks":count,
                    "directory":str(run_dir),"command":["--infa-checkpoint",command[command.index("--infa-checkpoint")+1],
                        "--infa-code-dir",str(self.job),"--infa-embedding-model",self.args.minilm_model]}
                write(run_dir/"run.json",job)
                write(root/"matrix.json",{"phase":phase,"profile":"strict","task_count":count,
                    "seeds":[42],"topologies":["star"],"source_sha256":"fixture-source-hash",
                    "dataset":{"source_sha256":digest(self.args.bundle)},"jobs":[job]})
            return 0
        finally:
            with self.lock:
                self.active -= 1
                if label.startswith("pilot"):self.pilots_active -= 1
                if label.startswith("generate"):self.generation_active -= 1

    def pipeline(self):
        return implementation().Pipeline(self.args, runner=self.execute)


class InfaPipelineTests(unittest.TestCase):
    def setUp(self):
        # Other agents edit this checkout concurrently; production launches from a frozen snapshot.
        self.original_source_fingerprint=implementation().source_fingerprint
        patcher=patch.object(implementation(),"source_fingerprint",return_value="fixture-source-hash")
        patcher.start(); self.addCleanup(patcher.stop)
        self.journal_patch=patch("tools.infa_generation_recovery.summarize_journal",
            side_effect=lambda directory,count,*policy:{"accepted_responses":count,"journal_sha256":"fixture-journal-hash",
                                                "accepted_token_budgets":{1024:count}})
        self.journal_summary=self.journal_patch.start();self.addCleanup(self.journal_patch.stop)

    def test_full_order_and_parallel_limit_and_fixed_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory); pipeline = fixture.pipeline(); pipeline.run()
            names = [c[0] for c in fixture.calls]
            self.assertEqual(names[0],"generate-00-protocol")
            self.assertEqual(set(names[1:21]),{f"generate-{i:02d}" for i in range(20)})
            self.assertEqual(names[21:27],["merge","embed","train","validate-checkpoint",
                "smoke-qwen","smoke-gemma31b"])
            self.assertEqual(set(names[27:]),{"pilot-qwen","pilot-gemma31b"})
            self.assertEqual(fixture.maximum,2)
            self.assertEqual({call.args[1] for call in self.journal_summary.call_args_list},{64,1280})
            for label, command, cpu in fixture.calls:
                self.assertTrue(str(command[2]).startswith(str(implementation().ROOT)))
                if label.startswith("generate"):
                    index = int(label.split("-")[1])
                    self.assertEqual(command[command.index("--service")+1],fixture.args.generation_services[index%2])
                if label.startswith(("smoke", "pilot")):
                    self.assertFalse(cpu)
                    self.assertEqual(command[command.index("--infa-source")+1],str(fixture.job))
                    self.assertEqual(command[command.index("--judge-service")+1],"inference2")
            state=json.loads((Path(fixture.args.run_root)/"pipeline-state.json").read_text())
            self.assertEqual(state["status"],"completed")
            self.assertEqual(state["completed_grids"],20)
            self.assertEqual(state["formal_dialogues"],800)
            self.assertEqual(state["child_pids"],{})
            self.assertTrue(state["training_completed"])
            self.assertNotIn("SECRET", json.dumps(state))

    def test_generation_exhaustion_never_trains_or_evaluates(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, workers=1); fixture.failures["generate-00"] = 10
            with self.assertRaisesRegex(RuntimeError,"generate-00"):
                fixture.pipeline().run()
            names = [c[0] for c in fixture.calls]
            self.assertEqual(names.count("generate-00"),3)
            self.assertNotIn("merge",names)
            self.assertFalse(any(n.startswith("smoke") for n in names))
            state=json.loads((Path(fixture.args.run_root)/"pipeline-state.json").read_text())
            self.assertEqual(state["status"],"failed")
            with self.assertRaises(RuntimeError):fixture.pipeline().run()
            self.assertEqual([c[0] for c in fixture.calls].count("generate-00"),3)

    def test_corrupted_completed_marker_is_not_skipped_or_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory); row=fixture.marker("generate-00-protocol")
            Path(row["output"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError,"hash"):
                fixture.pipeline().run()
            self.assertEqual(fixture.calls,[])

    def test_completed_stages_and_evaluations_resume_without_subprocesses(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory); fixture.pipeline().run(); fixture.calls.clear()
            fixture.pipeline().run()
            self.assertEqual(fixture.calls,[])
            trace=next((Path(fixture.args.run_root)/"evaluations").rglob("trace.jsonl"))
            trace.write_text("{}\n")
            with self.assertRaisesRegex(ValueError,"hash|count"):
                fixture.pipeline().run()
            self.assertEqual(fixture.calls,[])

    def test_incomplete_training_log_blocks_checkpoint_validation_and_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory); fixture.bad_epoch=True
            with self.assertRaisesRegex(ValueError,"50|epochs"):
                fixture.pipeline().run()
            names=[c[0] for c in fixture.calls]
            self.assertNotIn("validate-checkpoint",names)
            self.assertNotIn("smoke-qwen",names)

    def test_smoke_exhaustion_blocks_all_pilots_and_uses_fresh_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory); fixture.failures["smoke-gemma31b"]=10
            with self.assertRaisesRegex(RuntimeError,"smoke-gemma31b"):
                fixture.pipeline().run()
            calls=[cmd for name,cmd,_ in fixture.calls if name == "smoke-gemma31b"]
            roots=[cmd[cmd.index("--run-root")+1] for cmd in calls]
            self.assertEqual(len(roots),3);self.assertEqual(len(set(roots)),3)
            self.assertFalse(any(name.startswith("pilot") for name,_,_ in fixture.calls))

    def test_response_warnings_never_pass_smoke_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory); fixture.warning=True
            with self.assertRaises(RuntimeError):fixture.pipeline().run()
            self.assertFalse(any(name.startswith("pilot") for name,_,_ in fixture.calls))

    def test_job_lock_prevents_second_controller_with_different_root(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory)
            with (fixture.results/"pipeline.lock").open("w") as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError,"lock|controller"):
                    fixture.pipeline().run()
            self.assertEqual(fixture.calls,[])

    def test_changed_inputs_do_not_erase_saved_retry_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory,workers=1);fixture.failures["generate-00"]=10
            with self.assertRaises(RuntimeError):fixture.pipeline().run()
            original=Path(fixture.args.bundle).read_bytes()
            write(Path(fixture.args.bundle),{"tasks":list(range(201))})
            with self.assertRaisesRegex(ValueError,"inputs"):
                fixture.pipeline().run()
            state=json.loads((Path(fixture.args.run_root)/"pipeline-state.json").read_text())
            self.assertEqual(state["attempts"].get("generate-00"),3)
            Path(fixture.args.bundle).write_bytes(original)
            with self.assertRaisesRegex(RuntimeError,"generate-00"):
                fixture.pipeline().run()
            self.assertEqual([c[0] for c in fixture.calls].count("generate-00"),3)

    def test_training_log_hash_is_rechecked_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory);fixture.pipeline().run();fixture.calls.clear()
            path=fixture.results/"train.log"
            path.write_text(path.read_text().replace("0.1,", "0.9,"))
            with self.assertRaisesRegex(ValueError,"hash"):
                fixture.pipeline().run()
            self.assertEqual(fixture.calls,[])

    def test_single_generation_service_is_never_used_concurrently(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory);fixture.args.generation_services=["inference2"]
            fixture.pipeline().run()
            self.assertEqual(fixture.generation_maximum,1)

    def test_nonzero_evaluation_exit_cannot_resume_as_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory)
            def failed_exit(label,command,log,cpu=True):
                code=fixture.execute(label,command,log,cpu)
                return 1 if label.startswith("smoke") else code
            pipeline=implementation().Pipeline(fixture.args,runner=failed_exit)
            with self.assertRaisesRegex(RuntimeError,"smoke-qwen"):pipeline.run()
            with self.assertRaisesRegex(RuntimeError,"smoke-qwen"):
                implementation().Pipeline(fixture.args,runner=failed_exit).run()
            names=[c[0] for c in fixture.calls]
            self.assertEqual(names.count("smoke-qwen"),3)
            self.assertNotIn("smoke-gemma31b",names)

    def test_both_pilots_run_in_parallel_after_both_smokes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory);fixture.pipeline().run()
            self.assertEqual(fixture.pilots_maximum,2)
            names=[c[0] for c in fixture.calls]
            self.assertLess(max(names.index("smoke-qwen"),names.index("smoke-gemma31b")),
                            min(names.index("pilot-qwen"),names.index("pilot-gemma31b")))

    def test_checkpoint_report_is_separate_from_warning_log(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory);fixture.pipeline().run()
            command=next(c for n,c,_ in fixture.calls if n == "validate-checkpoint")
            self.assertIn("--output",command)
            self.assertTrue(read_report := json.loads((Path(fixture.args.run_root)/"checkpoint-validation.json").read_text()))
            self.assertTrue(read_report["strict_load"])

    def test_protocol_reply_journal_hash_is_verified_before_formal_generation(self):
        self.journal_patch.stop()
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory,real_journal=True);marker=fixture.marker("generate-00-protocol")
            path=Path(marker["recovery_journal"])/"000000.json"
            row=json.loads(path.read_text());row["response"]["choices"][0]["message"]["content"]="tampered"
            write(path,row)
            with self.assertRaisesRegex(ValueError,"hash"):fixture.pipeline().run()
            self.assertEqual(fixture.calls,[])

    def test_protocol_reply_journal_must_have_all_64_accepted_responses(self):
        self.journal_patch.stop()
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory,real_journal=True);marker=fixture.marker("generate-00-protocol")
            (Path(marker["recovery_journal"])/"000063.json").unlink()
            with self.assertRaisesRegex(ValueError,"count|64"):fixture.pipeline().run()
            self.assertEqual(fixture.calls,[])

    def test_changed_native_source_is_rejected_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory)
            (fixture.job/"native.py").write_text("# changed source\n")
            with self.assertRaisesRegex(ValueError,"hash"):
                fixture.pipeline()._preflight()

    def test_completed_evaluation_from_other_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory);pipeline=fixture.pipeline();pipeline._preflight()
            checkpoint=Path(next(iter(fixture.marker("train")["checkpoints"])))
            root=Path(fixture.args.run_root)/"evaluation-fixture"
            command=pipeline._evaluation_command("smoke","qwen",root,checkpoint)
            fixture.execute("smoke-qwen",command,Path(fixture.args.run_root)/"unused.log",False)
            matrix=json.loads((root/"matrix.json").read_text());matrix["source_sha256"]="other-source"
            write(root/"matrix.json",matrix)
            with self.assertRaisesRegex(ValueError,"source"):
                pipeline._validate_evaluation(root,"smoke","qwen",checkpoint)

    def test_source_fingerprint_matches_matrix_runner(self):
        from tools.run_appworld_matrix import source_fingerprint
        self.assertEqual(self.original_source_fingerprint(),source_fingerprint())

    def test_real_subprocess_cpu_environment_and_pid_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory); pipeline=implementation().Pipeline(fixture.args)
            log=Path(fixture.args.run_root)/"cpu.log"
            code=pipeline._run_command("cpu-probe",[sys.executable,"-c",
                "import os;print(os.environ['OMP_NUM_THREADS'],os.environ['MKL_NUM_THREADS'],repr(os.environ['CUDA_VISIBLE_DEVICES']),os.environ['PYTHONPATH'])"],log,True)
            self.assertEqual(code,0)
            self.assertIn("2 2 '' /mnt/public/data/wj/baseline-training-runtime",log.read_text())
            self.assertEqual(pipeline.state["child_pids"],{})

if __name__ == "__main__":unittest.main()
