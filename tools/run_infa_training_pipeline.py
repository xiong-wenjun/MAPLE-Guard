"""Resume a prepared, pinned INFA job through training and gated AppWorld tests.

Launch this script from the frozen MAPLE snapshot. Its ROOT is the sole source
for every child command. Generation retries reuse the release-stage response
journal; merge, embedding and training are never restarted automatically.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = "/mnt/public/data/wj/venvs/maple-experiments/bin/python"
TRAINING_RUNTIME = "/mnt/public/data/wj/baseline-training-runtime"
MAX_ATTEMPTS = 3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def require_hash(path, expected):
    if not Path(path).is_file() or sha256(path) != expected:
        raise ValueError("Artifact hash mismatch: " + str(path))


def source_fingerprint(root=None):
    root = ROOT if root is None else Path(root)
    digest = hashlib.sha256()
    for folder in ("maple_guard", "evaluate", "configs", "prompts", "tools"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".json", ".sh"):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


class Pipeline:
    def __init__(self, args, runner=None):
        self.args = args
        self.recipe_path = Path(args.recipe).resolve()
        self.recipe = read_json(self.recipe_path)
        self.job = Path(self.recipe["working_directory"]).resolve()
        self.results = self.job / "stage-results"
        self.run_root = Path(args.run_root).resolve()
        self.results.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_root / "pipeline-state.json"
        self.snapshot = self.run_root / "recipe.snapshot.json"
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.processes = {}
        self.lock_handle = None
        self.validated_grids = set()
        self.runner = runner or self._run_command
        self.state = {"schema_version":1, "status":"prepared", "stage":"preflight", "child_pids":{},
            "completed_grids":0, "formal_dialogues":0, "training_completed":False,
            "stages":{}, "attempts":{}, "evaluations":{}}
        services = read_json(args.services)
        self.secrets = [str(value["api_key"]) for value in services.values()
                        if isinstance(value,dict) and value.get("api_key")]

    def _redact(self, value):
        value = str(value)
        for secret in self.secrets:
            value = value.replace(secret, "[REDACTED]")
        return value

    def _save(self):
        with self.mutex:
            self.state["updated_at"] = time.time()
            atomic_json(self.state_path, self.state)

    def _update(self, **values):
        with self.mutex:
            self.state.update(values)
            self._save()

    def _inputs(self):
        return {**({"resume_from":str(Path(self.args.resume_from).resolve())}
                  if getattr(self.args,"resume_from",None) else {}),"source_root":str(ROOT), "source_sha256":source_fingerprint(),
            "recipe_path":str(self.recipe_path), "recipe_sha256":sha256(self.recipe_path),
            "heldout":{str(Path(p).resolve()):sha256(p) for p in self.args.heldout},
            "bundle":str(Path(self.args.bundle).resolve()), "bundle_sha256":sha256(self.args.bundle),
            "services_path":str(Path(self.args.services).resolve()), "services_sha256":sha256(self.args.services),
            "generation_services":list(self.args.generation_services),
            "minilm_model":str(Path(self.args.minilm_model).resolve()), "workers":self.args.workers}

    def _migrate_source(self, inputs):
        """Audit a fresh controller against a stopped predecessor; never edit its history."""
        previous_root = Path(self.args.resume_from).resolve()
        if previous_root == self.run_root:
            raise ValueError("Source migration requires a fresh run root")
        previous_path = previous_root / "pipeline-state.json"
        previous = read_json(previous_path)
        if previous.get("status") not in ("failed", "interrupted") or previous.get("child_pids"):
            raise ValueError("Source migration requires a stopped predecessor without children")
        if previous.get("training_completed") or previous.get("evaluations"):
            raise ValueError("Generation recovery cannot migrate completed training or evaluations")
        old_inputs = previous["inputs"]
        unchanged = lambda value: {key:item for key,item in value.items()
                                   if key not in ("source_root", "source_sha256", "resume_from")}
        if unchanged(old_inputs) != unchanged(inputs):
            raise ValueError("Source migration must preserve recipe, services, data and generation settings")
        if source_fingerprint(old_inputs["source_root"]) != old_inputs["source_sha256"]:
            raise ValueError("Previous frozen source hash changed")
        require_hash(previous_root / "recipe.snapshot.json", old_inputs["recipe_sha256"])
        for label, known in previous.get("stages", {}).items():
            if not label.startswith("generate-"):
                raise ValueError("Only generation-stage recovery is supported")
            require_hash(self.results / (label + ".json"), known["marker_sha256"])
            if self._validate_marker(label) is None:
                raise ValueError("Previously verified generation marker missing")
        self.state["stages"] = dict(previous.get("stages", {}))
        report = {"schema_version":1, "previous_run_root":str(previous_root),
            "previous_state_sha256":sha256(previous_path), "previous_inputs":old_inputs,
            "new_inputs":inputs, "preserved_stages":self.state["stages"],
            "previous_attempts":previous.get("attempts", {}),
            "attempt_policy":"new bounded attempts after verified wrapper-source migration; old history retained",
            "recipe_policy":"byte-identical recipe and native source; only wrapper source may change"}
        atomic_json(self.run_root / "source-transition.json", report)
        self.state["source_transition_sha256"] = sha256(self.run_root / "source-transition.json")

    def _preflight(self):
        # Preserve audit/retry history even when a new invocation fails preflight.
        if self.state_path.exists():
            self.state = read_json(self.state_path)
        if not 1 <= self.args.workers <= 2:
            raise ValueError("Generation workers must be 1 or 2")
        if (not self.args.generation_services or len(self.args.generation_services) > 2 or
                len(set(self.args.generation_services)) != len(self.args.generation_services) or
                set(self.args.generation_services) - {"inference1", "inference2"}):
            raise ValueError("Choose distinct generation services inference1 and/or inference2")
        if len(self.args.heldout) != 5 or len({str(Path(p).resolve()) for p in self.args.heldout}) != 5:
            raise ValueError("Five distinct heldout paths are required")
        generation = self.recipe["generation"]
        train = self.recipe["training"]["argv"]
        if (len(generation) != 20 or self.recipe["expected_dialogues"] != 800 or
                any(g["argv"][g["argv"].index("--samples") + 1] != "40" for g in generation) or
                train[train.index("--epochs") + 1] != "50"):
            raise ValueError("Require the pinned 20 x 40 generation grid and 50 epochs")
        if self.recipe.get("generation_profile") not in ("qwen_no_thinking_recover", "qwen_no_thinking_released_budget"):
            raise ValueError("Controller requires the declared resumable generation profile")
        if self.recipe["generation_profile"]=="qwen_no_thinking_released_budget":
            policy=self.recipe.get("generation_recovery",{})
            if policy.get("token_budgets")!=[1024] or policy.get("accepted_finish_reasons")!=["stop","length"]:
                raise ValueError("Released-budget profile requires fixed 1024 stop/length policy")
        if read_json(self.job / "maple-reproduction.json")["recipe"] != self.recipe:
            raise ValueError("Recipe differs from prepared INFA source")
        source_files = self.recipe.get("source_files", {})
        if not source_files:
            raise ValueError("Prepared recipe has no pinned source hashes")
        for name, expected in source_files.items():
            require_hash(self.job / name, expected)
        if len(read_json(self.args.bundle)["tasks"]) < 200:
            raise ValueError("AppWorld pilot requires 200 frozen tasks")
        inputs = self._inputs()
        if self.state_path.exists():
            previous = read_json(self.state_path)
            if previous.get("inputs") != inputs:
                raise ValueError("Pipeline inputs or frozen source changed; refusing resume")
            # A previous hard interruption must never launch over a live child.
            for pid in previous.get("child_pids", {}).values():
                try:
                    os.killpg(int(pid), 0)
                except ProcessLookupError:
                    continue
                raise RuntimeError("Previous controller child is still alive; inspect it before resume")
            self.state = previous
            self.state["child_pids"] = {}
            require_hash(self.snapshot, inputs["recipe_sha256"])
            if self.state.get("source_transition_sha256"):
                require_hash(self.run_root / "source-transition.json", self.state["source_transition_sha256"])
        else:
            if getattr(self.args,"resume_from",None):
                self._migrate_source(inputs)
            self.snapshot.write_bytes(self.recipe_path.read_bytes())
            self.state["inputs"] = inputs
        self._update(status="running", stage="protocol", error=None, controller_pid=os.getpid(),
                     completed_grids=0, formal_dialogues=0, training_completed=False, finished_at=None)

    def _run_command(self, label, command, log_path, cpu=True):
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        if cpu:
            env.update(PYTHONPATH=TRAINING_RUNTIME, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with log_path.open("w") as log:
            with self.mutex:
                if self.stop.is_set():
                    raise RuntimeError("Pipeline interrupted before " + label)
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1, start_new_session=True,
                    pass_fds=(self.lock_handle.fileno(),) if self.lock_handle else ())
                self.processes[label] = proc
                self.state["child_pids"][label] = proc.pid
                self._save()
            try:
                for line in proc.stdout:
                    log.write(self._redact(line)); log.flush()
                return proc.wait()
            finally:
                if proc.poll() is None:
                    try: os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                        proc.wait()
                proc.stdout.close()
                with self.mutex:
                    self.processes.pop(label, None)
                    self.state["child_pids"].pop(label, None)
                    self._save()

    def _terminate_children(self):
        self.stop.set()
        with self.mutex:
            processes = list(self.processes.values())
        for proc in processes:
            if proc.poll() is None:
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for proc in processes:
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()

    def _validate_marker(self, label):
        marker = self.results / (label + ".json")
        if not marker.exists():
            return None
        row = read_json(marker)
        stage = "generate" if label.startswith("generate") else label
        if row.get("status") != "completed" or row.get("stage") != stage:
            raise ValueError("Invalid completed stage marker: " + label)
        for key in ("source_revision", "generation_profile", "generator_model"):
            if row.get(key) != self.recipe[key]:
                raise ValueError("Stage marker provenance differs: " + label)
        formal = Path(self.recipe["merge"]["argv"][-1]) / "train"
        if stage == "train":
            checkpoints = row.get("checkpoints", {})
            if len(checkpoints) != 1:
                raise ValueError("Require a single upstream best checkpoint")
            for path, expected in checkpoints.items():
                if not Path(path).resolve().is_relative_to(self.job / "checkpoints"):
                    raise ValueError("Checkpoint escaped the prepared job")
                require_hash(path, expected)
        else:
            output = Path(row["output"]).resolve()
            if stage == "generate":
                index = int(label.split("-")[1]); grid = self.recipe["generation"][index]
                protocol = label.endswith("protocol")
                folder = self.job / "protocol-check/PI/csqa/train" if protocol else formal
                suffix = f"-num_attackers_{grid['attackers']}-sparsity_{grid['sparsity']}.json"
                if (output.parent != folder.resolve() or not output.name.endswith(suffix) or
                        row.get("protocol_check") is not protocol or row.get("seed") != grid["seed"]):
                    raise ValueError("Stage marker grid identity differs: " + label)
                count = 2 if protocol else 40
                if row.get("validation", {}).get("dialogues") != count:
                    raise ValueError("Unexpected validated dialogue count: " + label)
                from tools.infa_generation_recovery import summarize_journal
                journal = self.results / (label + "-responses")
                if Path(row.get("recovery_journal", "")).absolute() != journal.absolute():
                    raise ValueError("Generation journal path differs: " + label)
                integrity = summarize_journal(journal, count * 8 * 4, self.recipe.get("generation_recovery"))
                declared = row.get("generation_integrity", {})
                if any(declared.get(key) != integrity[key] for key in ("accepted_responses", "journal_sha256")):
                    raise ValueError("Generation journal hash/count differs: " + label)
                if self.recipe.get("generation_profile")=="qwen_no_thinking_released_budget":
                    for key in ("accepted_finish_reasons", "import_provenance_sha256"):
                        if declared.get(key)!=integrity.get(key):
                            raise ValueError("Generation response/import provenance differs: " + label)
            elif stage == "merge":
                if output != (formal / "dataset.json").resolve():
                    raise ValueError("Unexpected merged dataset output")
                count = 800
            else:
                train = self.recipe["training"]["argv"]
                if output != Path(train[train.index("--dataset_path") + 1]).resolve():
                    raise ValueError("Unexpected embedding output")
            require_hash(output, row["sha256"])
            if stage in ("generate", "merge") and len(read_json(output)) != count:
                raise ValueError("Output dialogue count differs: " + label)
        known = self.state["stages"].get(label)
        if known and known["marker_sha256"] != sha256(marker):
            raise ValueError("Stage marker hash changed: " + label)
        return row

    def _record_stage(self, label):
        with self.mutex:
            self.state["stages"][label] = {"status":"completed", "marker_sha256":sha256(self.results/(label+".json"))}
            if re.fullmatch(r"generate-\d{2}", label): self.validated_grids.add(label)
            self.state["completed_grids"] = len(self.validated_grids)
            self.state["formal_dialogues"] = self.state["completed_grids"] * 40
            self._save()

    def _stage(self, label, stage, index=0, protocol=False):
        existing = self._validate_marker(label)
        if existing is not None:
            self._record_stage(label)
            return existing
        command = [PYTHON, "-B", str(ROOT/"tools/run_infa_release_stage.py"),
            "--recipe", str(self.snapshot), "--heldout", *self.args.heldout, "--stage", stage]
        limit = MAX_ATTEMPTS if stage == "generate" else 1
        if stage == "generate":
            command += ["--grid-index", str(index), "--services", self.args.services,
                        "--service", self.args.generation_services[index % len(self.args.generation_services)]]
            if protocol: command += ["--protocol-check"]
        while self.state["attempts"].get(label, 0) < limit:
            if self.stop.is_set(): raise RuntimeError("Pipeline stopped before " + label)
            with self.mutex:
                attempt = self.state["attempts"].get(label, 0) + 1
                self.state["attempts"][label] = attempt
                self._save()
            log = self.run_root / "logs" / f"{label}-attempt-{attempt:02d}.log"
            code = self.runner(label, command, log, True)
            # A corrupt marker is never treated as a recoverable process failure.
            completed = self._validate_marker(label)
            if code == 0 and completed is not None:
                self._record_stage(label)
                return completed
            if completed is not None:
                raise RuntimeError(label + " exited nonzero despite a completion marker")
        raise RuntimeError(label + f" failed after {limit} process attempts; downstream blocked")

    def _generate(self):
        self._update(stage="generation")
        # Service lanes keep at most one request-producing process per service.
        lanes = min(self.args.workers, len(self.args.generation_services))
        def lane(number):
            try:
                for index in range(number, 20, lanes):
                    if self.stop.is_set(): return
                    self._stage(f"generate-{index:02d}", "generate", index)
            except BaseException:
                self._terminate_children()
                raise
        with ThreadPoolExecutor(max_workers=lanes) as pool:
            futures = [pool.submit(lane, number) for number in range(lanes)]
            for future in futures: future.result()
        if self.stop.is_set(): raise RuntimeError("Generation stopped; downstream blocked")
        for index in range(20): self._validate_marker(f"generate-{index:02d}")

    def _training_checkpoint(self, marker):
        if self.state.get("training_log_sha256"):
            require_hash(self.results / "train.log", self.state["training_log_sha256"])
        log = (self.results / "train.log").read_text()
        epochs = re.findall(r"^Epoch (\d+)/50 \|\| Train Loss: ([^\r\n]+)", log, re.MULTILINE)
        if [int(i) for i, _ in epochs] != list(range(50)) or "Training parameters: epochs=50," not in log:
            raise ValueError("Training log does not prove exactly 50 completed epochs")
        if any(re.search(r"\b(?:nan|inf)\b", values, re.I) for _, values in epochs):
            # Labels use 'inf' for infection, so inspect only numeric nonfinite tokens.
            if re.search(r"(?:[:/ ]|^)(?:nan|[-+]?inf)(?=[/% ,]|$)", "\n".join(v for _,v in epochs), re.I):
                raise ValueError("Training log contains nonfinite metrics")
        if " || Save!" not in log or "Training completed. Best test accuracy:" not in log:
            raise ValueError("Upstream training did not complete and select a best checkpoint")
        checkpoint = Path(next(iter(marker["checkpoints"]))).resolve()
        actual = {p.resolve() for p in (self.job/"checkpoints").rglob("*.pth")}
        if actual != {checkpoint}:
            raise ValueError("Require a single upstream best checkpoint on disk")
        selected = checkpoint.parent / "latest_model_path.txt"
        if (not selected.is_file() or Path(selected.read_text().strip()).resolve() != checkpoint or
                "Model saved to: " + str(checkpoint) not in log):
            raise ValueError("Upstream best checkpoint selection differs")
        self._update(training_completed=True, checkpoint=str(checkpoint),
                     checkpoint_sha256=sha256(checkpoint), training_log_sha256=sha256(self.results/"train.log"))
        return checkpoint

    def _checkpoint_gate(self, checkpoint):
        self._update(stage="validate-checkpoint")
        output = self.run_root / "checkpoint-validation.json"
        previous = self.state.get("checkpoint_validation")
        if previous:
            require_hash(output, previous["sha256"])
        elif not output.exists():
            command = [PYTHON, "-B", str(ROOT/"tools/official_reproduction.py"), "validate-infa-checkpoint",
                       "--source", str(self.job), "--checkpoint", str(checkpoint), "--output", str(output)]
            log = self.run_root / "logs/checkpoint-validation.log"
            if self.runner("validate-checkpoint", command, log, True) != 0:
                raise RuntimeError("Strict checkpoint load/forward failed; evaluations blocked")
        report = read_json(output)
        if (report.get("strict_load") is not True or report.get("forward_shape") != [2,2] or
                report.get("checkpoint_sha256") != sha256(checkpoint)):
            raise ValueError("Checkpoint validation does not prove strict load and finite native forward")
        self._update(checkpoint_validation={"status":"completed","sha256":sha256(output)})

    def _evaluation_command(self, phase, backbone, root, checkpoint):
        return [PYTHON, "-B", str(ROOT/"tools/run_appworld_matrix.py"), "--bundle", self.args.bundle,
            "--services", self.args.services, "--task-service", "inference2" if backbone == "qwen" else "inference3",
            "--judge-service", "inference2", "--backbone", backbone, "--profile", "strict",
            "--phase", phase, "--tasks", "2" if phase == "smoke" else "200", "--workers", "1",
            "--methods", "infa_guard_full", "--seeds", "42", "--topologies", "star",
            "--infa-checkpoint", str(checkpoint), "--infa-source", str(self.job),
            "--minilm-model", self.args.minilm_model, "--run-root", str(root), "--execute"]

    def _validate_evaluation(self, root, phase, backbone, checkpoint):
        matrix_path = root / "matrix.json"
        if not matrix_path.is_file(): return None
        matrix = read_json(matrix_path)
        if matrix.get("source_sha256") != self.state["inputs"]["source_sha256"]:
            raise ValueError("Evaluation source hash differs from the frozen controller snapshot")
        count = 2 if phase == "smoke" else 200
        if (matrix.get("phase") != phase or matrix.get("profile") != "strict" or
                matrix.get("task_count") != count or matrix.get("seeds") != [42] or
                matrix.get("topologies") != ["star"] or
                matrix.get("dataset", {}).get("source_sha256") != sha256(self.args.bundle) or
                len(matrix.get("jobs", [])) != 1):
            raise ValueError("Evaluation manifest configuration differs")
        job = matrix["jobs"][0]
        if job.get("status") != "completed": return None
        run_id = f"{phase}_strict_{backbone}_infa_guard_full_star_s42"
        directory = root / run_id
        if (job.get("run_id") != run_id or Path(job["directory"]).resolve() != directory.resolve() or
                job.get("method") != "infa_guard_full" or job.get("seed") != 42 or
                job.get("topology") != "star" or job.get("exit_code") != 0 or job.get("completed_tasks") != count):
            raise ValueError("Completed evaluation identity/count differs")
        command = job.get("command", [])
        for flag, value in (("--infa-checkpoint",str(checkpoint)), ("--infa-code-dir",str(self.job)),
                            ("--infa-embedding-model",self.args.minilm_model)):
            if flag not in command or command[command.index(flag)+1] != value:
                raise ValueError("Evaluation assets differ: " + flag)
        run_path, trace, summary = (directory/name for name in ("run.json", "trace.jsonl", "trace.summary.json"))
        if read_json(run_path) != job or not summary.is_file() or not trace.is_file():
            raise ValueError("Completed evaluation artifacts missing or inconsistent")
        with trace.open() as handle:
            if sum(bool(line.strip()) for line in handle) != count:
                raise ValueError("Completed evaluation trace count differs")
        read_json(summary)
        usage = job.get("api_usage", {})
        if any(usage.get(key,0) for key in ("truncated_chat_requests", "chat_requests_without_final_content", "http_errors")):
            return None
        return {str(path):sha256(path) for path in (matrix_path,run_path,trace,summary)}

    def _evaluation_state(self, label, value):
        with self.mutex:
            self.state["evaluations"][label] = value
            self._save()

    def _evaluate(self, phase, backbone, checkpoint):
        label = phase + "-" + backbone
        self._update(stage=label)
        previous = self.state["evaluations"].get(label)
        if previous and previous.get("status") == "completed":
            for path, digest in previous["artifacts"].items(): require_hash(path, digest)
            if not self._validate_evaluation(Path(previous["run_root"]), phase, backbone, checkpoint):
                raise ValueError("Completed evaluation no longer passes its gate")
            return
        for attempt in range(1, MAX_ATTEMPTS+1):
            root = self.run_root / "evaluations" / label / f"attempt-{attempt:02d}"
            # Root allocation is durable even if the runner fails before creating it.
            ticket = root.with_suffix(".json")
            if ticket.exists() or root.exists():
                if ticket.exists() and read_json(ticket).get("status") == "failed":
                    continue
                artifacts = self._validate_evaluation(root,phase,backbone,checkpoint)
                if artifacts:
                    self._evaluation_state(label, {"status":"completed","attempt":attempt,
                        "run_root":str(root),"artifacts":artifacts})
                    return
                continue
            root.parent.mkdir(parents=True,exist_ok=True)
            atomic_json(ticket,{"status":"allocated","attempt":attempt})
            self._evaluation_state(label, {"status":"running","attempt":attempt,"run_root":str(root)})
            code = self.runner(label,self._evaluation_command(phase,backbone,root,checkpoint),
                               self.run_root/"logs"/f"{label}-attempt-{attempt:02d}.log",False)
            artifacts = self._validate_evaluation(root,phase,backbone,checkpoint)
            status = "completed" if code == 0 and artifacts else "failed"
            self._evaluation_state(label, {"status":status,"attempt":attempt,"run_root":str(root),
                                           "exit_code":code,"artifacts":artifacts})
            atomic_json(ticket,{"status":status,"exit_code":code,"attempt":attempt})
            self._save()
            if status == "completed": return
        raise RuntimeError(label + " failed after 3 fresh attempts; downstream blocked")

    def run(self):
        with (self.results/"pipeline.lock").open("a+") as handle:
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another controller holds this job's pipeline lock") from exc
            self.lock_handle = handle
            old_handlers = {}
            if threading.current_thread() is threading.main_thread():
                def interrupted(signum, _frame):
                    self._terminate_children()
                    raise KeyboardInterrupt(f"Controller received signal {signum}")
                for signum in (signal.SIGTERM, signal.SIGINT):
                    old_handlers[signum] = signal.signal(signum, interrupted)
            try:
                self._preflight()
                self._stage("generate-00-protocol","generate",0,True)
                self._generate()
                for stage in ("merge","embed","train"):
                    self._update(stage=stage)
                    marker = self._stage(stage,stage)
                checkpoint = self._training_checkpoint(marker)
                self._checkpoint_gate(checkpoint)
                for backbone in ("qwen","gemma31b"):
                    self._evaluate("smoke",backbone,checkpoint)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    pilots = [pool.submit(self._evaluate,"pilot",backbone,checkpoint)
                              for backbone in ("qwen","gemma31b")]
                    for future in pilots: future.result()
                self._update(status="completed", stage="completed", finished_at=time.time())
                return self.state
            except BaseException as exc:
                self._terminate_children()
                for value in self.state["evaluations"].values():
                    if value.get("status") == "running": value["status"] = "failed"
                self._update(status="failed", error={"type":type(exc).__name__,"message":self._redact(exc)},
                             finished_at=time.time())
                raise
            finally:
                for signum, handler in old_handlers.items(): signal.signal(signum,handler)
                self.lock_handle = None


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe",required=True)
    parser.add_argument("--heldout",nargs=5,required=True)
    parser.add_argument("--services",required=True)
    parser.add_argument("--generation-services",nargs="+",required=True)
    parser.add_argument("--bundle",required=True)
    parser.add_argument("--minilm-model",required=True)
    parser.add_argument("--run-root",required=True)
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    parser.add_argument("--resume-from",help="Stopped generation controller to verify when changing frozen wrapper source")
    args = parser.parse_args(argv)
    for field in ("recipe","services","bundle","minilm_model","run_root"):
        setattr(args,field,str(Path(getattr(args,field)).resolve()))
    args.heldout = [str(Path(path).resolve()) for path in args.heldout]
    return args


def main(argv=None):
    os.umask(0o077)
    pipeline = Pipeline(arguments(argv))
    try: pipeline.run()
    except (Exception,KeyboardInterrupt) as exc:
        print(json.dumps({"status":"failed","type":type(exc).__name__,"state":str(pipeline.state_path)}),flush=True)
        return 1
    print(json.dumps({"status":"completed","state":str(pipeline.state_path)}),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
