"""Pinned release assets and recipes; no inferred paper parameters."""
from __future__ import annotations
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tempfile

LOCK_PATH = Path(__file__).resolve().parents[2] / "configs/baselines/official_reproduction.lock.json"

def load_lock():
    return json.loads(LOCK_PATH.read_text())

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_files(root, expected):
    root = Path(root).resolve()
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Required release file missing: {relative}")
        if sha256(path) != digest:
            raise ValueError(f"Release hash mismatch: {relative}")
    return dict(expected)

def verify_source(method, root):
    entry = load_lock()["methods"][method]
    return {"source_revision":entry["revision"],
            "source_sha256":verify_files(root, entry["files"])}

def source_function(path, name, namespace):
    """Execute only an audited function definition, never a module's demo calls."""
    tree = ast.parse(Path(path).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError(f"Expected one undecorated function: {name}")
    module = ast.Module(body=functions, type_ignores=[])
    scope = dict(namespace)
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]

def agentsafe_relations(source, names, seed):
    provenance = verify_source("agentsafe", source)
    if len(names) < 2 or len(set(names)) != len(names) or any(
            not name or re.search(r"[(),:\n\r]", name) for name in names):
        raise ValueError("Unique names without relation-file delimiters are required")
    function = source_function(Path(source)/"Code/initial.py", "initialize_relations",
                               {"os":os, "random":random.Random(seed)})
    # The released function writes relative to cwd; call only in this dedicated process.
    with tempfile.TemporaryDirectory() as folder:
        before = Path.cwd()
        try:
            os.chdir(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                function([{"name":name} for name in names])
            text = Path("relations.txt").read_text()
        finally:
            os.chdir(before)
    levels = {"Stranger":1, "Colleague":2, "Friend":3, "Family":4}
    ids = {name:str(i) for i,name in enumerate(names)}
    relations = {str(i):{} for i in range(len(names))}
    for line in text.splitlines():
        match = re.fullmatch(r"\((.+), (.+)\): (Family|Friend|Colleague|Stranger)", line)
        if not match:
            raise ValueError("Unrecognized released relation record")
        source_name, target_name, relation = match.groups()
        relations[ids[source_name]][ids[target_name]] = levels[relation]
    policy = {"identities":{str(i):name for i,name in enumerate(names)},
              "relations":relations, "default_level":1, "self_level":4,
              "provenance":{**provenance, "seed":seed, "seed_origin":"preregistered reproduction seed; upstream unspecified",
                            "profile":"released_relationship_generator_maple_id_mapping",
                            "self_level_origin":"MAPLE host convention",
                            "full_paper_configuration_recovered":False}}
    return text, policy

def guardian_parser(code_dir):
    return source_function(Path(code_dir)/"utils.py", "parse_single_choice", {"re":re})

def verify_bert(directory):
    directory = Path(directory).resolve()
    if directory.name != "bert-base-uncased":
        raise ValueError("Released GUARDIAN requires the local directory basename bert-base-uncased")
    files = verify_files(directory, load_lock()["assets"]["bert_base_uncased"]["files"])
    return {"bert_revision":load_lock()["assets"]["bert_base_uncased"]["revision"],
            "bert_sha256":files}

def infa_recipe(source, job, dataset, model, seed):
    source, job, dataset = map(lambda p:str(Path(p).resolve()), (source, job, dataset))
    name = "native_infa_s" + str(seed)
    graph = str(Path(job)/f"output/output_{name}/agent_graph_dataset_{name}")
    feature = str(Path(job)/f"output/output_{name}/ModelTrainingSet_{name}/csqa/dataset.pkl")
    jobs = []
    for attackers in (1,2,3,4):
        for density in (.2,.4,.6,.8,1.):
            jobs.append({"seed":seed+len(jobs), "attackers":attackers, "sparsity":density,
                "argv":["generate_data/gen_graph.py", "--model_type",model,"--attack_mode","PI",
                        "--phase","train","--num_nodes","8","--sparsity",str(density),
                        "--num_attackers",str(attackers),"--save_dir",graph,"--dataset","csqa",
                        "--dataset_path",dataset,"--num_dialogue_turns","3",
                        "--num_graphs","20","--samples","40"]})
    return {"schema_version":1,"source":source, "working_directory":job,
        "profile":"official_release_training_recipe",
        "source_revision":load_lock()["methods"]["infa"]["revision"],
        "source_files":load_lock()["methods"]["infa"]["files"],
        "generator_model":model,"original_generator_model":"gpt-4o-mini",
        "model_substitution":model != "gpt-4o-mini",
        "original_random_seed_known":False,"reproduction_seed":seed,
        "seed_policy":"Python, NumPy and Torch: seed + generation grid index; training: seed",
        "expected_dialogues":800,"observed_turns":4,"generation":jobs,
        "merge":{"argv":["generate_data/dataset_utils/merge_datasets.py","--phase","train",
                         "--attack_mode","PI","--dataset","csqa","--root",graph+"/PI/csqa"],
                 "must_be_fresh_directory":True,
                 "note":"Released os.listdir order must be recorded; never merge again with dataset.json present."},
        "embedding":{"argv":["generate_data/gen_training_dataset.py","--attack_mode","PI",
                             "--dataset","csqa","--name",name],
                     "model":"sentence-transformers/all-MiniLM-L6-v2","dimension":384},
        "training":{"seed":seed,"argv":["train/trainer.py","--epochs","50","--batch_size","32",
                    "--lr","0.001","--save_dir",str(Path(job)/"checkpoints"),"--attack_mode","PI",
                    "--dataset","csqa","--dataset_path",feature,"--name",name,
                    "--selective_training","--guard","ours"],
                    "effective_parameters":load_lock()["methods"]["infa"]["effective_training"]},
        "required_preconditions":["Separate official CSQA train parquet; exclude all five held-out benchmark questions",
            "Pinned source copied into a new working_directory; no edits to audited reference source",
            "Pinned local MiniLM linked at train/models/sentence-transformers/all-MiniLM-L6-v2",
            "OPENAI_API_KEY and BASE_URL injected into process environment; never saved in recipe",
            "Complete four-turn, eight-node traces with per-turn infection labels; no API errors",
            "Record merge input order before released merge; audit 640/160 train/validation question overlap",
            "Preserve training logs and selected checkpoint; native strict load and real forward pass"],
        "author_checkpoint":False,"training_completed":False,
        "claim_after_training":"independently retrained official architecture/recipe, with listed substitutions"}

def normalize_question(value):
    return " ".join(str(value).casefold().split())

def heldout_questions(paths):
    result = set()
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).casefold().replace(" ", "_") in {"question", "stem", "query", "instruction", "user_instruction"} and isinstance(item,str):
                    result.add(normalize_question(item))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    for path in paths:
        visit(json.loads(Path(path).read_text()))
    if not result:
        raise ValueError("No held-out question text found; supply raw benchmark files")
    return result

def validate_infa_dialogues(rows, heldout, expected_count=None):
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} dialogues, found {len(rows)}")
    heldout = {normalize_question(q) for q in heldout}
    question_counts = {}
    for i, row in enumerate(rows):
        q = normalize_question(row.get("question", ""))
        # Compare both full prompt and question stem before appended choices.
        stem = normalize_question(re.split(r"\nA[.)]\s", str(row.get("question","")), maxsplit=1)[0])
        if q in heldout or stem in heldout:
            raise ValueError(f"Dialogue {i} overlaps held-out test question")
        turns = row.get("communication_data", [])
        labels = row.get("infected_idxes_per_turn")
        if not isinstance(labels,list) or len(labels) != 4 or len(turns) != 4:
            raise ValueError(f"Dialogue {i} must have four genuine per-turn infection labels and replies")
        adjacency = row.get("adj_matrix", [])
        if not q or len(adjacency) != 8 or any(len(r) != 8 for r in adjacency):
            raise ValueError(f"Dialogue {i} must have a question and eight-node graph")
        if len(row.get("system_prompts",[])) != 8 or not 1 <= len(row.get("attacker_idxes",[])) <= 4:
            raise ValueError(f"Dialogue {i} lacks official graph metadata")
        for replies, infected in zip(turns, labels):
            if (len(replies) != 8 or {r[0] for r in replies} != set(range(8)) or
                any(not isinstance(r[1],str) or not r[1].strip() for r in replies)):
                raise ValueError(f"Dialogue {i} has missing/empty node replies")
            if not isinstance(infected,list) or any(not isinstance(n,int) or n not in range(8) for n in infected):
                raise ValueError(f"Dialogue {i} has invalid per-turn infection labels")
        question_counts[q] = question_counts.get(q,0) + 1
    split = int(len(rows)*.8)
    train = {normalize_question(r["question"]) for r in rows[:split]}
    val = {normalize_question(r["question"]) for r in rows[split:]}
    return {"dialogues":len(rows),"heldout_overlap":0,"unique_questions":len(question_counts),
            "released_split_sizes":[split,len(rows)-split],
            "released_train_val_question_overlap":len(train & val),
            "warning":"Released row split may repeat questions; a grouped split is a separately named adaptation"}


def prepare_infa_training_data(raw, heldout_paths, destination):
    import pandas as pd
    asset = load_lock()["assets"]["csqa_train"]
    if sha256(raw) != asset["sha256"]:
        raise ValueError("Official CSQA training asset hash mismatch")
    if len(heldout_paths) != 5 or len({str(Path(p).resolve()) for p in heldout_paths}) != 5:
        raise ValueError("Five distinct held-out benchmark files are required")
    questions = heldout_questions(heldout_paths)
    frame = pd.read_parquet(raw)
    exclude = frame["question"].map(normalize_question).isin(questions)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    parquet = target/"train-00000-of-00001.parquet"
    if exclude.any():
        frame.loc[~exclude].to_parquet(parquet,index=False)
    else:
        parquet.symlink_to(Path(raw).resolve())
    report = {"dataset":asset["repository"], "revision":asset["revision"], "split":"train",
              "original_sha256":sha256(raw), "original_rows":len(frame),
              "heldout_questions":len(questions), "excluded_rows":int(exclude.sum()),
              "training_rows":int((~exclude).sum()), "training_sha256":sha256(parquet),
              "heldout_files":{str(Path(p).resolve()):sha256(p) for p in heldout_paths},
              "excluded_ids":frame.loc[exclude,"id"].tolist(),
              "normalization":"casefold and whitespace; exact question-stem comparison",
              "not_semantic_deduplication":True}
    (target/"provenance.json").write_text(json.dumps(report,indent=2)+"\n")
    return report
