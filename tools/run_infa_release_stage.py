"""Execute one pinned INFA stage in an isolated server workspace.

The recipe's generator model must match the chosen service. A two-dialogue
protocol check writes outside the formal training set and cannot train a model.
No checkpoints or labels are invented.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate.defense_methods import reproduction as r

def install_api_audit(path):
    """Observe the upstream request unchanged and fail on incomplete generations."""
    from openai.resources.chat.completions import AsyncCompletions
    original = AsyncCompletions.create
    async def audited(self, *args, **kwargs):
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            with Path(path).open("a") as handle:
                handle.write(json.dumps({"model":kwargs.get("model"),"api_error":type(exc).__name__})+"\n")
            raise
        choices = response.choices
        choice = choices[0] if choices else None
        content = choice.message.content if choice is not None else None
        reason = choice.finish_reason if choice is not None else None
        row = {"model":kwargs.get("model"),"finish_reason":reason,
               "content_chars":len(content or ""),
               "completion_tokens":getattr(response.usage,"completion_tokens",None),
               "prompt_tokens":getattr(response.usage,"prompt_tokens",None),
               "max_tokens":kwargs.get("max_tokens"),"temperature":kwargs.get("temperature")}
        with Path(path).open("a") as handle:
            handle.write(json.dumps(row)+"\n")
        if reason != "stop" or not content or not content.strip():
            raise RuntimeError("Incomplete upstream generation; refusing to train on truncated or empty replies")
        return response
    AsyncCompletions.create = audited


def prepare_native_imports(job):
    # This command owns its process. Avoid resolving upstream evaluate/utils/train
    # imports to MAPLE modules already imported by the preparation helper.
    prefixes = ("evaluate", "utils", "train", "MAS", "generate_data")
    for key in list(sys.modules):
        if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[key]
    sys.path.insert(0, str(job))


def require_training_data(recipe, heldout):
    directory=Path(recipe["generation"][0]["argv"][recipe["generation"][0]["argv"].index("--dataset_path")+1])
    manifest=json.loads((directory/"provenance.json").read_text())
    parquet=directory/"train-00000-of-00001.parquet"
    if r.sha256(parquet)!=manifest["training_sha256"]:
        raise ValueError("Frozen training parquet changed")
    supplied={str(Path(p).resolve()):r.sha256(p) for p in heldout}
    if supplied!=manifest["heldout_files"]:
        raise ValueError("Held-out datasets differ from the training input audit")
    return manifest

def prepare(recipe, minilm, heldout):
    source=Path(recipe["source"])
    r.verify_source("infa",source)
    data_manifest=require_training_data(recipe,heldout)
    destination=Path(recipe["working_directory"])
    if destination.exists():
        raise FileExistsError("Choose a fresh INFA working_directory")
    if not (Path(minilm)/"config.json").is_file() or not (Path(minilm)/"modules.json").is_file():
        raise ValueError("Complete local MiniLM assets are required")
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns(".git","__pycache__","*.pyc"))
    link=destination/"train/models/sentence-transformers/all-MiniLM-L6-v2"
    link.parent.mkdir(parents=True,exist_ok=True)
    if link.exists():raise FileExistsError("Refusing to replace an upstream encoder")
    link.symlink_to(Path(minilm).resolve(),target_is_directory=True)
    # Persist the exact expected recipe. Future stage calls must match it.
    manifest={"recipe":recipe,"data":data_manifest,
              "minilm_path":str(Path(minilm).resolve()),
              "minilm_sha256":{str(p.relative_to(minilm)):r.sha256(p)
                   for p in sorted(Path(minilm).rglob("*")) if p.is_file()},
              "status":"prepared_not_trained"}
    (destination/"maple-reproduction.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return {"status":"prepared_not_trained","directory":str(destination)}

def run_stage(recipe, stage, grid_index, heldout, services="", service="", protocol_check=False):
    job=Path(recipe["working_directory"])
    record=json.loads((job/"maple-reproduction.json").read_text())
    if record["recipe"]!=recipe:raise ValueError("Recipe changed after workspace preparation")
    r.verify_source("infa",job)
    require_training_data(recipe,heldout)
    r.verify_files(record["minilm_path"],record["minilm_sha256"])
    formal=Path(recipe["merge"]["argv"][-1])/"train"
    if protocol_check and stage!="generate":
        raise ValueError("Protocol checks cannot enter embedding or training")
    if stage=="generate":
        if grid_index not in range(20):raise ValueError("Grid index must be 0..19")
        selected=recipe["generation"][grid_index]
        argv=list(selected["argv"]);seed=selected["seed"]
        config=json.loads(Path(services).read_text())[service]
        if config["model"]!=recipe["generator_model"]:
            raise ValueError("Configured API model differs from declared recipe model")
        # Public source consumes these exact environment variables.
        os.environ["OPENAI_API_KEY"]=config["api_key"]
        os.environ["BASE_URL"]=config["base_url"]
        label=f"generate-{grid_index:02d}"
        destination=formal
        if protocol_check:
            label+="-protocol"
            argv[argv.index("--samples")+1]="2"
            argv[argv.index("--save_dir")+1]=str(job/"protocol-check")
            destination=job/"protocol-check/PI/csqa/train"
        if list(destination.glob(f"*-num_attackers_{selected['attackers']}-sparsity_{selected['sparsity']}.json")):
            raise FileExistsError("This grid output already exists; inspect it before rerunning")
    else:
        argv=list(recipe[{"merge":"merge","embed":"embedding","train":"training"}[stage]]["argv"])
        seed=recipe["reproduction_seed"];label=stage
        destination=formal
        if stage=="merge":
            if (formal/"dataset.json").exists():raise FileExistsError("Released merge is not idempotent")
            files=list(formal.glob("*.json"))
            if len(files)!=20:raise ValueError("Require exactly 20 complete grid files before merge")
            rows=[]
            for path in files:
                block=json.loads(path.read_text())
                if len(block)!=40:raise ValueError("Each official grid file must contain 40 dialogues")
                rows.extend(block)
            r.validate_infa_dialogues(rows,r.heldout_questions(heldout),800)
            (job/"merge-input-order.json").write_text(json.dumps([p.name for p in files],indent=2))
        else:
            rows=json.loads((formal/"dataset.json").read_text())
            r.validate_infa_dialogues(rows,r.heldout_questions(heldout),800)
            if stage=="embed":
                output=Path(recipe["training"]["argv"][recipe["training"]["argv"].index("--dataset_path")+1])
                if output.exists():raise FileExistsError("Embedding output already exists")
            else:
                # Require the embedding step's completion marker, not an arbitrary pickle.
                if not (job/"stage-results/embed.json").is_file():
                    raise ValueError("Run and validate the pinned embedding stage first")
                embedded=json.loads((job/"stage-results/embed.json").read_text())
                expected_feature=Path(argv[argv.index("--dataset_path")+1])
                if embedded["output"]!=str(expected_feature) or r.sha256(expected_feature)!=embedded["sha256"]:
                    raise ValueError("Training feature pickle differs from the verified embedding output")
    logdir=job/"stage-results";logdir.mkdir(exist_ok=True)
    marker=logdir/(label+".json")
    if marker.exists():raise FileExistsError("Stage has already completed")
    # Each invocation runs in its own process: global RNG/cwd changes cannot affect evaluations.
    import numpy as np
    import torch
    torch.set_num_threads(2)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    before=set(destination.glob("*.json")) if destination.exists() else set()
    old=Path.cwd()
    if stage=="generate":
        install_api_audit(logdir/(label+"-api.jsonl"))
    try:
        os.chdir(job);prepare_native_imports(job);sys.argv=argv
        with (logdir/(label+".log")).open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            runpy.run_path(str(job/argv[0]),run_name="__main__")
    finally:
        os.chdir(old)
    import importlib.metadata
    versions={}
    for distribution in ("torch","torch-geometric","torch-scatter","transformers",
                         "sentence-transformers","openai","datasets","pyarrow","numpy"):
        try: versions[distribution]=importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError: versions[distribution]=None
    imported={}
    for name in ("evaluate.evaluate_output","MAS.agents","train.models.defender.model"):
        module=sys.modules.get(name)
        if module is not None:
            path=Path(module.__file__).resolve()
            if not path.is_relative_to(job.resolve()):
                raise ValueError(f"Native import escaped the pinned source: {name}")
            imported[name]=str(path)
    report={"runtime_versions":versions,"native_import_paths":imported,"stage":stage,"status":"completed","protocol_check":protocol_check,
            "source_revision":recipe["source_revision"],"seed":seed,
            "generator_model":recipe["generator_model"],"model_substitution":recipe["model_substitution"],
            "author_checkpoint":False}
    if stage=="generate":
        after=set(destination.glob("*.json"))-before
        if len(after)!=1:raise ValueError("Expected exactly one generated graph file")
        output=after.pop()
        report["validation"]=r.validate_infa_dialogues(json.loads(output.read_text()),
                    r.heldout_questions(heldout),2 if protocol_check else 40)
        report["output"]=str(output);report["sha256"]=r.sha256(output)
    elif stage in ("merge","embed"):
        output=(formal/"dataset.json") if stage=="merge" else Path(recipe["training"]["argv"][recipe["training"]["argv"].index("--dataset_path")+1])
        report["output"]=str(output);report["sha256"]=r.sha256(output)
    else:
        weights=list((job/"checkpoints").rglob("*.pth"))
        if not weights:raise ValueError("Training produced no checkpoint")
        report["checkpoints"]={str(p):r.sha256(p) for p in weights}
        report["evaluation_ready"]=False
        report["next_gate"]="strict native checkpoint load and held-out protocol tasks"
    marker.write_text(json.dumps(report,indent=2)+"\n")
    return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe",required=True)
    p.add_argument("--stage",choices=("prepare","generate","merge","embed","train"),required=True)
    p.add_argument("--grid-index",type=int,default=0)
    p.add_argument("--minilm-model",default="")
    p.add_argument("--heldout",nargs=5,required=True)
    p.add_argument("--services",default="")
    p.add_argument("--service",default="")
    p.add_argument("--protocol-check",action="store_true")
    args=p.parse_args(argv)
    recipe=json.loads(Path(args.recipe).read_text())
    expected=r.infa_recipe(recipe["source"],recipe["working_directory"],
         recipe["generation"][0]["argv"][recipe["generation"][0]["argv"].index("--dataset_path")+1],
         recipe["generator_model"],recipe["reproduction_seed"])
    # Reject edited commands even if someone retained an old provenance block.
    candidate={key:value for key,value in recipe.items() if key!="provenance"}
    if candidate!=expected:raise ValueError("Recipe differs from the pinned release specification")
    if args.stage=="prepare":
        report=prepare(recipe,args.minilm_model,args.heldout)
    else:
        report=run_stage(recipe,args.stage,args.grid_index,args.heldout,args.services,args.service,args.protocol_check)
    print(json.dumps(report,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
