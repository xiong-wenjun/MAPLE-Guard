"""Prepare auditable release configurations; never synthesize missing trained assets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate.defense_methods import reproduction as repro

def write_new(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:
        handle.write(content if isinstance(content,str) else json.dumps(content,indent=2,ensure_ascii=False)+"\n")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action",required=True)
    audit = sub.add_parser("audit-source")
    audit.add_argument("--method",choices=("agentsafe","infa","guardian"),required=True)
    audit.add_argument("--source",required=True)
    safe = sub.add_parser("agentsafe-relations")
    safe.add_argument("--source",required=True)
    safe.add_argument("--names",nargs="+",required=True)
    safe.add_argument("--seed",type=int,required=True)
    safe.add_argument("--output-dir",required=True)
    recipe = sub.add_parser("infa-recipe")
    recipe.add_argument("--source",required=True)
    recipe.add_argument("--job-dir",required=True)
    recipe.add_argument("--dataset-dir",required=True)
    recipe.add_argument("--model",default="gpt-4o-mini")
    recipe.add_argument("--generation-profile",choices=("released","qwen_no_thinking"),default="released")
    recipe.add_argument("--seed",type=int,required=True)
    recipe.add_argument("--output",required=True)
    data = sub.add_parser("prepare-infa-data")
    data.add_argument("--raw-parquet",required=True)
    data.add_argument("--heldout",nargs=5,required=True)
    data.add_argument("--output-dir",required=True)
    validate = sub.add_parser("validate-infa-dialogues")
    validate.add_argument("--data",required=True)
    validate.add_argument("--heldout",nargs=5,required=True)
    validate.add_argument("--expected-count",type=int,default=800)
    check = sub.add_parser("validate-infa-checkpoint")
    check.add_argument("--source",required=True)
    check.add_argument("--checkpoint",required=True)
    bert = sub.add_parser("verify-bert")
    bert.add_argument("--directory",required=True)
    args = parser.parse_args(argv)
    if args.action == "audit-source":
        report = repro.verify_source(args.method,args.source)
    elif args.action == "agentsafe-relations":
        destination=Path(args.output_dir)
        destination.mkdir(parents=True,exist_ok=False)
        text,policy = repro.agentsafe_relations(args.source,args.names,args.seed)
        write_new(destination/"relations.txt",text)
        write_new(destination/"maple-policy.json",policy)
        report={"output_dir":str(destination),"relations":len(text.splitlines()),
                "criteria_recovered":False,"admission_threshold_recovered":False,
                "full_paper_configuration_recovered":False,"provenance":policy["provenance"]}
        write_new(destination/"provenance.json",report)
    elif args.action == "infa-recipe":
        provenance = repro.verify_source("infa",args.source)
        report = repro.infa_recipe(args.source,args.job_dir,args.dataset_dir,args.model,args.seed,args.generation_profile)
        report["provenance"] = provenance
        write_new(args.output,report)
    elif args.action == "prepare-infa-data":
        report = repro.prepare_infa_training_data(args.raw_parquet,args.heldout,args.output_dir)
    elif args.action == "validate-infa-dialogues":
        rows=json.loads(Path(args.data).read_text())
        report=repro.validate_infa_dialogues(rows,repro.heldout_questions(args.heldout),args.expected_count)
        report["data_sha256"]=repro.sha256(args.data)
        report["heldout_sha256"]={str(p):repro.sha256(p) for p in args.heldout}
    elif args.action == "verify-bert":
        report = repro.verify_bert(args.directory)
    else:
        import torch
        from evaluate.defense_methods.infa_full import validate_checkpoint_payload, load_native_class
        provenance = repro.verify_source("infa",args.source)
        state = validate_checkpoint_payload(torch.load(args.checkpoint,map_location="cpu",weights_only=True))
        cls,_ = load_native_class(args.source)
        model=cls(384,1024,2,heads=8,edge_dim=(4,384),guard="ours")
        model.load_state_dict(state,strict=True)
        model.eval()
        edges=torch.tensor([[0,1],[1,0]],dtype=torch.long)
        with torch.no_grad():
            output=model(torch.zeros(2,384),edges,torch.zeros(2,4,384),
                         node_self_replies=torch.zeros(2,4,384))
        if tuple(output.shape) != (2,2) or not torch.isfinite(output).all():
            raise ValueError("Native INFA must return two finite scores per node")
        report={**provenance,"checkpoint_sha256":repro.sha256(args.checkpoint),
                "strict_load":True,"forward_shape":list(output.shape),
                "training_provenance_verified":False,"note":"Structural compatibility alone does not prove training"}
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
