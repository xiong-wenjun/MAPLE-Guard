"""Audit fixed user benchmark bundles and unwrap supported QA datasets."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from maple_guard.benchmarks.benchmark_bundle import load_bundle,appworld_cases,normalized_rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",required=True)
    p.add_argument("--output-dir",required=True)
    args=p.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    audit=[]
    for source in sorted(Path(args.input_dir).glob("*_200.json")):
        bundle,manifest=load_bundle(source)
        benchmark=manifest["benchmark"]
        if benchmark=="appworld":
            cases=appworld_cases(bundle,str(source))
            manifest.update(evaluation_protocol="appworld_action_selection_proxy",native_execution=False,metadata_policy="provided_specs_only")
        else:
            rows=normalized_rows(bundle)
            target=output/(benchmark+"_200.json")
            if target.resolve()==source.resolve(): raise ValueError("Refusing to overwrite source bundle")
            target.write_text(json.dumps(rows,ensure_ascii=False))
            manifest["normalized_file"]=str(target.resolve())
            if benchmark=="injecagent":
                manifest["runner_status"]="native source cases only; transfer runner requires explicit adapter, not generic QA"
        audit.append(manifest)
    if not audit: raise ValueError("No benchmark bundles found")
    (output/"dataset-audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps([{"benchmark":m["benchmark"],"count":m["total_cases"],"sha256":m["source_sha256"]} for m in audit],indent=2))
if __name__=="__main__":main()
