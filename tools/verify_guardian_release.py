"""Real CPU/BERT differential check against the pinned public GUARDIAN entry points.

This validates the detector and graph adapter, not benchmark accuracy or the
original end-to-end agent protocol. Runs the original 20 epochs in both paths.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate.defense_methods import guardian_defense as g
from evaluate.defense_methods import reproduction
from evaluate.defense_methods.base import DefenseContext, OfficialDefenseState

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--bert-dir",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args(argv)
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    code=Path(args.source).resolve()/"code/communication-targeted_error_injection_and_propagation"
    cfg=SimpleNamespace(official_defense_guardian_profile="released_detector",
          official_defense_guardian_epochs=20,official_defense_gnn_device="cpu",
          official_defense_guardian_bert_dir=args.bert_dir)
    ctx=DefenseContext("guardian","protocol_check","Which option?",0,[[0,1,0],[0,0,1],[0,0,0]],cfg)
    assets=g._validate_released_profile(ctx,str(code))
    runtime,error=g._load_official_runtime(OfficialDefenseState(),str(code))
    if runtime is None:raise RuntimeError(error)
    parse=reproduction.guardian_parser(code)
    turns=[{0:parse("Answer: (A)"),1:parse("Answer: (A)"),2:parse("Answer: (D)")},
           {0:parse("Answer: (A)"),1:parse("Answer: (B)"),2:parse("Answer: (D)")}]
    records=[]
    for count in (1,2):
        # Independent public graph construction, then compare with host adapter.
        ids=[0,1,2]
        edge_index=torch.tensor([[i,j] for i in ids for j in ids if i!=j]).t().contiguous()
        adjacency=torch.ones(3,3)-torch.eye(3)
        direct=[runtime["Data"](text=[turn[i] for i in ids],
                 edge_index=edge_index.clone(),adj=adjacency.clone()) for turn in turns[:count]]
        adapted=g._build_graph_data(runtime,ctx,turns[:count],ids)
        for left,right in zip(direct,adapted):
            assert left.text==right.text
            assert torch.equal(left.edge_index,right.edge_index)
            assert torch.equal(left.adj,right.adj)
        torch.manual_seed(args.seed)
        with g._bert_lookup_context(ctx),contextlib.redirect_stdout(io.StringIO()):
            if count==1:
                model=runtime["static_module"].DOMINANTDetector(hid_dim=128,num_gnn_layers=2)
                optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                model.fit(direct[0],num_epochs=20,optimizer=optimizer)
                result=model.detect(direct[0])
                del optimizer
            else:
                model=runtime["temporal_module"].train_model(direct)
                result=model.detect(direct)
        expected_index=int(result[1]);expected=np.asarray(result[2],dtype=float).tolist()
        del model,result
        gc.collect()
        torch.manual_seed(args.seed)
        index,scores,detector=g._run_official_guardian(runtime,adapted,ctx)
        np.testing.assert_allclose(scores,expected,rtol=1e-6,atol=1e-7)
        if index!=expected_index:raise AssertionError("Pruned node differs from public detector")
        records.append({"turns":count,"detector":detector,"epochs":20,"node_index":index,
                        "scores":scores,"max_abs_difference":float(np.max(np.abs(np.array(scores)-expected))),
                        "graph_equal":True,"parsed_answers_equal":True})
        print(json.dumps(records[-1]),flush=True)
        gc.collect()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as handle:
        json.dump({"status":"detector_parity_passed","assets":assets,"seed":args.seed,
                   "torch_version":torch.__version__,"checks":records,
                   "benchmark_tested":False,"end_to_end_paper_reproduction":False},handle,indent=2)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
