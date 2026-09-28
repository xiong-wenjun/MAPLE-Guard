"""Validate full-baseline configuration or inspect a checkpoint without agent calls."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluate.defense_methods.full_runtime import FULL_METHODS, FullRuntime, add_full_baseline_args


def inspect_checkpoint(path):
    import torch
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("Expected a raw state dict or wrapped model_state_dict")
    state = value.get("model_state_dict", value)
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no state dictionary")
    groups = ("shared_convs.", "branch_convs.", "input_proj.",
              "branch_heads_mal.", "branch_heads_inf.")
    return {"checkpoint":str(path.resolve()),
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "model_class":value.get("model_class"),
            "model_kwargs":value.get("model_kwargs"),
            "method_scope":value.get("method_scope"),
            "parameter_shapes":{name:list(tensor.shape) for name,tensor in state.items() if hasattr(tensor,"shape")},
            "native_infa_key_groups":{prefix:any(name.startswith(prefix) for name in state) for prefix in groups},
            "strict_model_load_tested":False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=FULL_METHODS)
    parser.add_argument("--inspect-checkpoint", default="")
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--chat-base-url", default="")
    parser.add_argument("--chat-model", default="")
    parser.add_argument("--embed-base-url", default="")
    parser.add_argument("--embed-model", default="")
    parser.add_argument("--memory-backend", choices=("memory", "memrl"), default="memory")
    parser.add_argument("--disable-chat-thinking", action="store_true")
    parser.add_argument("--preload-haystack", action="store_true")
    add_full_baseline_args(parser)
    args = parser.parse_args(argv)
    try:
        if args.inspect_checkpoint:
            report = inspect_checkpoint(args.inspect_checkpoint)
        else:
            if not args.method:
                parser.error("--method or --inspect-checkpoint is required")
            runtime = FullRuntime(args)
            report = {"method":args.method, "status":"configuration_loaded",
                      "live_agent_calls":False, "benchmark_performance_tested":False,
                      "guard_class":type(runtime.guard).__name__,
                      "state_path":runtime.state_path or None,
                      "provenance":getattr(runtime.guard, "provenance", None)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status":"not_ready", "error_type":type(exc).__name__,
                          "error":str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
