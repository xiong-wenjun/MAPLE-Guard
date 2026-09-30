"""Explicit, non-pickle serializers for task-boundary defense runtimes."""
from __future__ import annotations
import copy
from types import SimpleNamespace

def _error(message):
    from .task_checkpoint import CheckpointError
    return CheckpointError(message)

def capture_guard(guard):
    if guard is None:
        return None
    from evaluate.defense_methods.agentsafe_full import AgentSafeFull
    from evaluate.defense_methods.amemguard_full import AMemGuardFull
    from evaluate.defense_methods.agentxposed_full import AgentXposedFull
    from evaluate.defense_methods.infa_full import InfaGuardFull
    from evaluate.defense_methods.piguard import PIGuardDetector
    supported = (AgentSafeFull, AMemGuardFull, AgentXposedFull, InfaGuardFull, PIGuardDetector)
    if type(guard) not in supported:
        raise _error("Unsupported guard checkpoint: " + type(guard).__name__)
    excluded = {"judge", "embed", "judge_budget", "_released_engine", "detector",
                "_torch", "model", "tokenizer"}
    result = {"class":type(guard).__name__,
              "attributes":{k:v for k,v in vars(guard).items() if k not in excluded},
              "clients":{k:copy.deepcopy(getattr(getattr(guard,k,None),"counters",None))
                         for k in ("judge","embed")}}
    if hasattr(guard, "state_dict"):
        result["validated_state"] = guard.state_dict()
    engine = getattr(guard, "_released_engine", None)
    if engine is not None:
        result["released_engine"] = {"calls":engine.calls, "logs":engine.logs}
    if isinstance(guard, InfaGuardFull):
        from evaluate.defense_methods.infa_full import NativeInfaDetector
        if type(guard.detector) is not NativeInfaDetector:
            raise _error("Only the native INFA detector supports durable resume")
        result["detector"] = {
            "weights":guard.detector.model.state_dict(),
            "training":guard.detector.model.training,
            "provenance":guard.detector.provenance}
    return result

def install_guard(guard, saved):
    if saved is None:
        if guard is not None:
            raise _error("Checkpoint guard identity mismatch")
        return
    if type(guard).__name__ != saved["class"]:
        raise _error("Checkpoint guard class mismatch")
    attributes = copy.deepcopy(saved["attributes"])
    for key, value in copy.deepcopy(attributes).items():
        setattr(guard, key, value)
    if "validated_state" in saved:
        guard.load_state_dict(saved["validated_state"])
        # Some sidecar loaders intentionally clear transient counters/prompts.
        for key, value in attributes.items():
            setattr(guard, key, value)
    for name, counters in saved["clients"].items():
        if counters is not None:
            client = getattr(guard, name)
            client.counters.clear()
            client.counters.update(counters)
    if "released_engine" in saved:
        for key,value in saved["released_engine"].items():
            setattr(guard._released_engine,key,value)
    if "detector" in saved:
        detector = saved["detector"]
        if guard.detector.provenance != detector["provenance"]:
            raise _error("INFA checkpoint assets or detector protocol changed")
        guard.detector.model.load_state_dict(detector["weights"], strict=True)
        guard.detector.model.train(detector["training"])

def capture_runtime(runtime, link):
    if runtime is None:
        return None
    from evaluate.defense_methods.comparison_runtime import ComparisonRuntime
    from evaluate.defense_methods.full_runtime import FullRuntime
    if type(runtime) not in (ComparisonRuntime, FullRuntime):
        raise _error("Unsupported runtime checkpoint")
    # Generation closures only exist within a task and begin_task clears them.
    # Persistent contexts, histories, overlays, references and counters are kept.
    excluded = {"args", "rules", "guard", "target_context_budget", "generators"}
    return {"kind":type(runtime).__name__, "method":runtime.method,
            "attributes":link({k:v for k,v in vars(runtime).items() if k not in excluded}),
            "guard":capture_guard(runtime.guard)}

def restore_runtime(saved, args, unlink):
    from evaluate.defense_methods.comparison_runtime import ComparisonRuntime
    from evaluate.defense_methods.full_runtime import FullRuntime, _factory
    if saved["method"] != args.method:
        raise _error("Checkpoint runtime method mismatch")
    if saved["kind"] == "ComparisonRuntime":
        runtime = ComparisonRuntime(args)
    elif saved["kind"] == "FullRuntime":
        runtime = FullRuntime(args, guard=_factory(args, checkpoint=saved["guard"]))
    else:
        raise _error("Unknown checkpoint runtime kind")
    install_guard(runtime.guard, saved["guard"])
    for key,value in unlink(saved["attributes"]).items():
        setattr(runtime,key,value)
    return runtime

def capture_official(state):
    from evaluate.defense_methods.base import OfficialDefenseState
    if type(state) is not OfficialDefenseState:
        raise _error("Unsupported communication state")
    fields = {k:v for k,v in vars(state).items() if k != "gnn_state"}
    graph = dict(state.gnn_state)
    # GUARDIAN creates/trains a new model and optimizer per round; these caches
    # contain imported modules/parser functions only, not learned parameters.
    guardian_cache = graph.pop("guardian_runtime_cache", {})
    fields["_checkpoint_guardian_caches"] = [v["code_dir"] for v in guardian_cache.values()]
    cache = graph.pop("runtime_cache", {})
    fields["gnn_state"] = graph
    fields["_checkpoint_gnn_caches"] = [
        {"key":key,"checkpoint":v["checkpoint"],"embedding_model":v["embedding_model"],
         "device":v["device"]} for key,v in cache.items()]
    return fields

def decode_official(fields):
    from evaluate.defense_methods.base import OfficialDefenseState
    state = OfficialDefenseState()
    vars(state).update(fields)
    return state

def hydrate_official(value, args):
    from evaluate.defense_methods.base import OfficialDefenseState, DefenseContext
    if isinstance(value, OfficialDefenseState):
        from evaluate.defense_methods.gnn_defense import _load_runtime
        from evaluate.defense_methods.guardian_defense import _load_official_runtime
        for code_dir in vars(value).pop("_checkpoint_guardian_caches", []):
            runtime,error = _load_official_runtime(value,code_dir)
            if error or runtime is None:
                raise _error("Cannot reconstruct pinned GUARDIAN runtime: " + str(error))
        for spec in vars(value).pop("_checkpoint_gnn_caches", []):
            local = copy.copy(args)
            local.official_defense_gnn_checkpoint = spec["checkpoint"]
            local.official_defense_embedding_model = spec["embedding_model"]
            local.official_defense_gnn_device = spec["device"]
            guard = spec["key"].split("|",1)[0]
            runtime,error = _load_runtime(value, DefenseContext(
                args.method,"", "",0,[],local), guard)
            if error or runtime is None:
                raise _error("Cannot reconstruct pinned GNN runtime: " + str(error))
    elif isinstance(value, dict):
        for item in value.values():
            hydrate_official(item,args)
    elif isinstance(value,(list,tuple)):
        for item in value:
            hydrate_official(item,args)
