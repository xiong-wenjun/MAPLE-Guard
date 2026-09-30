"""Runtime for matched lifecycle controls and source-based memory defenses."""
from __future__ import annotations
from maple_guard import budget_outcomes as budget
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from .rule_baseline import ProvenanceACL

MEMORY_METHODS = ("provenance_acl", "maple_guard_retrieval_only", "amemguard_full",
                  "piguard_retrieval", "piguard_lifecycle")
MATCHED_METHODS = ("maple_guard", "maple_guard_retrieval_only")

def model_clients(args):
    from .amemguard_transport import max_attempts_from_environment, request_json
    from .full_runtime import _positive_judge_timeout
    configured_timeout = getattr(args, "full_judge_timeout", None)
    judge_timeout = 120.0 if configured_timeout is None else _positive_judge_timeout(configured_timeout)
    max_attempts = max_attempts_from_environment()
    def request(base, path, body, key):
        return request_json(base, path, body, key,
                            timeout=judge_timeout if path == "/chat/completions" else 120.0,
                            max_attempts=max_attempts)
    @budget.role_call("defense")
    def judge(messages):
        base=getattr(args,"full_judge_base_url","") or getattr(args,"chat_base_url","")
        model=getattr(args,"full_judge_model","") or getattr(args,"chat_model","")
        if not base or not model: raise ValueError("A-MemGuard requires a real judge endpoint/model")
        body={"model":model,"messages":messages,"temperature":0.0,
              "max_tokens":getattr(args,"full_judge_max_tokens",4096)}
        if getattr(args,"disable_chat_thinking",False):
            body["chat_template_kwargs"]={"enable_thinking":False}
        key=getattr(args,"full_judge_api_key","") or os.getenv("FULL_BASELINE_API_KEY","") or os.getenv("OPENAI_API_KEY","")
        data=request(base,"/chat/completions",body,key)
        budget.handle_response(data, body, request_id=data.get("_maple_request_id"), endpoint=base.rstrip("/") + "/chat/completions", timeout=judge_timeout)
        choice=data["choices"][0]
        if choice.get("finish_reason")=="length": raise RuntimeError("A-MemGuard judge output truncated")
        return choice["message"]["content"]
    def embed(text):
        base=getattr(args,"embed_base_url","")
        model=getattr(args,"embed_model","")
        if not base or not model: raise ValueError("A-MemGuard requires a real embedding endpoint/model")
        key=getattr(args,"embed_api_key","") or os.getenv("EMBED_API_KEY","")
        return request(base,"/embeddings",{"model":model,"input":text},key)["data"][0]["embedding"]
    from .full_runtime import tracked_client
    return tracked_client(judge),tracked_client(embed)

class ComparisonRuntime:
    def __init__(self,args,guard=None):
        self.args,self.method=args,args.method
        from .full_runtime import STRICT_COMMUNICATION_METHODS
        self.uses_legacy_communication_adapter = self.method in STRICT_COMMUNICATION_METHODS
        from .full_runtime import experiment_identity
        self.experiment_identity=experiment_identity(args)
        self._scope_identity=self.experiment_identity
        self.rules=ProvenanceACL()
        self.guard=guard
        if guard is None and self.method=="amemguard_full":
            from .amemguard_full import AMemGuardFull
            judge,embed=model_clients(args)
            self.guard=AMemGuardFull(args,judge,embed)
        if guard is None and self.method.startswith("piguard_"):
            from .piguard import PIGuardDetector
            self.guard=PIGuardDetector(args)
        self.state_path=str(getattr(args,"baseline_state_path","") or "")
        if getattr(args,"memory_backend","")=="memrl" and not self.state_path:
            raise ValueError("Strict persistent comparisons require --baseline-state-path")
        self._pi_cache={}
        self.detector_wall_seconds=0.0
        self._embedding_cache={}
        self.replacements={}
        self.ledger={}
        self.selected={}
        self.consumed={}
        self.received={}
        self.pending=[]
        self.retrieval_queries={}
        self.question=""
        self.task_id=""
        if self.state_path and Path(self.state_path).exists():
            state=json.loads(Path(self.state_path).read_text())
            if state.get("version")!=2 or state.get("method")!=self.method:
                raise ValueError("Comparison sidecar belongs to a different method/version")
            if state.get("experiment_identity")!=self.experiment_identity:
                raise ValueError("Comparison state belongs to a different experiment/configuration")
            self.ledger=state["ledger"]
            for record in self.ledger.values(): self.rules.validate(record["label"])
            if self.guard is not None and hasattr(self.guard,"load_state_dict"):
                self.guard.load_state_dict(state.get("guard",{}))

    @staticmethod
    def digest(entry):
        return hashlib.sha256((str(entry.intent)+"\n"+str(entry.experience)).encode()).hexdigest()

    def save(self):
        if not self.state_path: return
        path=Path(self.state_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        state={"version":2,"method":self.method,"ledger":self.ledger,"experiment_identity":self.experiment_identity,
               "guard":self.guard.state_dict() if hasattr(self.guard,"state_dict") else {}}
        temp=path.with_name(path.name+".tmp."+str(os.getpid()))
        temp.write_text(json.dumps(state,ensure_ascii=False))
        os.replace(temp,path)

    def begin_task(self,task_id,question):
        self.task_id,self.question=str(task_id),str(question)
        self.selected,self.consumed,self.received={}, {}, {}
        self.retrieval_queries={}
        self._pi_cache={}
        if hasattr(self.guard,"begin_task"): self.guard.begin_task(str(task_id))

    def _apply_view(self,entry,label,channel,parent_ids):
        entry.baseline_metadata=copy.deepcopy(getattr(entry,"baseline_metadata",{}))
        entry.baseline_metadata["operational"]=copy.deepcopy(label)
        entry.baseline_metadata["history_display"]={}
        entry.allowed_agents=list(label["readers"])
        entry.allowed_task_classes=[]
        entry.allowed_tools=[]
        entry.parents=list(parent_ids)
        entry.source_type="longmemeval_user_history_seed" if channel=="user_history" else channel
        entry.memory_type="user_history" if channel=="user_history" else "episodic_experience"
        entry.taint="external" if "untrusted_external" in label["taints"] else ("unverified" if label["taints"] else "clean")
        entry.provenance_trust=0.5
        entry.source_agent_trust=0.5
        entry.content_hazard=0.0
        # Q updates remain a separately controlled feedback protocol; never seed
        # the value from attacker metadata or benchmark success tags.
        entry.utility_q=0.0
        entry.success_count=entry.failure_count=0
        entry.embedding=copy.deepcopy(self._embedding_cache.get(self.digest(entry)))

    def observe_ingress(self,entry,channel,scope,recipient,parents=None):
        history_display={}
        if channel=="user_history":
            for parent in getattr(entry,"parents",[]):
                key,sep,value=str(parent).partition("=")
                if sep and key in {"session_id","round","date","granularity"}:
                    history_display[key]=value
        readers=list(range(int(getattr(self.args,"agents",1)))) if scope=="team" else [int(recipient)]
        owner=int(recipient)  # authenticated store custodian; source identity may be -1
        generated = channel in {"agent_output", "peer_message"}
        dependencies=list(parents if parents is not None else (self.consumed.get(owner,{}).values() if generated else []))
        labels=[self._label(m) for m in dependencies]
        if generated:labels.extend(self.received.get(owner,[]))
        label=self.rules.label(owner=owner,readers=readers,channel=channel,parents=labels)
        ids=[str(m.memory_id) for m in dependencies]
        entry.memory_scope="team" if scope=="team" else "agent_private"
        entry.status="active"
        self._apply_view(entry,label,channel,ids)
        entry.baseline_metadata["history_display"]=copy.deepcopy(history_display)
        self.ledger[str(entry.memory_id)]={"digest":self.digest(entry),"label":copy.deepcopy(label),
                                         "channel":channel,"parents":ids,"history_display":history_display,
                                         "storage":{field:getattr(entry,field) for field in self.STORAGE_FIELDS}}
        self.save()

    def _label(self,entry):
        record=self.ledger.get(str(entry.memory_id))
        if record is None or record["digest"]!=self.digest(entry):
            label=self.rules.label(owner=-1,readers=[],channel="unknown")
            # Unauthenticated loaded/mutated memory cannot bring its own trusted label.
            self._apply_view(entry,label,"unknown",[])
            entry.status="active"
            entry.origin_task="unknown"
            entry.origin_agent=-1
            entry.retrieval_key=""
            return label
        self._apply_view(entry,record["label"],record["channel"],record["parents"])
        entry.baseline_metadata["history_display"]=copy.deepcopy(record.get("history_display",{}))
        for field,value in record.get("storage",{}).items():
            if field in self.STORAGE_FIELDS:setattr(entry,field,value)
        for field,value in record.get("feedback",{}).items():
            if field in self.FEEDBACK_FIELDS:setattr(entry,field,value)
        return record["label"]

    STORAGE_FIELDS=("origin_agent","origin_task","origin_round","memory_scope","status","retrieval_key")

    def record_embedding(self,entry):
        self._embedding_cache[self.digest(entry)]=copy.deepcopy(entry.embedding)

    FEEDBACK_FIELDS=("utility_q","success_count","failure_count","provenance_trust","content_hazard")

    def record_feedback(self,entry):
        record=self.ledger.get(str(entry.memory_id))
        if record is not None and record["digest"]==self.digest(entry):
            record["feedback"]={field:getattr(entry,field) for field in self.FEEDBACK_FIELDS}
            self.save()

    def record_store(self,entry):
        # Called only after authorized defense rewrite/store operations.
        record=self.ledger.get(str(entry.memory_id))
        if record is not None:
            record["digest"]=self.digest(entry)
            record["storage"]={field:getattr(entry,field) for field in self.STORAGE_FIELDS}
            self.record_feedback(entry)

    @staticmethod
    def _decision(entry,agent,allowed,stage,details):
        return {"agent_id":int(agent),"memory_id":str(entry.memory_id),
                "action":"allow" if allowed else "block","reason":stage,
                "details":{"method_scope":"strict_memory_comparison",**details}}

    def _pi(self,entry,agent,stage):
        key=(self.task_id,stage,int(agent),str(entry.memory_id),self.digest(entry))
        if key in self._pi_cache:
            allowed,decision=copy.deepcopy(self._pi_cache[key])
            decision["details"]["cache_hit"]=True
            return allowed,decision
        start=time.perf_counter()
        try:
            result=self.guard.score(str(entry.intent)+"\n"+str(entry.experience))
        finally:
            self.detector_wall_seconds+=time.perf_counter()-start
        score=float(result["injection_score"] if isinstance(result,dict) else result)
        threshold=float(getattr(self.args,"piguard_threshold",0.5))
        allowed=score<=threshold
        decision=self._decision(entry,agent,allowed,stage,
                    {"injection_score":score,"threshold":threshold,"cache_hit":False,
                     "detection_event_id":hashlib.sha256(repr(key).encode()).hexdigest()})
        self._pi_cache[key]=(allowed,copy.deepcopy(decision))
        return allowed,decision

    def admit(self,entry,scope,recipient):
        label=self._label(entry)
        decisions=[]
        if self.method=="provenance_acl":
            phases=["write"]+(["promotion"] if scope=="team" else [])
            for phase in phases:
                recipients=list(range(int(getattr(self.args,"agents",1)))) if phase=="promotion" else [recipient]
                allowed,details=self.rules.decide(label,phase,actor=recipient,recipients=recipients)
                decisions.append(self._decision(entry,recipient,allowed,"provenance_acl_"+phase,details))
                if not allowed: return False,decisions
        elif self.method=="piguard_lifecycle":
            for phase in ["write"]+(["promotion"] if scope=="team" else []):
                allowed,decision=self._pi(entry,recipient,"piguard_"+phase)
                decisions.append(decision)
                if not allowed:return False,decisions
        if hasattr(self.guard,"register_entries"):
            readers=label["readers"] if scope=="team" else [recipient]
            for aid in readers:self.guard.register_entries([entry],aid)
        self.save()
        return True,decisions

    def prepare_entries(self,entries):
        for entry in entries:self._label(entry)
        return entries

    def filter_entries(self,entries,recipient):
        retained,decisions=[],[]
        for entry in entries:
            label=self._label(entry)
            allowed=True
            if self.method=="provenance_acl":
                allowed,details=self.rules.decide(label,"retrieval",actor=recipient)
                decisions.append(self._decision(entry,recipient,allowed,"provenance_acl_retrieval",details))
            elif self.method.startswith("piguard_"):
                allowed,decision=self._pi(entry,recipient,"piguard_retrieval")
                if not decision["details"]["cache_hit"]:decisions.append(decision)
            if allowed:retained.append(entry)
        if hasattr(self.guard,"register_entries"):
            self.guard.register_entries(retained,recipient)
        self.save()
        return retained,decisions

    def validate_retrieval(self,entries,query,recipient):
        if self.method!="amemguard_full":return entries,[]
        self.retrieval_queries[recipient]=query
        retained,details=self.guard.select(query,list(entries),recipient)
        kept={str(x.memory_id) for x in retained}
        audited=set(details.get("retained_ids",[])) | set(details.get("rejected_ids",[]))
        decisions=[self._decision(m,recipient,str(m.memory_id) in kept,"amemguard_consensus",details) for m in entries if str(m.memory_id) in audited]
        self.save()
        return retained,decisions

    def select_memories(self,selected):
        self.selected={i:list(records) for i,records in selected.items()}
        for aid,records in selected.items():
            bank=self.consumed.setdefault(aid,{})
            for entry in records:bank[str(entry.memory_id)]=entry
        return {i:list(records) for i,records in selected.items()}

    @staticmethod
    def _distinct_labels(labels):
        # ACL intersection and source/taint union are idempotent. Routing a
        # cycle must not enumerate every path through the communication graph.
        result, seen = [], set()
        for label in labels:
            key = json.dumps(label, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(copy.deepcopy(label))
        return result

    def route(self,text,sender,recipient):
        labels=[self._label(m) for m in self.consumed.get(sender,{}).values()]
        labels.extend(self.received.get(sender,[]))
        labels=self._distinct_labels(labels)
        if self.method=="provenance_acl":
            for label in labels:
                allowed,details=self.rules.decide(label,"transfer",actor=sender,recipients=[recipient])
                self.pending.append({"agent_id":sender,"action":"allow" if allowed else "block",
                                     "reason":"provenance_acl_transfer","details":details})
                if not allowed:return None
        elif self.method=="piguard_lifecycle":
            from types import SimpleNamespace
            allowed,decision=self._pi(SimpleNamespace(intent="",experience=text,memory_id="communication"),sender,"piguard_transfer")
            self.pending.append(decision)
            if not allowed:return None
        self.received[recipient]=self._distinct_labels([*self.received.get(recipient,[]),*labels])
        return text

    def peer_context(self,outputs,adjacency,recipient):
        if not getattr(self.args,"peer_communication",False):return ""
        from .full_runtime import peer_context
        return peer_context(outputs,adjacency,recipient,self.route)

    def observe_task_inputs(self,raw):
        """Label lower-trust tool observations from the actual input carrier.
        Authenticated user requests are application roots. Never read attack
        capability/attacker identity/poison labels to decide trust.
        """
        if not isinstance(raw,dict):return
        keys=("tool_output","tool_outputs","external_observations","eop_tool_output")
        if not any(raw.get(key) for key in keys):return
        readers=list(range(int(getattr(self.args,"agents",1))))
        label=self.rules.label(owner=-1,readers=readers,channel="external")
        for agent in readers:self.received.setdefault(agent,[]).append(copy.deepcopy(label))

    def register_task_context(self,agent_id,messages):pass
    def memory_owner(self,agent_id):return agent_id
    def active(self,agent_id):return True

    def generate(self,agent_id,messages,generate):
        messages=copy.deepcopy(messages)
        if self.method=="amemguard_full":
            lesson=self.guard.lesson_prompt(self.retrieval_queries.get(agent_id,self.question),self.selected.get(agent_id,[]),agent_id)
            if lesson:
                messages.append({"role":"user","content":lesson})
        output=generate(messages)
        if not isinstance(output,str):raise RuntimeError("Agent model returned non-text output")
        return output

    def defend_communication(self, outputs, state, round_idx, adjacency):
        """Keep the existing adapter and caller-owned state lifetime.

        Core/OpenQA callers reset communication state per task; the transfer
        runner carries it across records. The matched memory sidecar must not
        override either protocol or substitute a memory defense for the guard.
        """
        if not self.uses_legacy_communication_adapter:
            raise ValueError("Method has no legacy communication adapter")
        from .adapter import apply_official_communication_defense
        updated, state, decisions = apply_official_communication_defense(
            self.method, dict(outputs), state, task_id=self.task_id,
            question=self.question, round_idx=round_idx, adj_matrix=adjacency,
            args=self.args)
        pending, self.pending = self.pending, []
        self.save()
        return updated, state, pending + decisions

    def defend(self,outputs,round_idx,adjacency):
        pending,self.pending=self.pending,[]
        self.save()
        return dict(outputs),pending
