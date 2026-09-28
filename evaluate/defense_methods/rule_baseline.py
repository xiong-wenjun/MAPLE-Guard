"""A deliberately simple, independent provenance/taint plus ACL control.

This is a custom experimental control, not an implementation of an external
paper. Labels are created by trusted runtime ingress hooks, never inferred from
benchmark attack tags, model confidence, success labels, or memory wording.
"""
from __future__ import annotations

class ProvenanceACL:
    CHANNELS = {"agent_output", "peer_message", "user_history", "external", "unknown"}
    PHASES = {"write", "retrieval", "promotion", "transfer"}

    @staticmethod
    def validate(label):
        if not isinstance(label, dict) or label.get("version") != 1:
            raise ValueError("Missing or unsupported operational provenance label")
        if not isinstance(label.get("owner"), int):
            raise ValueError("Operational owner must be an integer")
        for name in ("readers", "sources", "taints"):
            if not isinstance(label.get(name), list):
                raise ValueError("Invalid operational provenance " + name)
        if any(type(x) is not int for x in label["readers"]):
            raise ValueError("ACL readers must be integer identities")
        if any(not isinstance(x, str) for name in ("sources","taints") for x in label[name]):
            raise ValueError("Provenance source/taint values must be strings")
        return label

    def label(self, *, owner, readers, channel, parents=()):
        if channel not in self.CHANNELS:
            raise ValueError("Unknown operational ingress channel: " + str(channel))
        permitted = set(int(x) for x in readers)
        sources = {channel, "agent:" + str(int(owner))}
        taints = {"untrusted_external"} if channel == "external" else ({"unknown_provenance"} if channel == "unknown" else set())
        for parent in parents:
            self.validate(parent)
            permitted.intersection_update(parent["readers"])
            sources.update(parent["sources"])
            taints.update(parent["taints"])
        return {"version":1, "owner":int(owner), "readers":sorted(permitted),
                "sources":sorted(sources), "taints":sorted(taints)}

    def decide(self, label, phase, *, actor, recipients=()):
        self.validate(label)
        if phase not in self.PHASES:
            raise ValueError("Unknown lifecycle phase: " + str(phase))
        details = {"phase":phase, "owner":label["owner"], "readers":list(label["readers"]),
                   "sources":list(label["sources"]), "taints":list(label["taints"])}
        if phase in {"write","promotion"} and int(actor) != label["owner"]:
            return False, {**details, "reason":"acl_owner_required"}
        if int(actor) not in label["readers"]:
            return False, {**details, "reason":"acl_reader_denied"}
        if any(int(x) not in label["readers"] for x in recipients):
            return False, {**details, "reason":"acl_no_permission_widening"}
        if label["taints"]:
            return False, {**details, "reason":"tainted_dependency"}
        return True, {**details, "reason":"operational_rules_allow"}
