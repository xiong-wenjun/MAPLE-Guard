"""Outcome-feedback policies for matched MAPLE placement and learning ablations."""
POLICIES = ("full", "no_q", "none")

def validate_policy(policy):
    if policy not in POLICIES:
        raise ValueError("Unknown outcome feedback policy: " + str(policy))
    return policy

def policy_from_args(args):
    return validate_policy(getattr(args, "outcome_feedback_policy", "full"))

def add_feedback_args(parser, config):
    defense = config.get("defense", {}) if isinstance(config, dict) else {}
    parser.add_argument("--outcome-feedback-policy", choices=POLICIES,
        default=defense.get("outcome_feedback_policy", "full"),
        help="MAPLE ablation: full keeps outcome feedback; no_q freezes reward-driven Q updates only; none removes reference-correctness feedback from state, experience construction and promotion. Evaluation metrics remain enabled.")

def validate_ablation(args):
    policy = policy_from_args(args)
    if policy != "full" and (
        args.method not in ("maple_guard", "maple_guard_retrieval_only")
        or not getattr(args, "strict_comparison", False)
    ):
        raise ValueError("Outcome feedback ablations require a strict MAPLE comparison")
