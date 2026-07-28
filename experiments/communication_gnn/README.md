# Communication GNN Workspace

This folder contains launchers for G-Safeguard-style communication-only GNN
baselines. Generated artifacts are kept under one run root:

```text
result_maple_guard/communication_gnn/<run_id>/
  traces/<bench>/
  datasets/<bench>/gsafeguard/
  checkpoints/<bench>/gsafeguard/
  logs/
  manifests/
```

The converter uses only per-round agent outputs, adjacency matrices, and
attacker ids. It intentionally excludes memory taint/provenance/retrieval and
defense decisions so the baseline remains communication-only.

