"""Inspect and execute the existing user checkpoints, without changing weights."""
import importlib.util
from pathlib import Path
import unittest

AVAILABLE = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"))

@unittest.skipUnless(AVAILABLE, "Optional CPU torch/PyG verification environment required")
class CheckpointCompatibilityTests(unittest.TestCase):
    def test_bundled_checkpoints_strictly_load_and_run_their_declared_model(self):
        import torch
        from maple_guard.communication_gnn.model import GSafeguardGAT
        torch.set_num_threads(1)
        root=Path(__file__).resolve().parents[1]
        for name in ("longmemeval.pth","appworld.pth"):
            with self.subTest(checkpoint=name):
                checkpoint=torch.load(root/"communication_gnn"/name,map_location="cpu",weights_only=True)
                self.assertEqual(checkpoint["model_class"],"GSafeguardGAT")
                model=GSafeguardGAT(**checkpoint["model_kwargs"])
                model.load_state_dict(checkpoint["model_state_dict"],strict=True)
                model.eval()
                edges=torch.tensor([[0,1,1,2],[1,0,2,1]])
                with torch.no_grad():
                    logits=model(torch.zeros(3,384),edges,torch.zeros(4,2,384))
                self.assertEqual(tuple(logits.shape),(3,1))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertFalse(any(key.startswith("branch_heads_inf.") for key in checkpoint["model_state_dict"]))

if __name__=="__main__": unittest.main()
