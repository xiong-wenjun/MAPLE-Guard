"""Train a G-Safeguard-style communication GNN from converted trace data."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .model import GSafeguardGAT


class TraceGraphDataset(torch.utils.data.Dataset):
    def __init__(self, records: Sequence[Dict]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Data:
        item = self.records[index]
        x = torch.tensor(np.asarray(item["features"]), dtype=torch.float32)
        edge_index = torch.tensor(np.asarray(item["edge_index"]), dtype=torch.long)
        edge_attr = torch.tensor(np.asarray(item["edge_attr"]), dtype=torch.float32)
        labels = torch.tensor(np.asarray(item["labels"])[:, 0], dtype=torch.float32).view(-1, 1)
        data = Data(x=x, y=labels, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = int(x.shape[0])
        return data


def load_records(path: str) -> List[Dict]:
    with open(path, "rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"No records found in {path}")
    return records


def make_loaders(records: List[Dict], batch_size: int, seed: int, val_ratio: float):
    indices = list(range(len(records)))
    if len(indices) < 2:
        train_indices = val_indices = indices
    else:
        train_indices, val_indices = train_test_split(indices, test_size=val_ratio, random_state=seed, shuffle=True)
    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]
    train_loader = DataLoader(TraceGraphDataset(train_records), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TraceGraphDataset(val_records), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_records, val_records


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    positives = 0
    predicted_positive = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.y)
            total_loss += float(loss.item())
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total += int(batch.y.numel())
            correct += int((preds == batch.y).sum().item())
            positives += int(batch.y.sum().item())
            predicted_positive += int(preds.sum().item())
    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": correct / max(total, 1),
        "positives": positives,
        "predicted_positive": predicted_positive,
        "total": total,
    }


def train(args: argparse.Namespace) -> Dict:
    records = load_records(args.dataset)
    train_loader, val_loader, train_records, val_records = make_loaders(records, args.batch_size, args.seed, args.val_ratio)
    first = records[0]
    in_channels = int(np.asarray(first["features"]).shape[-1])
    edge_dim = int(np.asarray(first["edge_attr"]).shape[-1])

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    model = GSafeguardGAT(
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        out_channels=1,
        heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        edge_dim=edge_dim,
        pool=args.pool,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-5)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pth"
    latest_path = save_dir / "latest_model_path.txt"
    history = []
    best_score = -1.0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        scheduler.step()
        train_eval = evaluate(model, train_loader, criterion, device)
        val_eval = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "train_accuracy": train_eval["accuracy"],
            "val_loss": val_eval["loss"],
            "val_accuracy": val_eval["accuracy"],
            "val_predicted_positive": val_eval["predicted_positive"],
            "val_positives": val_eval["positives"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if val_eval["accuracy"] > best_score:
            best_score = val_eval["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_class": "GSafeguardGAT",
                    "model_kwargs": {
                        "in_channels": in_channels,
                        "hidden_channels": args.hidden_dim,
                        "out_channels": 1,
                        "heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "edge_dim": edge_dim,
                        "pool": args.pool,
                    },
                    "dataset": args.dataset,
                    "method_scope": "communication_only",
                },
                best_path,
            )
            latest_path.write_text(str(best_path), encoding="utf-8")

    summary = {
        "dataset": args.dataset,
        "save_dir": str(save_dir),
        "best_model": str(best_path),
        "export_path": str(args.export_path or ""),
        "best_val_accuracy": best_score,
        "train_graphs": len(train_records),
        "val_graphs": len(val_records),
        "epochs": args.epochs,
        "history": history,
        "method_scope": "communication_only",
    }
    if args.export_path:
        export_path = Path(args.export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, export_path)
        summary["export_path"] = str(export_path)
        print(f"Exported best checkpoint to: {export_path}")
    with open(save_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a G-Safeguard communication GNN.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--export-path", default="", help="Optional stable path to copy the best checkpoint after training.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pool", choices=["mean", "last"], default="mean")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
