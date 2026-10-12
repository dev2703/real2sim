"""Train the recovery specialist MLP on expert corrections (plan §4 M6 step 5, D-005).

    uv run python -m real2sim.recovery.train_mlp --data <corrections.csv|.parquet> \
        --config configs/recovery.yaml --out outputs/recovery_mlp

Held-out = whole correction episodes; early stopping on the held-out Huber loss. Normalisation
statistics come from the training rows only and are stored in the checkpoint. The CLI refuses
any provenance outside the allowed set (see ``dataset.py``). Writes ``recovery_mlp.pt`` and
``train_log.json`` under ``--out``; never touches ACT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from real2sim.config import load_yaml
from real2sim.recovery.dataset import CorrectionDataset
from real2sim.recovery.mlp import CONST_TOL, STD_FLOOR, MLPSpec, RecoveryMLP


def _huber(delta: float) -> nn.Module:
    return nn.HuberLoss(delta=delta)


@torch.no_grad()
def evaluate(model: RecoveryMLP, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> float:
    model.eval()
    pred = model.forward_normalised(x)
    return float(loss_fn(pred, model.normalise_y(y)))


def train(
    config: dict[str, Any],
    dataset: CorrectionDataset,
    out_dir: str | Path | None = None,
    max_epochs: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train and return a log dict (train/heldout loss per epoch, best epoch, stopped_early)."""
    m = config["mlp"]
    seed = int(m.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec = MLPSpec.from_config(config)
    train_mask, held_mask, split = dataset.split(float(m.get("heldout_fraction", 0.2)), seed)

    x = torch.as_tensor(dataset.x, dtype=torch.float32, device=device)
    y = torch.as_tensor(dataset.y, dtype=torch.float32, device=device)
    xt, yt = x[torch.as_tensor(train_mask)], y[torch.as_tensor(train_mask)]
    xh, yh = x[torch.as_tensor(held_mask)], y[torch.as_tensor(held_mask)]

    model = RecoveryMLP(spec, dataset.feature_names).to(device)
    norm = m.get("normalisation", {})
    model.set_normalisation(
        xt.mean(0),
        xt.std(0, unbiased=False),
        yt.mean(0),
        yt.std(0, unbiased=False),
        std_floor=float(norm.get("std_floor", STD_FLOOR)),
        const_tol=float(norm.get("const_tol", CONST_TOL)),
    )

    if str(m.get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("D-005 fixes the optimiser to AdamW")
    opt = torch.optim.AdamW(model.parameters(), lr=float(m["lr"]), weight_decay=float(m["weight_decay"]))
    loss_fn = _huber(float(m["huber_delta"])) if m.get("loss", "huber") == "huber" else nn.L1Loss()
    batch = int(m["batch_size"])
    es = m.get("early_stopping", {})
    patience = int(es.get("patience", 10))
    min_delta = float(es.get("min_delta", 1e-5))
    epochs = int(max_epochs if max_epochs is not None else m.get("max_epochs", 200))

    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_epoch, best_state, bad = float("inf"), -1, None, 0
    log: dict[str, Any] = {
        "train_loss": [],
        "heldout_loss": [],
        "n_train_rows": int(train_mask.sum()),
        "n_heldout_rows": int(held_mask.sum()),
        "train_episode_ids": [str(e) for e in split.train_ids],
        "heldout_episode_ids": [str(e) for e in split.heldout_ids],
        "feature_names": dataset.feature_names,
        "spec": spec.__dict__,
        "constant_columns": model.constant_columns(),  # unscaled in normalisation (see mlp.py)
    }
    n = len(xt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model.forward_normalised(xt[idx]), model.normalise_y(yt[idx]))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        train_loss = total / n
        held_loss = evaluate(model, xh, yh, loss_fn)
        log["train_loss"].append(train_loss)
        log["heldout_loss"].append(held_loss)
        if held_loss < best - min_delta:
            best, best_epoch, bad = held_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    log["epochs_run"] = len(log["train_loss"])
    log["best_epoch"] = best_epoch
    log["best_heldout_loss"] = best
    log["stopped_early"] = log["epochs_run"] < epochs
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save(out / "recovery_mlp.pt", extra={"best_epoch": best_epoch, "best_heldout_loss": best})
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        log["checkpoint"] = str(out / "recovery_mlp.pt")
    log["model"] = model
    return log


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m real2sim.recovery.train_mlp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="correction table (.csv or .parquet)")
    p.add_argument("--config", default="configs/recovery.yaml")
    p.add_argument("--out", default="outputs/recovery_mlp")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    config = load_yaml(a.config)
    ds = CorrectionDataset.from_file(a.data, tuple(config["dataset"]["allowed_provenance"]))
    log = train(config, ds, a.out, a.max_epochs, a.device)
    print(
        json.dumps(
            {k: log[k] for k in ("epochs_run", "best_epoch", "best_heldout_loss", "stopped_early", "n_train_rows", "n_heldout_rows", "checkpoint")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
