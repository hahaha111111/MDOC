import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from data import create_loaders
from model import LatentDiscriminator, MDOC


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_trainable(module, value):
    for parameter in module.parameters():
        parameter.requires_grad = value


def compactness_loss(features):
    center = features.mean(dim=0, keepdim=True)
    return torch.norm(features - center, dim=1).mean()


def compute_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {
            "auc": float("nan"),
            "aupr_ap": float("nan"),
            "aupr_np": float("nan"),
            "acc": float("nan"),
            "sen": float("nan"),
            "spe": float("nan"),
            "threshold": float("nan"),
        }
    auc = roc_auc_score(labels, scores)
    aupr_ap = average_precision_score(labels, scores)
    aupr_np = average_precision_score(1 - labels, -scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_index = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best_index])
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    return {
        "auc": float(auc),
        "aupr_ap": float(aupr_ap),
        "aupr_np": float(aupr_np),
        "acc": float(acc),
        "sen": float(sen),
        "spe": float(spe),
        "threshold": threshold,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels = []
    scores = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        recon, _, _, _, _ = model(images, noise_std=0.0)
        error = F.mse_loss(recon, images, reduction="none").flatten(1).mean(dim=1)
        labels.extend(targets.numpy().tolist())
        scores.extend(error.cpu().numpy().tolist())
    return labels, scores, compute_metrics(labels, scores)


def train_one_epoch(model, discriminator, loader, optimizer_model, optimizer_disc, args, device):
    model.train()
    discriminator.train()
    total_loss = 0.0
    total_re = 0.0
    total_dc = 0.0
    total_u = 0.0
    total_d = 0.0
    batches = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)
        real = torch.empty(batch_size, args.latent_dim, device=device).uniform_(-1, 1)
        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)

        set_trainable(discriminator, True)
        with torch.no_grad():
            _, f_u, _, _, _ = model(images, noise_std=args.noise_std)
        loss_disc = 0.5 * (
            F.binary_cross_entropy(discriminator(real), ones)
            + F.binary_cross_entropy(discriminator(f_u.detach()), zeros)
        )
        optimizer_disc.zero_grad(set_to_none=True)
        loss_disc.backward()
        optimizer_disc.step()

        set_trainable(discriminator, False)
        recon, f_u, f_d, _, _ = model(images, noise_std=args.noise_std)
        loss_re = F.mse_loss(recon, images)
        loss_dc = compactness_loss(f_d)
        loss_u = F.binary_cross_entropy(discriminator(f_u), ones)
        loss = args.lambda_re * loss_re + args.lambda_dc * loss_dc + args.lambda_u * loss_u
        optimizer_model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_model.step()
        set_trainable(discriminator, True)

        total_loss += float(loss.item())
        total_re += float(loss_re.item())
        total_dc += float(loss_dc.item())
        total_u += float(loss_u.item())
        total_d += float(loss_disc.item())
        batches += 1

    return {
        "train_loss": total_loss / max(batches, 1),
        "loss_re": total_re / max(batches, 1),
        "loss_dc": total_dc / max(batches, 1),
        "loss_u": total_u / max(batches, 1),
        "loss_disc": total_d / max(batches, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--output-dir", default="runs/mdoc")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3])
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-dc", type=float, default=1.0)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--lambda-re", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.4472136)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="split")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_loaders(
        data_root=args.data_root,
        metadata=args.metadata,
        target_label=args.target_label,
        image_size=args.image_size,
        channels=args.channels,
        batch_size=args.batch_size,
        workers=args.workers,
        path_column=args.path_column,
        label_column=args.label_column,
        split_column=args.split_column,
    )
    model = MDOC(in_channels=args.channels, latent_dim=args.latent_dim).to(device)
    discriminator = LatentDiscriminator(latent_dim=args.latent_dim).to(device)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_auc = -float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            discriminator=discriminator,
            loader=train_loader,
            optimizer_model=optimizer_model,
            optimizer_disc=optimizer_disc,
            args=args,
            device=device,
        )
        labels, scores, metrics = evaluate(model, test_loader, device)
        record = {"epoch": epoch, **train_stats, **metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        current_auc = metrics["auc"]
        if not np.isnan(current_auc) and current_auc >= best_auc:
            best_auc = current_auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                },
                output_dir / "best.pt",
            )
            pd.DataFrame({"label": labels, "score": scores}).to_csv(output_dir / "scores.csv", index=False)

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(history[-1], file, indent=2)


if __name__ == "__main__":
    main()
