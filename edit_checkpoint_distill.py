#!/usr/bin/env python3
"""Utility to override distillation weights stored inside a training checkpoint."""

import argparse
from pathlib import Path
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update distillation settings inside checkpoint_latest.pt"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/dinov3_vitt_distill_disthyper_rir/checkpoint_latest.pt"),
        help="Path to checkpoint_latest.pt",
    )
    parser.add_argument(
        "--distill-feature-weight",
        type=float,
        default=0.0,
        help="New value for train_config['distill_feature_weight']",
    )
    parser.add_argument(
        "--distill-pixel-weight",
        type=float,
        default=0.0,
        help="New value for train_config['distill_pixel_weight']",
    )
    parser.add_argument(
        "--enable-distill-decay",
        type=str,
        choices={"true", "false", "unchanged"},
        default="unchanged",
        help="Override for train_config['enable_distill_decay'] (leave unchanged by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, dict):
        train_config = {}

    train_config["distill_feature_weight"] = float(args.distill_feature_weight)
    train_config["distill_pixel_weight"] = float(args.distill_pixel_weight)
    if args.enable_distill_decay != "unchanged":
        train_config["enable_distill_decay"] = args.enable_distill_decay == "true"

    checkpoint["train_config"] = train_config
    torch.save(checkpoint, ckpt_path)

    print(
        "Updated checkpoint at",
        ckpt_path,
        "with",
        {
            "distill_feature_weight": train_config["distill_feature_weight"],
            "distill_pixel_weight": train_config["distill_pixel_weight"],
            "enable_distill_decay": train_config.get("enable_distill_decay"),
        },
    )


if __name__ == "__main__":
    main()
