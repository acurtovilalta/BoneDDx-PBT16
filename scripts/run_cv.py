#!/usr/bin/env python3
"""
Run 7-fold cross-validation using a YAML config.

Usage:
    python run_cv.py --config configs/cv_7_3heads.yaml
    uv run run_cv.py --config configs/config.yaml
"""
import argparse
import sys
import yaml
from train import cross_validate


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow either a flat mapping or a top-level "CFG" block
    if isinstance(cfg, dict) and "CFG" in cfg and isinstance(cfg["CFG"], dict):
        cfg = cfg["CFG"]
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping/dict.")

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run CV with YAML config.")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/config.yaml",
        help="Path to YAML config (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    try:
        cfg = load_cfg(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to load config '{args.config}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Using config: {args.config}")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")

    cross_validate(cfg)


if __name__ == "__main__":
    main()
