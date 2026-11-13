"""YAML configuration files as argparse defaults.

A config file sets default values for the arguments of a script; anything given explicitly on the
command line still wins. Keys use the argument destination names (``max_new_tokens``, not
``--max-new-tokens``); unknown keys are rejected so that typos do not pass silently.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return {str(k).replace("-", "_"): v for k, v in cfg.items()}


def parse_args_with_config(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` after applying the defaults of an optional ``--config FILE``."""
    if not any(a.dest == "config" for a in parser._actions):
        parser.add_argument("--config", help="YAML file with default argument values")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    if known.config:
        cfg = load_config(known.config)
        valid = {a.dest for a in parser._actions}
        unknown = sorted(set(cfg) - valid)
        if unknown:
            parser.error(f"unknown keys in {known.config}: {', '.join(unknown)}")
        parser.set_defaults(**cfg)
    return parser.parse_args(argv)
