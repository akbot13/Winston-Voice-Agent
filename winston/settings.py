"""Loads config.json, then applies your own values from config.local.json (not uploaded)."""
import json
import os


def load(base):
    with open(os.path.join(base, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    local = os.path.join(base, "config.local.json")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg
