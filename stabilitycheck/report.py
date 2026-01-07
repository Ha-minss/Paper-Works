from __future__ import annotations
import os, json
import pandas as pd

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)


# Backward-compatible alias
def ensure_dir(outdir: str):
    ensure_outdir(outdir)

def save_csv(outdir: str, name: str, df: pd.DataFrame):
    ensure_outdir(outdir)
    df.to_csv(os.path.join(outdir, name), index=False)


def save_csv_file(path: str, df: pd.DataFrame):
    """Save a DataFrame to an explicit file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def save_json(outdir: str, name: str, obj):
    ensure_outdir(outdir)
    with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def save_json_file(path: str, obj):
    """Save JSON to an explicit file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
