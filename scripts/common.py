"""Small shared helpers; use from repository root."""
from __future__ import annotations
import json
import random
from pathlib import Path
import yaml


def config(path: str = "configs/experiment.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"缺少配置文件: {p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("配置文件必须是 YAML mapping")
    return cfg


def read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"缺少 {path}，请先构建真实语料数据，模板不能直接作为作业结果")
    rows = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n} JSON 格式错误: {exc}") from exc
        if not isinstance(rec, dict):
            raise ValueError(f"{path}:{n} 必须是一条 JSON object")
        rows.append(rec)
    if not rows:
        raise ValueError(f"{path} 为空")
    return rows


def dump_jsonl(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def init_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
