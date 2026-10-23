"""Paired comparison for category-aware v5 LLM-judge score files."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = ("semantic_fidelity", "style", "persona", "relevance", "grounding", "overall_100")


def read(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} 存在重复 id")
    return result


def paired(values: list[float], seed: int, samples: int = 5000) -> dict:
    rng = random.Random(seed)
    distribution = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return {
        "n": len(values), "mean_delta": round(mean(values), 2),
        "paired_bootstrap_95_ci": [round(distribution[int(samples * 0.025)], 2), round(distribution[int(samples * 0.975) - 1], 2)],
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def summarize(base: dict[str, dict], finetuned: dict[str, dict], ids: list[str], seed: int) -> dict:
    result = {}
    for metric in METRICS:
        usable = [sample_id for sample_id in ids if metric in base[sample_id] and metric in finetuned[sample_id]]
        if usable:
            result[metric] = paired([float(finetuned[i][metric]) - float(base[i][metric]) for i in usable], seed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    base, finetuned = read(args.base), read(args.finetuned)
    if set(base) != set(finetuned):
        raise ValueError("base 与 finetuned 裁判结果的 id 不一致")
    ids = sorted(base)
    by_category: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        if base[sample_id].get("category") != finetuned[sample_id].get("category"):
            raise ValueError(f"{sample_id} 的类别不一致")
        by_category[str(base[sample_id]["category"])].append(sample_id)
    result = {
        "base": str(args.base), "finetuned": str(args.finetuned), "n": len(ids),
        "metrics": list(METRICS), "overall": summarize(base, finetuned, ids, args.seed),
        "by_category": {category: summarize(base, finetuned, group, args.seed) for category, group in sorted(by_category.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
