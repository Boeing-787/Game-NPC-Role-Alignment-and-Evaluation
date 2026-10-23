"""Create the v5 ablation subset with all manual Boundary rows removed.

The frozen official reference test is deliberately not copied or modified: it
contains only Daily/Lore official-answer pairs and must stay identical between
the main experiment and this ablation.  Train and validation rows preserve all
metadata, so every retained sample remains traceable to its original source.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/v5"
TARGET = ROOT / "data/v5_no_boundary"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def digest(rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    split: dict[str, list[dict]] = {}
    audit: dict[str, object] = {"source": "data/v5", "rule": "drop rows whose category is boundary"}
    for name in ("train", "val"):
        original = read_jsonl(SOURCE / f"{name}.jsonl")
        retained = [row for row in original if row.get("category") != "boundary"]
        dropped = [row for row in original if row.get("category") == "boundary"]
        if any(row.get("category") == "boundary" for row in retained):
            raise RuntimeError(f"{name} still contains Boundary rows")
        if not dropped:
            raise RuntimeError(f"{name} contains no Boundary rows; refusing an uninformative ablation")
        write_jsonl(TARGET / f"{name}.jsonl", retained)
        split[name] = retained
        audit[name] = {
            "before": len(original),
            "after": len(retained),
            "removed_boundary": len(dropped),
            "category_counts": dict(Counter(str(row.get("category")) for row in retained)),
            "retained_content_sha256": digest(retained),
        }

    # Verify that reference evaluation remains an identical, official-only test.
    reference = read_jsonl(SOURCE / "test_reference_eval.jsonl")
    if any(row.get("category") == "boundary" for row in reference):
        raise RuntimeError("reference test unexpectedly contains Boundary rows")
    audit["reference_test"] = {
        "path": "data/v5/test_reference_eval.jsonl (shared, unchanged)",
        "count": len(reference),
        "content_sha256": hashlib.sha256((SOURCE / "test_reference_eval.jsonl").read_bytes()).hexdigest(),
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "subset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (TARGET / "README.md").write_text(
        "# v5 no-Boundary ablation subset\n\n"
        "This directory is derived from v5 by removing every `category=boundary` row from train and validation. "
        "All remaining official Daily/Lore rows retain their original metadata and text. "
        "The frozen official reference test remains shared at `data/v5/test_reference_eval.jsonl`.\n",
        encoding="utf-8",
    )
    print("Wrote", TARGET / "train.jsonl", "rows:", len(split["train"]))
    print("Wrote", TARGET / "val.jsonl", "rows:", len(split["val"]))
    print("Audit:", TARGET / "subset_audit.json")


if __name__ == "__main__":
    main()
