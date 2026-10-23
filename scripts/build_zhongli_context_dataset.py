"""Build auditable Zhongli SFT candidates from game-text-data's Genshin corpus.

The input must be the CHS quest corpus emitted by game-text-data, not the raw
AnimeGameData dump.  It preserves the dialog graph semantics by following only
unique predecessor edges; ambiguous branches are rejected rather than guessed.

This script deliberately writes *candidates* instead of replacing curated.jsonl.
Review the resulting samples and source table before calling prepare_data.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data/game-text-data/corpus/genshin/quests_chs.jsonl"
DEFAULT_OUTPUT = ROOT / "data/zhongli_context.CANDIDATES.jsonl"
DEFAULT_SOURCES = ROOT / "data/zhongli_context.sources.csv"
DEFAULT_AUDIT = ROOT / "data/zhongli_context.audit.json"
GITLAB_BLOB = "https://gitlab.com/Dimbreath/animegamedata2/-/blob"

SPACE = re.compile(r"\s+")
COLOR_TAG = re.compile(r"</?color(?:=[^>]+)?>|</?(?:i|b|size)[^>]*>", re.IGNORECASE)
GENDER_FORM = re.compile(r"\{F#([^{}]*)\}\{M#([^{}]*)\}")
UNRESOLVED_MARKUP = re.compile(r"[{}<>]")


def open_jsonl(path: Path) -> Iterable[dict]:
    """Read plain JSONL, or .zst JSONL when zstandard is installed."""
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    yield json.loads(line)
        return
    zst_path = Path(str(path) + ".zst")
    if not zst_path.exists():
        raise FileNotFoundError(
            f"找不到 {path} 或 {zst_path}。请先在 data/game-text-data 运行 "
            "`uv run gametext genshin all --no-fetch --langs chs --compress none`。"
        )
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"检测到压缩语料 {zst_path}，但当前 Python 缺少 zstandard；"
            "请改用 --compress none 重新导出，或安装 zstandard。"
        ) from exc
    with zst_path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        import io

        with io.TextIOWrapper(reader, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    yield json.loads(line)


def clean_text(value: object) -> str | None:
    """Apply only safe presentation cleanup; reject unresolved game markup."""
    if not isinstance(value, str):
        return None
    text = value.replace("{NICKNAME}", "旅行者")
    text = GENDER_FORM.sub(r"\1", text)
    text = COLOR_TAG.sub("", text)
    text = SPACE.sub(" ", text).strip()
    if not text or UNRESOLVED_MARKUP.search(text):
        return None
    return text


def text_key(text: str) -> str:
    return re.sub(r"[\s，。！？、：:；;「」『』“”\"'…—\-]", "", text).lower()


def int_id(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def role_of(node: dict) -> tuple[str, int | None]:
    role = node.get("role") or {}
    if not isinstance(role, dict):
        return "", None
    return str(role.get("type") or ""), int_id(role.get("id"))


def node_label(node: dict) -> str:
    speaker = clean_text(node.get("speaker"))
    if speaker:
        return speaker
    role_type, _ = role_of(node)
    if role_type == "TALK_ROLE_PLAYER":
        return "旅行者"
    if role_type == "TALK_ROLE_NPC":
        return "某人"
    if role_type == "TALK_ROLE_BLACK_SCREEN":
        return "旁白"
    return "对方"


def make_source_url(commit: object, source: object) -> str:
    ref = str(commit or "main")
    rel = str(source or "")
    if rel:
        return f"{GITLAB_BLOB}/{ref}/BinOutput/Talk/{rel}"
    return f"https://gitlab.com/Dimbreath/animegamedata2/-/tree/{ref}/BinOutput/Talk"


def predecessors(nodes: dict[int, dict]) -> dict[int, list[int]]:
    reverse: dict[int, list[int]] = defaultdict(list)
    for dialog_id, node in nodes.items():
        for next_id in node.get("next") or []:
            target = int_id(next_id)
            if target in nodes:
                reverse[target].append(dialog_id)
    return reverse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CORPUS,
                        help="game-text-data 生成的 quests_chs.jsonl（默认自动尝试 .zst）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sources-output", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--avatar-role-id", type=int, action="append", default=[10000030],
                        help="钟离的 TALK_ROLE_AVATAR id；可重复指定，默认 10000030")
    parser.add_argument("--npc-role-id", type=int, action="append", default=[10232],
                        help="人工核验后的钟离 NPC role.id；默认 10232，可重复指定")
    parser.add_argument("--speaker", action="append", default=["钟离"],
                        help="上游显式 speaker 名称；可重复指定")
    parser.add_argument("--context-turns", type=int, default=2,
                        help="目标台词前保留的唯一前驱轮数，范围 1-3")
    parser.add_argument("--min-target-chars", type=int, default=4)
    parser.add_argument("--max-target-chars", type=int, default=180)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="正数时仅输出前 N 条，便于人工试审；0 为不限")
    args = parser.parse_args()
    if not 1 <= args.context_turns <= 3:
        raise ValueError("--context-turns 必须在 1-3")
    if args.min_target_chars < 1 or args.max_target_chars < args.min_target_chars:
        raise ValueError("目标文本长度参数无效")

    avatar_ids = set(args.avatar_role_id)
    npc_ids = set(args.npc_role_id)
    speaker_names = {text_key(name) for name in args.speaker if clean_text(name)}
    if not speaker_names:
        raise ValueError("至少指定一个有效 --speaker")

    def attribution(node: dict) -> str | None:
        role_type, role_id = role_of(node)
        speaker = clean_text(node.get("speaker"))
        if role_type == "TALK_ROLE_AVATAR" and role_id in avatar_ids:
            return "avatar_role_id"
        if role_type == "TALK_ROLE_NPC" and role_id in npc_ids:
            return "verified_npc_role_id"
        if speaker and text_key(speaker) in speaker_names:
            return "explicit_speaker"
        return None

    rejected: Counter[str] = Counter()
    by_evidence: Counter[str] = Counter()
    records: list[dict] = []
    sources: list[dict] = []
    seen_targets: set[str] = set()
    seen_prompts: set[str] = set()
    quests_seen = 0

    for quest in open_jsonl(args.input):
        quests_seen += 1
        quest_id = int_id(quest.get("id"))
        if quest_id is None:
            rejected["quest_without_id"] += 1
            continue
        raw_nodes = quest.get("dialog") or []
        nodes: dict[int, dict] = {}
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                continue
            dialog_id = int_id(raw.get("dialog_id"))
            if dialog_id is not None and dialog_id not in nodes:
                nodes[dialog_id] = raw
        reverse = predecessors(nodes)
        for dialog_id, node in nodes.items():
            evidence = attribution(node)
            if not evidence:
                continue
            if node.get("attribution") != "explicit":
                rejected["inferred_task_attribution"] += 1
                continue
            answer = clean_text(node.get("text"))
            if answer is None:
                rejected["target_has_unresolved_markup_or_no_text"] += 1
                continue
            if not args.min_target_chars <= len(answer) <= args.max_target_chars:
                rejected["target_length_out_of_range"] += 1
                continue

            chain: list[dict] = []
            cursor = dialog_id
            ambiguous = False
            for _ in range(args.context_turns):
                parent_ids = reverse.get(cursor, [])
                if len(parent_ids) != 1:
                    ambiguous = bool(parent_ids)
                    break
                parent = nodes[parent_ids[0]]
                parent_text = clean_text(parent.get("text"))
                if parent_text is None or attribution(parent):
                    break
                chain.append(parent)
                cursor = parent_ids[0]
            if not chain:
                rejected["no_safe_non_zhongli_context"] += 1
                continue
            if ambiguous:
                rejected["ambiguous_predecessor_branch"] += 1
                continue
            chain.reverse()
            context_lines = []
            invalid_context = False
            for context in chain:
                context_text = clean_text(context.get("text"))
                if context_text is None:
                    invalid_context = True
                    break
                context_lines.append(f"{node_label(context)}：{context_text}")
            if invalid_context:
                rejected["context_has_unresolved_markup_or_no_text"] += 1
                continue
            prompt = "请以钟离的身份接续下面的对话，只输出自然回应。\n\n" + "\n".join(context_lines)
            target_key = text_key(answer)
            prompt_key = text_key(prompt)
            if target_key in seen_targets:
                rejected["duplicate_target_text"] += 1
                continue
            if prompt_key in seen_prompts:
                rejected["duplicate_prompt"] += 1
                continue
            seen_targets.add(target_key)
            seen_prompts.add(prompt_key)

            index = len(records) + 1
            source_id = f"ZL_CTX_SRC_{index:05d}"
            talk_id = int_id(node.get("talk_id")) or 0
            context_ids = [int_id(item.get("dialog_id")) for item in chain]
            record_id = f"ZLCTX{index:05d}"
            records.append({
                "id": record_id,
                "group_id": f"quest-{quest_id}-talk-{talk_id}",
                "source_ids": [source_id],
                "category": "lore",
                "construction_method": "grounded_adaptation",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
            })
            source_path = node.get("source")
            sources.append({
                "source_id": source_id,
                "source_title": f"原神任务对话：任务 {quest_id}，节点 {dialog_id}",
                "source_url": make_source_url((quest.get("upstream") or {}).get("commit"), source_path),
                "source_type": "game_dialogue",
                "original_excerpt": answer.replace("\n", "\\n"),
                "notes": (
                    f"role={role_of(node)}; evidence={evidence}; talk_id={talk_id}; "
                    f"source={source_path}; context_dialog_ids={context_ids}"
                ),
            })
            by_evidence[evidence] += 1
            if args.max_samples and len(records) >= args.max_samples:
                break
        if args.max_samples and len(records) >= args.max_samples:
            break

    for path in (args.output, args.sources_output, args.audit_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.sources_output.open("w", encoding="utf-8", newline="") as handle:
        fields = ["source_id", "source_title", "source_url", "source_type", "original_excerpt", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sources)
    audit = {
        "input": str(args.input),
        "records": len(records),
        "quests_seen": quests_seen,
        "avatar_role_ids": sorted(avatar_ids),
        "npc_role_ids": sorted(npc_ids),
        "speaker_names": sorted(args.speaker),
        "context_turns": args.context_turns,
        "accepted_by_evidence": dict(by_evidence),
        "rejected": dict(rejected),
        "warning": "候选语料必须人工核验角色归属、上下文和设定，再作为正式训练数据使用。",
    }
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} candidate records: {args.output}")
    print(f"Wrote {len(sources)} source rows: {args.sources_output}")
    print(f"Audit: {args.audit_output}")
    print("Accepted evidence:", dict(by_evidence))
    print("Rejected:", dict(rejected))


if __name__ == "__main__":
    main()
