"""Build the v5 Zhongli corpus from traceable official dialogue and voice text.

This is intentionally independent of every v3/v4 curated/train/validation
file.  Its inputs are:

* ``game-text-data/corpus/genshin/quests_chs.jsonl`` -- a local, reproducible
  projection of ``animegamedata2``; every retained row points back to its
  upstream commit, Talk file and dialogue nodes; and
* ``Genshin-Impact-NPC-Audio-Texts-Dataset/lines.json`` -- a title-to-text
  collection of Zhongli's official Chinese profile voice lines; and
* the separately maintained manual Boundary authoring file.  It is never
  represented as an in-game quote and is recorded as ``human_authored``.

The corpus has only the three task dimensions required by the experiment:
Daily, Lore/Values (stored as ``lore`` and ``values`` for analysis), and
Boundary.  It writes a group-isolated source-reference test set in addition to
train/validation splits.  Manual Boundary examples stay out of that reference
test because they do not have an official in-game answer.

Run from the project root:
    python scripts/build_zhongli_v5.py

Review ``data/v5/curated.REVIEW.jsonl`` and ``data/v5/build_audit.json``
before starting a formal experiment.  The formal JSONL files deliberately
contain no reviewer-only fields.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from build_zhongli_context_dataset import clean_text, int_id, make_source_url, role_of


ROOT = Path(__file__).resolve().parents[1]
FORMAL_FIELDS = ("id", "group_id", "source_ids", "category", "construction_method", "messages")
SPACE = re.compile(r"\s+")

# These terms do not create content: they merely document how an already
# quoted official reply is allocated to the experiment's three dimensions.
VALUE_TERMS = (
    "契约", "约定", "交易", "公平", "责任", "信义", "规矩", "磨损", "人治", "自由", "变迁", "选择", "守护",
)
LORE_TERMS = (
    "璃月", "岩王帝君", "摩拉克斯", "魔神", "仙人", "夜叉", "归终", "若陀", "七星", "往生堂", "千岩军",
    "岩神", "神之心", "神明", "魔神战争", "层岩", "玉京台", "奥藏", "琉璃", "海灯", "请仙", "送仙", "神之眼",
)
AUDIO_SOURCE_URL = "https://github.com/chenllliang/Genshin-Impact-NPC-Audio-Texts-Dataset/blob/master/lines.json"


def load_npc_name_map(npc_config: Path, text_map: Path) -> dict[int, str]:
    """Resolve explicit TALK_ROLE_NPC ids to a local CHS display name.

    The extracted quest corpus usually has no populated ``speaker`` field.
    Its ``role.id`` is nevertheless retained and can be joined to
    ``NpcExcelConfigData.id``.  ``TextMap_MediumCHS`` contains the matching
    localized name for the config's ``nameTextMapHash``.  We intentionally do
    not fall back to a generic label such as "某人": an unresolved context
    speaker is excluded instead of being silently anonymised.
    """
    npc_rows = json.loads(npc_config.read_text(encoding="utf-8"))
    text_rows = json.loads(text_map.read_text(encoding="utf-8"))
    if not isinstance(npc_rows, list) or not isinstance(text_rows, dict):
        raise ValueError("NPC 配置或中文 TextMap 的 JSON 结构不符合预期")
    resolved: dict[int, str] = {}
    for row in npc_rows:
        if not isinstance(row, dict):
            continue
        npc_id = int_id(row.get("id"))
        name_hash = row.get("nameTextMapHash")
        if npc_id is None or name_hash is None:
            continue
        name = clean_text(text_rows.get(str(name_hash)))
        if name:
            resolved[npc_id] = name
    if not resolved:
        raise ValueError("未能从 NPC 配置与 TextMap 解析任何中文角色名")
    return resolved


def resolved_node_label(node: dict, npc_names: dict[int, str]) -> tuple[str | None, str]:
    """Return a traceable scene label, or ``None`` when it cannot be resolved."""
    speaker = clean_text(node.get("speaker"))
    if speaker:
        return speaker, "node_speaker"
    role_type, role_id = role_of(node)
    if role_type == "TALK_ROLE_PLAYER":
        return "旅行者", "role_type_player"
    if role_type == "TALK_ROLE_BLACK_SCREEN":
        return "旁白", "role_type_black_screen"
    if role_type == "TALK_ROLE_NPC":
        name = npc_names.get(role_id) if role_id is not None else None
        return (name, "npc_excel_textmap") if name else (None, "unresolved_npc_role_id")
    return None, f"unresolved_role_type:{role_type or 'empty'}"

# These are explicit title-level annotations, rather than a keyword heuristic.
# Combat, death and ascension clips have no conversational training role and
# are intentionally left out of all three sets.
AUDIO_DAILY_TITLES = {
    "初次见面", "早上好", "中午好", "晚上好", "晚安", "闲聊·旅程", "闲聊·游览", "闲聊·怀旧",
    "雨过天晴", "刮大风了", "感兴趣的见闻", "关于胡桃", "关于达达利亚·沉着", "关于达达利亚·酒桌",
    "想要了解钟离·其一", "想要了解钟离·其四", "想要了解钟离·其五", "钟离的爱好",
    "喜欢的食物", "讨厌的食物", "生日",
}
AUDIO_LORE_TITLES = {
    "闲聊·璃月", "关于魈", "关于温迪", "关于凝光", "想要了解钟离·其二", "想要了解钟离·其三",
}
AUDIO_VALUES_TITLES = {
    "关于钟离自己", "关于我们·黄金", "关于我们·契约", "关于我们·公平", "关于我们·困惑",
    "关于「神之眼」", "有什么想要分享", "关于甘雨", "关于刻晴", "关于烟绯", "钟离的烦恼",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def open_jsonl(path: Path) -> Iterable[dict]:
    """Read plain JSONL or the standard zstd corpus without changing it."""
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    compressed = Path(str(path) + ".zst")
    if not compressed.exists():
        raise FileNotFoundError(f"找不到 {path} 或 {compressed}")
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("读取压缩语料需要 zstandard；或先以 --compress none 导出 CHS 语料。") from exc
    import io

    with compressed.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        with io.TextIOWrapper(reader, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def normalise(text: str) -> str:
    return re.sub(r"[\s，。！？、：:；;「」『』“”\"'…—\-]", "", text).lower()


def clean_audio_title(title: str) -> str:
    return title.strip().rstrip("·…").strip()


def stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def is_zhongli(node: dict, npc_role_id: int) -> bool:
    role_type, role_id = role_of(node)
    return node.get("attribution") == "explicit" and role_type == "TALK_ROLE_NPC" and role_id == npc_role_id


def predecessor_map(nodes: dict[int, dict]) -> dict[int, list[int]]:
    reverse: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes.items():
        for next_id in node.get("next") or []:
            target = int_id(next_id)
            if target in nodes:
                reverse[target].append(node_id)
    return reverse


def classify_official(target: str, history: str) -> tuple[str, str]:
    """Allocate an official quote to a task dimension using auditable terms."""
    if any(term in target for term in VALUE_TERMS):
        return "values", "target_value_term"
    evidence = target + " " + history
    if any(term in evidence for term in LORE_TERMS):
        return "lore", "target_or_context_lore_term"
    return "daily", "no_lore_or_value_term"


def audio_candidates(path: Path) -> tuple[list[dict], Counter[str]]:
    """Load Zhongli's official profile voice lines with a fixed title taxonomy."""
    data = json.loads(path.read_text(encoding="utf-8"))
    zhongli = data.get("钟离") if isinstance(data, dict) else None
    if not isinstance(zhongli, dict):
        raise ValueError(f"{path} 中未找到钟离的标题-文本字典")
    candidates: list[dict] = []
    audit: Counter[str] = Counter()
    for raw_title, raw_text in zhongli.items():
        if not isinstance(raw_title, str):
            audit["non_string_title"] += 1
            continue
        title = clean_audio_title(raw_title)
        if title.startswith(("元素战技", "元素爆发", "死亡", "突破")):
            audit["excluded_non_conversational_voice"] += 1
            continue
        if title in AUDIO_DAILY_TITLES:
            category = "daily"
        elif title in AUDIO_LORE_TITLES:
            category = "lore"
        elif title in AUDIO_VALUES_TITLES:
            category = "values"
        else:
            audit[f"unmapped_title:{title}"] += 1
            continue
        text = clean_text(raw_text)
        if text is None:
            audit["unresolved_or_empty_voice_text"] += 1
            continue
        candidates.append({
            "title": title,
            "text": text,
            "category": category,
            "classification_reason": "explicit_voice_title_taxonomy",
        })
    return candidates, audit


def official_candidates(
    corpus: Path,
    npc_role_id: int,
    npc_names: dict[int, str],
    *,
    max_history_nodes: int,
    min_target_chars: int,
    max_target_chars: int,
) -> tuple[list[dict], Counter[str]]:
    """Extract only explicit, unbranched predecessor paths from raw quest data."""
    candidates: list[dict] = []
    audit: Counter[str] = Counter()
    for quest in open_jsonl(corpus):
        quest_id = int_id(quest.get("id"))
        if quest_id is None:
            audit["quest_without_id"] += 1
            continue
        nodes = {
            int_id(node.get("dialog_id")): node
            for node in quest.get("dialog") or []
            if isinstance(node, dict) and int_id(node.get("dialog_id")) is not None
        }
        reverse = predecessor_map(nodes)
        for dialog_id, node in nodes.items():
            if not is_zhongli(node, npc_role_id):
                continue
            target = clean_text(node.get("text"))
            if target is None:
                audit["target_unresolved_or_empty"] += 1
                continue
            if not min_target_chars <= len(target) <= max_target_chars:
                audit["target_length_out_of_range"] += 1
                continue

            # Trace a local prefix only while every edge is unambiguous.  This
            # does not invent a path through player choices.
            chain: list[dict] = []
            cursor = dialog_id
            for _ in range(max_history_nodes):
                parents = reverse.get(cursor, [])
                if len(parents) != 1:
                    if parents:
                        audit["ambiguous_predecessor"] += 1
                    break
                parent = nodes[parents[0]]
                text = clean_text(parent.get("text"))
                if text is None:
                    audit["history_unresolved_or_empty"] += 1
                    break
                chain.append(parent)
                cursor = parents[0]
            if not chain:
                audit["no_unambiguous_history"] += 1
                continue
            chain.reverse()
            if is_zhongli(chain[-1], npc_role_id):
                # The answer would follow Zhongli himself; this is not a
                # meaningful player/scene cue for another Zhongli completion.
                audit["history_ends_in_zhongli"] += 1
                continue

            messages: list[dict] = []
            speaker_trace: list[dict] = []
            valid = True
            for previous in chain:
                text = clean_text(previous.get("text"))
                if text is None:
                    valid = False
                    break
                if is_zhongli(previous, npc_role_id):
                    messages.append({"role": "assistant", "content": text})
                    speaker_trace.append({"label": "钟离", "resolution": "explicit_target_role"})
                else:
                    label, resolution = resolved_node_label(previous, npc_names)
                    if label is None:
                        audit[resolution] += 1
                        valid = False
                        break
                    _, role_id = role_of(previous)
                    messages.append({"role": "user", "content": f"{label}：{text}"})
                    speaker_trace.append({"label": label, "role_id": role_id, "resolution": resolution})
            if not valid or not messages or messages[0]["role"] != "user":
                audit["invalid_role_sequence"] += 1
                continue
            # The prompt-completion trainer requires alternating roles.  A
            # block of two non-Zhongli speakers is represented as one scene
            # cue, preserving the original text without inventing a reply.
            collapsed: list[dict] = []
            for message in messages:
                if collapsed and collapsed[-1]["role"] == message["role"] == "user":
                    collapsed[-1]["content"] += "\n" + message["content"]
                else:
                    collapsed.append(message)
            if collapsed[-1]["role"] != "user":
                audit["collapsed_history_ends_in_assistant"] += 1
                continue
            collapsed.append({"role": "assistant", "content": target})
            history_text = "\n".join(message["content"] for message in collapsed[:-1])
            # Allocate categories by the original textual evidence, not by a
            # resolved name (for example, a named NPC should not turn an
            # otherwise ordinary exchange into Lore merely by being present).
            raw_history_text = "\n".join(clean_text(previous.get("text")) or "" for previous in chain)
            category, reason = classify_official(target, raw_history_text)
            upstream = quest.get("upstream") or {}
            candidates.append({
                "quest_id": quest_id,
                "quest_name": clean_text(quest.get("name")) or f"任务 {quest_id}",
                "dialog_id": dialog_id,
                "talk_id": int_id(node.get("talk_id")) or 0,
                "source": node.get("source"),
                "commit": upstream.get("commit"),
                "category": category,
                "classification_reason": reason,
                "messages": collapsed,
                "speaker_trace": speaker_trace,
                "target_key": normalise(target),
                "prompt_key": normalise(history_text),
            })
    return candidates, audit


def sample_official(
    candidates: list[dict],
    wanted: dict[str, int],
    *,
    seed: int,
    max_per_quest_category: int,
) -> tuple[list[dict], Counter[str]]:
    """Deduplicate first, then deterministically sample each official category."""
    selected: list[dict] = []
    audit: Counter[str] = Counter()
    seen_target: set[str] = set()
    seen_prompt: set[str] = set()
    per_group_category: Counter[tuple[int, str]] = Counter()
    for category in ("daily", "lore", "values"):
        pool = [row for row in candidates if row["category"] == category]
        pool.sort(key=lambda row: stable_key(seed, category, f"{row['quest_id']}:{row['dialog_id']}"))
        for row in pool:
            if len([x for x in selected if x["category"] == category]) >= wanted[category]:
                break
            key = (row["quest_id"], category)
            if per_group_category[key] >= max_per_quest_category:
                audit[f"{category}_quest_cap"] += 1
                continue
            if row["target_key"] in seen_target:
                audit["duplicate_target"] += 1
                continue
            if row["prompt_key"] in seen_prompt:
                audit["duplicate_prompt"] += 1
                continue
            selected.append(row)
            seen_target.add(row["target_key"])
            seen_prompt.add(row["prompt_key"])
            per_group_category[key] += 1
        got = sum(row["category"] == category for row in selected)
        if got < wanted[category]:
            audit[f"{category}_shortfall"] = wanted[category] - got
    return selected, audit


def select_boundary(path: Path, *, seed: int, per_subtype: int) -> tuple[list[dict], Counter[str]]:
    """Use the independent human-authoring ledger, not any v3/v4 corpus."""
    raw = read_jsonl(path)
    groups: dict[str, list[dict]] = defaultdict(list)
    audit: Counter[str] = Counter()
    for row in raw:
        if row.get("category") != "boundary" or row.get("construction_method") != "authored_scenario":
            audit["invalid_boundary_schema"] += 1
            continue
        messages = row.get("messages") or []
        roles = [message.get("role") for message in messages]
        if len(messages) < 2 or roles[0] != "user" or roles[-1] != "assistant" or any(
            role != ("user" if index % 2 == 0 else "assistant") for index, role in enumerate(roles)
        ):
            audit["invalid_boundary_messages"] += 1
            continue
        subtype = str(row.get("boundary_subtype") or "unspecified")
        groups[subtype].append(row)
    selected: list[dict] = []
    for subtype in sorted(groups):
        rows = sorted(groups[subtype], key=lambda row: stable_key(seed, subtype, str(row.get("id"))))
        selected.extend(rows[:per_subtype])
        audit[f"boundary_available_{subtype}"] = len(rows)
    return selected, audit


def split_raw_groups(rows: list[dict], ratios: tuple[float, float, float], seed: int) -> dict[str, list[dict]]:
    """Assign whole quests while keeping category totals close to 70/15/15."""
    names = ("train", "val", "test_reference")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    totals = Counter(row["category"] for row in rows)
    targets = {name: {category: totals[category] * ratio for category in totals} for name, ratio in zip(names, ratios)}
    group_items = [(group_id, group_rows, Counter(row["category"] for row in group_rows)) for group_id, group_rows in groups.items()]

    def score(placement: dict[str, str]) -> float:
        counts = {name: Counter() for name in names}
        sizes = Counter()
        for group_id, group_rows, category_counts in group_items:
            destination = placement[group_id]
            counts[destination].update(category_counts)
            sizes[destination] += len(group_rows)
        category_error = sum(
            ((counts[name][category] - targets[name][category]) / max(1.0, targets[name][category])) ** 2
            for name in names for category in totals
        )
        total_error = sum(
            ((sizes[name] - len(rows) * ratios[names.index(name)]) / max(1.0, len(rows) * ratios[names.index(name)])) ** 2
            for name in names
        )
        # Categories are the key requirement, but the total split must not
        # drift merely because a quest happens to contain many turns.
        return category_error + 0.5 * total_error

    # Hash partitions give deterministic, independent starting points.  We
    # retain the best and then make only objective-improving whole-quest moves.
    cumulative = (ratios[0], ratios[0] + ratios[1])
    best: dict[str, str] | None = None
    best_score = math.inf
    for trial in range(96):
        placement: dict[str, str] = {}
        for group_id, _, _ in group_items:
            fraction = int(stable_key(seed + trial, "split-init", group_id)[:12], 16) / float(16 ** 12)
            placement[group_id] = "train" if fraction < cumulative[0] else "val" if fraction < cumulative[1] else "test_reference"
        if all(any(destination == name for destination in placement.values()) for name in names):
            candidate_score = score(placement)
            if candidate_score < best_score:
                best, best_score = placement, candidate_score
    if best is None:
        raise RuntimeError("无法为 source-group split 生成非空训练/验证/测试集")

    # A local search is needed because one quest may contain several selected
    # replies.  It never splits that quest across datasets.
    for _ in range(len(group_items) * 12):
        current_score = score(best)
        move: tuple[str, str] | None = None
        move_score = current_score
        split_group_counts = Counter(best.values())
        for group_id, _, _ in group_items:
            source = best[group_id]
            if split_group_counts[source] <= 1:
                continue
            for destination in names:
                if destination == source:
                    continue
                candidate = dict(best)
                candidate[group_id] = destination
                candidate_score = score(candidate)
                if candidate_score + 1e-12 < move_score:
                    move, move_score = (group_id, destination), candidate_score
        if move is None:
            break
        best[move[0]] = move[1]

    assigned: dict[str, list[dict]] = {name: [] for name in names}
    for group_id, group_rows, _ in group_items:
        assigned[best[group_id]].extend(group_rows)
    return assigned


def formal(row: dict) -> dict:
    return {field: row[field] for field in FORMAL_FIELDS}


def to_reference_eval(row: dict) -> dict:
    """Preserve the original dialogue history for reference-answer generation."""
    category = "lore" if row["category"] in {"lore", "values"} else row["category"]
    return {
        "id": row["id"],
        "category": category,
        "messages": row["messages"][:-1],
        "reference_answer": row["messages"][-1]["content"],
        "source_ids": row["source_ids"],
        "original_category": row["category"],
    }


def archive_existing_v5(output: Path, archive: Path) -> list[str]:
    """Copy an existing v5 output once before a deliberate rebuild."""
    expected = (
        "curated.REVIEW.jsonl", "audio.REVIEW.jsonl", "curated.jsonl", "train.jsonl", "val.jsonl",
        "test_reference.jsonl", "test_reference_eval.jsonl", "sources.csv", "build_audit.json",
    )
    present = [name for name in expected if (output / name).exists()]
    if not present:
        return []
    if archive.exists():
        raise FileExistsError(
            f"检测到既有 v5 数据，且归档目录已存在：{archive}。为避免覆盖归档，请显式指定新的 --archive-dir。"
        )
    archive.mkdir(parents=True)
    for name in present:
        shutil.copy2(output / name, archive / name)
    return present


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/game-text-data/corpus/genshin/quests_chs.jsonl")
    parser.add_argument("--audio-input", type=Path,
                        default=ROOT / "data/Genshin-Impact-NPC-Audio-Texts-Dataset/lines.json")
    parser.add_argument("--npc-config", type=Path,
                        default=ROOT / "data/animegamedata2/ExcelBinOutput/NpcExcelConfigData.json")
    parser.add_argument("--text-map", type=Path,
                        default=ROOT / "data/animegamedata2/TextMap/TextMap_MediumCHS.json")
    parser.add_argument("--boundary-input", type=Path, default=ROOT / "data/boundary/boundary_v4.REVIEW.jsonl",
                        help="独立维护的人工 Boundary 标注；不是 v3/v4 训练集。")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/v5")
    parser.add_argument("--npc-role-id", type=int, default=10232)
    parser.add_argument("--max-history-nodes", type=int, default=3)
    parser.add_argument("--min-target-chars", type=int, default=12)
    parser.add_argument("--max-target-chars", type=int, default=240)
    parser.add_argument("--daily", type=int, default=60,
                        help="任务对话 Daily 目标数；另会合并固定标题分类的角色语音 Daily。")
    parser.add_argument("--lore", type=int, default=60)
    parser.add_argument("--values", type=int, default=24)
    parser.add_argument("--boundary-per-subtype", type=int, default=3)
    parser.add_argument("--max-per-quest-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--archive-dir", type=Path,
                        default=ROOT / "data/archive/v5_before_named_speaker_rebuild",
                        help="若 output-dir 已有 v5 文件，先复制到此目录；目录已存在时终止以避免覆盖归档。")
    args = parser.parse_args()
    if args.max_history_nodes < 1 or args.min_target_chars < 1 or args.max_target_chars < args.min_target_chars:
        raise ValueError("上下文或回复长度参数无效")
    if min(args.daily, args.lore, args.values, args.boundary_per_subtype) < 1:
        raise ValueError("每类目标样本数必须为正数")

    npc_names = load_npc_name_map(args.npc_config, args.text_map)
    candidates, extraction_audit = official_candidates(
        args.corpus, args.npc_role_id, npc_names, max_history_nodes=args.max_history_nodes,
        min_target_chars=args.min_target_chars, max_target_chars=args.max_target_chars,
    )
    official, sampling_audit = sample_official(
        candidates, {"daily": args.daily, "lore": args.lore, "values": args.values},
        seed=args.seed, max_per_quest_category=args.max_per_quest_category,
    )
    voice, voice_audit = audio_candidates(args.audio_input)
    selected_boundary, boundary_audit = select_boundary(
        args.boundary_input, seed=args.seed, per_subtype=args.boundary_per_subtype,
    )

    review: list[dict] = []
    sources: list[dict] = []
    raw_rows: list[dict] = []
    audio_review: list[dict] = []
    for index, row in enumerate(official, 1):
        record_id = f"ZLV5_RAW_{index:04d}"
        source_id = f"ZLV5_RAW_SRC_{index:04d}"
        formal_row = {
            "id": record_id,
            "group_id": f"quest-{row['quest_id']}",
            "source_ids": [source_id],
            "category": row["category"],
            "construction_method": "grounded_named_multiturn" if len(row["messages"]) > 2 else "grounded_named_scene",
            "messages": row["messages"],
        }
        review.append({
            **formal_row,
            "origin": "animegamedata2_explicit_unbranched_dialogue",
            "review_status": "source_traceable_pending_semantic_audit",
            "source_verified": True,
            "quality_tags": ["official_dialogue", "explicit_npc_role_id", "unbranched_predecessor", "named_context_speakers"],
            "classification_reason": row["classification_reason"],
            "notes_for_reviewer": (
                f"quest_id={row['quest_id']}; dialog_id={row['dialog_id']}; talk_id={row['talk_id']}; "
                f"speaker_resolution={json.dumps(row['speaker_trace'], ensure_ascii=False)}; "
                "核验分类、角色名解析与上下文的语义连续性。"
            ),
        })
        sources.append({
            "source_id": source_id,
            "source_title": f"原神任务对话：{row['quest_name']}（任务 {row['quest_id']}，节点 {row['dialog_id']}）",
            "source_url": make_source_url(row["commit"], row["source"]),
            "source_type": "animegamedata2_task_dialog",
            "original_excerpt": "\n".join(message["content"] for message in row["messages"]),
            "notes": (
                f"upstream_commit={row['commit']}; role=TALK_ROLE_NPC:{args.npc_role_id}; "
                f"talk_id={row['talk_id']}; dialog_id={row['dialog_id']}; "
                f"classification={row['category']}:{row['classification_reason']}; "
                f"speaker_resolution={json.dumps(row['speaker_trace'], ensure_ascii=False)}"
            ),
        })
        raw_rows.append(formal_row)

    # Profile voice is not a quest conversation, so the user side is a minimal
    # template of the official voice title.  The response itself is retained
    # verbatim (except safe markup cleanup), and the title is recorded in both
    # REVIEW and sources.csv.
    raw_target_keys = {normalise(row["messages"][-1]["content"]) for row in raw_rows}
    for index, row in enumerate(voice, 1):
        if normalise(row["text"]) in raw_target_keys:
            voice_audit["duplicate_with_task_dialogue"] += 1
            continue
        record_id = f"ZLV5_AUDIO_{index:03d}"
        source_id = f"ZLV5_AUDIO_SRC_{index:03d}"
        formal_row = {
            "id": record_id,
            "group_id": f"voice-title-{normalise(row['title'])}",
            "source_ids": [source_id],
            "category": row["category"],
            "construction_method": "grounded_adaptation",
            "messages": [
                {"role": "user", "content": f"钟离，在“{row['title']}”的情境下，你会如何回应？"},
                {"role": "assistant", "content": row["text"]},
            ],
        }
        audio_row = {
            **formal_row,
            "origin": "official_character_voice_title_to_text",
            "review_status": "source_traceable_pending_semantic_audit",
            "source_verified": True,
            "quality_tags": ["official_voice_text", "title_grounded_prompt", row["category"]],
            "classification_reason": row["classification_reason"],
            "notes_for_reviewer": (
                f"语音标题={row['title']}；提示仅由标题模板化生成，assistant 为官方语音文本。"
                "核验该标题的 Daily/Lore/Values 归类。"
            ),
        }
        review.append(audio_row)
        audio_review.append(audio_row)
        sources.append({
            "source_id": source_id,
            "source_title": f"原神钟离角色语音：{row['title']}",
            "source_url": AUDIO_SOURCE_URL,
            "source_type": "genshin_character_voice_text",
            "original_excerpt": row["text"],
            "notes": (
                f"dataset_file=lines.json; character=钟离; voice_title={row['title']}; "
                f"classification={row['category']}:explicit_voice_title_taxonomy"
            ),
        })
        raw_rows.append(formal_row)

    boundary_rows: list[dict] = []
    for index, row in enumerate(selected_boundary, 1):
        record_id = f"ZLV5_BOUNDARY_{index:03d}"
        source_id = f"ZLV5_BOUNDARY_SRC_{index:03d}"
        subtype = str(row.get("boundary_subtype") or "unspecified")
        formal_row = {
            "id": record_id,
            "group_id": f"v5-boundary-{subtype}-{index:02d}",
            "source_ids": [source_id],
            "category": "boundary",
            "construction_method": "authored_scenario",
            "messages": row["messages"],
        }
        review.append({
            **formal_row,
            "origin": "human_authored_boundary_ledger",
            "review_status": "retained_manual_boundary",
            "source_verified": True,
            "quality_tags": ["not_official_quote", "manual_boundary", subtype],
            "notes_for_reviewer": "人工撰写的越界场景，不得描述为游戏原文；核验拒答、纠错和角色口吻。",
        })
        sources.append({
            "source_id": source_id,
            "source_title": f"钟离 Boundary 人工标注：{subtype}",
            "source_url": "manual://review-required",
            "source_type": "human_authored",
            "original_excerpt": "\n".join(message["content"] for message in row["messages"]),
            "notes": f"authoring_input={args.boundary_input.name}; original_id={row.get('id')}; boundary_subtype={subtype}; not_an_official_quote",
        })
        boundary_rows.append(formal_row)

    raw_split = split_raw_groups(raw_rows, (0.70, 0.15, 0.15), args.seed)
    # Boundary is a training objective and validation safety check, but has no
    # official reference answer.  It therefore never enters test_reference.
    boundary_rows.sort(key=lambda row: stable_key(args.seed, "boundary-val", row["id"]))
    n_boundary_val = max(1, round(len(boundary_rows) * 0.15))
    split = {
        "train": raw_split["train"] + boundary_rows[n_boundary_val:],
        "val": raw_split["val"] + boundary_rows[:n_boundary_val],
        "test_reference": raw_split["test_reference"],
    }
    for name in split:
        split[name].sort(key=lambda row: row["id"])

    output = args.output_dir
    archived_files = archive_existing_v5(output, args.archive_dir)
    output.mkdir(parents=True, exist_ok=True)
    curated = [formal(row) for row in review]
    curated.sort(key=lambda row: row["id"])
    write_jsonl(output / "curated.REVIEW.jsonl", review)
    write_jsonl(output / "audio.REVIEW.jsonl", audio_review)
    write_jsonl(output / "curated.jsonl", curated)
    write_jsonl(output / "train.jsonl", split["train"])
    write_jsonl(output / "val.jsonl", split["val"])
    write_jsonl(output / "test_reference.jsonl", split["test_reference"])
    write_jsonl(output / "test_reference_eval.jsonl", [to_reference_eval(row) for row in split["test_reference"]])
    with (output / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["source_id", "source_title", "source_url", "source_type", "original_excerpt", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sources)

    audit = {
        "builder": "scripts/build_zhongli_v5.py",
        "principle": {
            "official_data": "Explicit Zhongli NPC role 10232 nodes from AnimeGameData2-derived CHS quests plus official Zhongli profile voice text; no v3/v4 curated data is read.",
            "speaker_resolution": "Every non-Zhongli quest context speaker must resolve from node.speaker, PLAYER/BLACK_SCREEN role type, or TALK_ROLE_NPC role.id -> NpcExcelConfigData.id -> TextMap_MediumCHS. Unresolved speakers are excluded; no generic '某人' label is emitted.",
            "dimensions": ["daily", "lore", "values", "boundary"],
            "boundary": "Manual Boundary scenarios are retained as human-authored training/validation data and excluded from reference-answer test.",
            "split": "Official examples are split by quest group into 70/15/15 train/val/test_reference; a quest never crosses splits.",
        },
        "inputs": {"corpus": str(args.corpus), "audio_input": str(args.audio_input), "npc_config": str(args.npc_config), "text_map": str(args.text_map), "boundary_input": str(args.boundary_input)},
        "speaker_name_map_size": len(npc_names),
        "archived_existing_files": archived_files,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "candidate_categories": dict(Counter(row["category"] for row in candidates)),
        "task_dialogue_selected": dict(Counter(row["category"] for row in official)),
        "voice_selected": dict(Counter(row["category"] for row in voice)),
        "official_selected": dict(Counter(row["category"] for row in raw_rows)),
        "boundary_selected": dict(Counter("boundary:" + tag for row in review if row["category"] == "boundary" for tag in row.get("quality_tags", []) if tag not in {"not_official_quote", "manual_boundary"})),
        "splits": {name: {"rows": len(rows), "categories": dict(Counter(row["category"] for row in rows)), "groups": len({row["group_id"] for row in rows})} for name, rows in split.items()},
        "group_overlap": {
            "train_val": sorted({row["group_id"] for row in split["train"]} & {row["group_id"] for row in split["val"]}),
            "train_test_reference": sorted({row["group_id"] for row in split["train"]} & {row["group_id"] for row in split["test_reference"]}),
            "val_test_reference": sorted({row["group_id"] for row in split["val"]} & {row["group_id"] for row in split["test_reference"]}),
        },
        "extraction_rejections": dict(extraction_audit),
        "sampling_rejections": dict(sampling_audit),
        "voice_audit": dict(voice_audit),
        "boundary_audit": dict(boundary_audit),
        "warning": "test_reference contains only official quote pairs. Do not use it to tune epochs, learning rate, decoding, prompts, or data selection.",
    }
    (output / "build_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # Hard failures make accidental leakage visible immediately.
    if any(audit["group_overlap"].values()):
        raise RuntimeError(f"group split leakage: {audit['group_overlap']}")
    anonymous_contexts = [
        row["id"] for row in curated
        if any("某人：" in message["content"] for message in row["messages"][:-1])
    ]
    if anonymous_contexts:
        raise RuntimeError(f"具名上下文重建失败，仍含某人标签: {anonymous_contexts[:10]}")
    print("v5 construction complete")
    for name, rows in split.items():
        print(f"{name}: {len(rows)} rows, {dict(Counter(row['category'] for row in rows))}")
    print(f"review: {output / 'curated.REVIEW.jsonl'}")
    print(f"audit:  {output / 'build_audit.json'}")


if __name__ == "__main__":
    main()
