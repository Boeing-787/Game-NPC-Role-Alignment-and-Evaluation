"""Category-aware, identity-grounded LLM judging for v5's reference set.

The judge is explicitly told that the target character is Zhongli from
Genshin Impact. Each case supplies the corresponding official reply as an
evidence-backed style/context anchor. It remains a reference, not a required
semantic target for open-ended Daily role-play. Boundary is evaluated in the
separate frozen 28-item human-blind suite because this reference set contains
no official Boundary replies.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


DAILY_PROMPT_VERSION = "v5_identity_grounded_reference_v3"
LORE_PROMPT_VERSION = "v5_identity_semantic_lore_v4"
JUDGE_SYSTEM = """你是严格、独立的中文角色对话评测员。当前评测的目标角色是游戏《原神》中的钟离。
钟离是璃月的往生堂客卿，表达通常沉稳、克制、书面而自然，重视契约、历史、秩序、责任与对人的体谅；
但不得把这份概括机械地套用到每一句话。评估匿名候选回答本身，不要推测候选来自哪个模型，
不因回答更长而加分，也不要求复述原作台词。题目会给出同一语境下的原作回答，作为可核验的角色风格、
语境与事实参照；请区分“风格相符”与“逐字或逐义复现”。只输出 JSON 对象。"""
METRICS_BY_CATEGORY = {
    "daily": ("style", "persona", "relevance", "grounding"),
    "lore": ("semantic_fidelity", "style", "persona", "grounding"),
}
WEIGHTS = {
    "daily": {"style": 0.40, "persona": 0.25, "relevance": 0.25, "grounding": 0.10},
    "lore": {"semantic_fidelity": 0.50, "grounding": 0.25, "persona": 0.15, "style": 0.10},
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_history(messages: list[dict]) -> str:
    return "\n".join(("钟离" if item["role"] == "assistant" else "对方") + "：" + item["content"] for item in messages)


def judge_prompt(case: dict, answer: str) -> str:
    category = case["category"]
    history = render_history(case["messages"])
    if category == "daily":
        return f"""请评估《原神》角色钟离在开放式日常互动中的回应质量。

对话上下文：
{history}

同一语境下的原作钟离回答（仅作风格、身份和回应方式参照，不是唯一答案）：
{case['reference_answer']}

匿名候选回答：
{answer}

这是开放式角色扮演题，没有唯一的语义标准答案。原作回答帮助你识别钟离在该语境的语气、人物关系和回应尺度，
但候选可以使用完全不同的内容与措辞；只要候选自然、符合钟离身份、贴合上下文且不编造，就应得到高分。
不得因为候选没有提及原作中的具体物品、事件或句式而扣分，也不得因其与原作表面相似却答非所问而给高分。

分别给出 0–4 整数分数：
1. style：钟离式克制、书面感、自然措辞；不应依赖夸张舞台动作或套话。
2. persona：身份、价值取向和与对方的互动方式是否符合钟离。
3. relevance：是否直接、完整、自然地回应当前话题。
4. grounding：是否避免无依据的新世界观事实、虚构经历或错误断言。

rationale 不超过 60 个中文字符。仅输出 JSON：
{{"style": 0, "persona": 0, "relevance": 0, "grounding": 0, "rationale": ""}}"""
    if category == "lore":
        return f"""请评估《原神》角色钟离对世界观或角色设定问题的回答质量。

对话上下文：
{history}

同一语境下的原作钟离回答（用于核验角色风格、核心事实或价值取向，不要求复述）：
{case['reference_answer']}

匿名候选回答：
{answer}

允许候选使用不同措辞、补充与原作不冲突的合理说明，但不得偏离原作的核心事实、结论、态度或交际意图。
若编造或与参考的关键事实、结论冲突，semantic_fidelity 与 grounding 均不得高于 1。

分别给出 0–4 整数分数：
1. semantic_fidelity：核心事实、结论、态度与交际意图是否和原作回答一致。4=实质等价且无冲突；3=主旨一致、仅轻微遗漏或改写；2=只保留部分要点；1=大多无关；0=明显矛盾或编造。
2. style：钟离式克制、书面感、自然措辞。
3. persona：身份、价值取向和人物关系是否符合钟离。
4. grounding：核心世界观事实、结论或价值判断是否与官方参考相容，且无无依据编造。

rationale 不超过 60 个中文字符。仅输出 JSON：
{{"semantic_fidelity": 0, "style": 0, "persona": 0, "grounding": 0, "rationale": ""}}"""
    raise ValueError(f"不支持的 reference 类别: {category}")


def call_api(*, base_url: str, api_key: str, model: str, prompt: str, timeout: int, retries: int) -> tuple[dict, dict]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 500,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url, data=raw,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return json.loads(body["choices"][0]["message"]["content"]), body.get("usage") or {}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"裁判 API 调用或 JSON 解析失败（重试 {retries} 次后）：{last_error}")


def validate(score: dict, category: str) -> dict:
    result: dict[str, object] = {}
    for metric in METRICS_BY_CATEGORY[category]:
        value = score.get(metric)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
            raise ValueError(f"裁判 {metric} 必须是 0–4 整数: {value!r}")
        result[metric] = value
    rationale = score.get("rationale", "")
    result["rationale"] = rationale if isinstance(rationale, str) else str(rationale)
    weighted = sum(WEIGHTS[category][metric] * float(result[metric]) for metric in METRICS_BY_CATEGORY[category])
    result["overall_100"] = round(weighted / 4 * 100, 2)
    return result


def bootstrap(values: list[float], seed: int, samples: int = 5000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    distribution = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [round(distribution[int(samples * 0.025)], 2), round(distribution[int(samples * 0.975) - 1], 2)]


def aggregate(rows: list[dict], metric: str, seed: int) -> dict:
    values = [float(row[metric]) for row in rows if metric in row]
    return {"n": len(values), "mean": round(mean(values), 2) if values else None, "bootstrap_95_ci": bootstrap(values, seed)}


def build_summary(rows: list[dict], metadata: dict, seed: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    metrics = ("semantic_fidelity", "style", "persona", "relevance", "grounding", "overall_100")
    return {
        "metadata": metadata,
        "overall": {metric: aggregate(rows, metric, seed) for metric in metrics},
        "by_category": {
            category: {metric: aggregate(group, metric, seed) for metric in metrics}
            for category, group in sorted(grouped.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, default=Path("data/v5/test_reference_eval.jsonl"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replace-categories", nargs="+", choices=("daily", "lore"),
                     help="只重新裁判指定类别，并在已有输出中原位替换该类别记录。")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"缺少 API key 环境变量：{args.api_key_env}")
    cases = {row["id"]: row for row in read_jsonl(args.test_file)}
    predictions = {row["id"]: row for row in read_jsonl(args.predictions)}
    if set(cases) != set(predictions):
        raise ValueError("predictions 与 test-file 的样本 id 不一致")
    if any(row.get("category") not in WEIGHTS for row in cases.values()):
        raise ValueError("reference test 只能包含 daily 或 lore")

    replace_categories = set(args.replace_categories or [])
    if replace_categories and not args.output.exists():
        raise FileNotFoundError("--replace-categories 只能用于已有评分输出")
    existing = read_jsonl(args.output) if (args.resume or replace_categories) and args.output.exists() else []
    if replace_categories:
        existing = [row for row in existing if cases.get(row.get("id"), {}).get("category") not in replace_categories]
    completed = {row.get("id") for row in existing}
    if not completed <= set(cases):
        raise ValueError("已有评分文件含不属于当前 test-file 的 id，不能 --resume")
    output_rows = list(existing)
    for index, sample_id in enumerate(sorted(cases), 1):
        if sample_id in completed:
            continue
        answer = predictions[sample_id].get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{sample_id} 缺少非空 answer")
        case = cases[sample_id]
        raw_score, usage = call_api(
            base_url=args.base_url, api_key=api_key, model=args.model,
            prompt=judge_prompt(case, answer), timeout=args.timeout, retries=args.retries,
        )
        score = validate(raw_score, case["category"])
        output_rows.append({
            "id": sample_id, "category": case["category"],
            "original_category": case.get("original_category", case["category"]),
            "candidate_answer": answer, "judge_model": args.model,
            "judge_prompt_version": DAILY_PROMPT_VERSION if case["category"] == "daily" else LORE_PROMPT_VERSION,
            **score, "usage": usage,
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
        print(f"[{index}/{len(cases)}] {sample_id}: {score['overall_100']}", flush=True)

    output_rows.sort(key=lambda row: row["id"])
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    metadata = {
        "test_file": str(args.test_file), "predictions": str(args.predictions), "judge_model": args.model,
        "base_url": args.base_url, "temperature": 0, "thinking": "disabled",
        "prompt_versions_by_category": {"daily": DAILY_PROMPT_VERSION, "lore": LORE_PROMPT_VERSION},
        "weights_by_category": WEIGHTS, "n": len(output_rows), "seed": args.seed,
        "judge_identity_policy": "The judge is explicitly told that the target character is Zhongli from Genshin Impact and receives a compact role card.",
        "daily_reference_policy": "Official replies are provided as style, identity, and context anchors; they are not unique semantic targets.",
        "lore_reference_policy": "Official replies are semantic, factual/value, and style anchors; paraphrases are allowed but core facts, conclusions, attitudes, and communicative intent must remain aligned.",
        "replaced_categories": sorted(replace_categories),
    }
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(build_summary(output_rows, metadata, args.seed), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
