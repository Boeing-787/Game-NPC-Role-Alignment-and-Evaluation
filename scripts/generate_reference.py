"""Generate base/LoRA answers for v5's frozen official-reference test set.

This is deliberately separate from ``eval/data``: the input is
``data/v5/test_reference_eval.jsonl``, where every row holds an original
dialogue history and its hidden official reply.  Generation results never
contain that reference reply, so they can be scored blindly by an LLM judge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM = "你正在扮演《原神》中的钟离。请依据角色已有设定与说话方式自然回应；不确定的设定不要编造。"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(rows: list[dict]) -> None:
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("id")
        messages = row.get("messages")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"测试样本 id 缺失或重复: {sample_id!r}")
        seen.add(sample_id)
        if row.get("category") not in {"daily", "lore"}:
            raise ValueError(f"{sample_id} 不是 v5 reference 类别: {row.get('category')!r}")
        roles = [message.get("role") for message in messages] if isinstance(messages, list) else []
        if not messages or roles[0] != "user" or any(
            role != ("user" if i % 2 == 0 else "assistant") for i, role in enumerate(roles)
        ):
            raise ValueError(f"{sample_id} 的 messages 不是合法的 user/assistant 历史")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="基座模型目录或 Hugging Face 模型 ID，例如 Qwen/Qwen3-4B")
    parser.add_argument("--variant", required=True, choices=("base", "finetuned"))
    parser.add_argument("--adapter", help="variant=finetuned 时必填的 LoRA adapter 目录")
    parser.add_argument("--test-file", type=Path, default=Path("data/v5/test_reference_eval.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--system-prompt", default=SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    if args.variant == "finetuned" and not args.adapter:
        parser.error("--variant finetuned 必须提供 --adapter")
    if args.out.exists():
        raise FileExistsError(f"输出已存在，为避免覆盖，请换 --out 路径: {args.out}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("生成需在 GPU 环境运行；当前脚本不会在 CPU pod 上执行正式生成。")
    rows = read_jsonl(args.test_file)
    validate(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    if args.variant == "finetuned":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    outputs: list[dict] = []
    for index, row in enumerate(rows, 1):
        history = [{"role": "system", "content": args.system_prompt}, *row["messages"]]
        tokens = tokenizer.apply_chat_template(
            history, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=eos,
            )
        answer = tokenizer.decode(generated[0, tokens.shape[-1]:], skip_special_tokens=True).strip()
        outputs.append({
            "id": row["id"],
            "category": row["category"],
            "original_category": row.get("original_category", row["category"]),
            "messages": row["messages"],
            "answer": answer,
            "variant": args.variant,
            "model_id": args.model_id,
            "adapter": args.adapter if args.variant == "finetuned" else None,
            "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "enable_thinking": False},
        })
        print(f"[{index}/{len(rows)}] {row['id']} {args.variant}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
