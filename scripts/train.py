"""TRL SFTTrainer + PEFT BF16 LoRA，train/val 的 metadata 不输入模型。"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import torch
from datasets import Dataset
# PEFT 提供 LoRA 配置；实际将 LoRA adapter 注入基座模型的动作由
# 下方 SFTTrainer(..., peft_config=lora) 触发。
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from common import config, init_seed, read_jsonl


def to_prompt_completion(rows: list[dict], system_prompt: str, tokenizer) -> Dataset:
    """Render Qwen3 as non-thinking plain-text prompt-completion, train only answer.

    Train prompt is exactly the same rendering used by generate.py. This avoids
    the Qwen3 default thinking-mode prefix and masks the system/user instructions.
    """
    converted = []
    for row in rows:
        # 训练文件保留来源、类别等元数据，但此处只读取真正的对话 messages。
        messages = row["messages"]
        roles = [m.get("role") for m in messages]
        if len(messages) < 2 or roles[0] != "user" or roles[-1] != "assistant" or any(
            roles[index] != ("user" if index % 2 == 0 else "assistant") for index in range(len(roles))
        ):
            raise ValueError(f"样本 {row.get('id')} 必须是 user/assistant 交替且以 assistant 结束")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            raise ValueError(f"样本 {row.get('id')} 含空消息")
        # 最后一个 assistant 回合是本条监督答案；此前的多轮对话均作为上下文。
        # system prompt 在训练时注入，保证与推理阶段使用相同的人设指令。
        prefix = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, *messages[:-1]],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        # Qwen3 uses <|im_end|> as assistant turn end. Verify at model-load time.
        completion = messages[-1]["content"].strip() + "<|im_end|>"
        converted.append({"prompt": prefix, "completion": completion})
    return Dataset.from_list(converted)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--max-steps", type=int, default=-1, help="正整数为冒烟测试步数，默认完整训练")
    args = p.parse_args()
    cfg = config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("没有找到 CUDA GPU；请先运行 nvidia-smi 与 torch.cuda.is_available()")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 PyTorch/CUDA 未报告 BF16 支持；检查驱动及 PyTorch 安装")
    init_seed(int(cfg["seed"]))
    print("CUDA GPU:", torch.cuda.get_device_name(0))
    # 读取已完成清洗与划分的 JSONL；不会在训练脚本内重新抽样或切分数据。
    rows_train = read_jsonl(cfg["train_file"])
    rows_val = read_jsonl(cfg["val_file"])
    print("训练/验证条数:", len(rows_train), len(rows_val))
    if len(rows_train) < 2 or len(rows_val) < 1:
        raise ValueError("训练/验证数据不足")
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    model_id = cfg["model_name"]
    model_path = Path(model_id)
    if model_id.startswith("/") and not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"本地模型路径无效或不完整: {model_path}；需要至少包含 config.json"
        )
    revision = cfg.get("model_revision") or None
    # 使用 Qwen3 的 chat template 将对话渲染为模型实际接收的 token 序列。
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token != "<|im_end|>":
        raise RuntimeError(f"Qwen3 EOS token 不符合预期: {tokenizer.eos_token!r}")
    # 每个 Dataset 样本只包含 prompt 与 completion 两列，供 TRL 构造 labels。
    train_ds = to_prompt_completion(rows_train, cfg["system_prompt"], tokenizer)
    val_ds = to_prompt_completion(rows_val, cfg["system_prompt"], tokenizer)
    # 加载冻结的 Qwen 基座参数。BF16 降低显存占用；梯度检查点在 TrainingArguments 中启用。
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    resolved_commit = getattr(model.config, "_commit_hash", None)
    model.config.use_cache = False
    # ---------- LoRA 的核心配置 ----------
    # r/alpha/dropout 与 YAML 对应；target_modules 指定 Qwen Transformer 中
    # 哪些线性层旁挂低秩可训练矩阵。原始基座权重不更新，只有这些 adapter 参数更新。
    lora = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["lora_target_modules"]),
        bias="none", task_type="CAUSAL_LM",
    )
    # TRL 的监督微调参数。这里的 batch_size=1 与梯度累积共同决定有效 batch size。
    training_args = SFTConfig(
        output_dir=str(outdir),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        bf16=True, fp16=False, tf32=True,
        gradient_checkpointing=True,
        max_length=int(cfg["max_length"]),
        # prompt/completion 格式下：system、user 和历史上下文都置为 labels=-100，
        # 仅最后一轮钟离回答参与 loss，避免模型学习复述提示词。
        completion_only_loss=True,
        assistant_only_loss=False,
        packing=False,
        eval_strategy="epoch" if args.max_steps == -1 else "no",
        save_strategy="epoch" if args.max_steps == -1 else "no",
        load_best_model_at_end=args.max_steps == -1,
        metric_for_best_model="eval_loss" if args.max_steps == -1 else None,
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=1,
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        seed=int(cfg["seed"]),
        data_seed=int(cfg["seed"]),
        max_steps=args.max_steps,
    )
    # 传入 peft_config 后，SFTTrainer 内部调用 PEFT：
    # 1) 按 target_modules 注入 LoRA adapter；2) 冻结基座权重；3) 仅优化 LoRA 参数。
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    # TRL 的 completion_only_loss 应遮盖 prompt token（labels=-100）。
    sample = next(iter(trainer.get_train_dataloader()))
    targets = int((sample["labels"] != -100).sum().item())
    if targets < 1:
        raise RuntimeError("有效 loss tokens 为零！请检查 prompt-completion 格式及截断")
    if targets == sample["labels"].numel():
        raise RuntimeError("没有遮蔽 prompt 或 padding tokens！请检查 completion_only_loss")
    print("检查 loss mask: 有效 completion tokens:", targets, "批次 tokens:", sample["labels"].numel())
    # 正式训练期间，每个 epoch 进行验证和保存；结束时自动恢复 eval_loss 最低的 checkpoint。
    trainer.train()
    if args.max_steps != -1:
        print("冒烟测试已结束；未保存正式 Adapter。请重新以无 --max-steps 的命令完整训练。")
        return
    # 保存的是 LoRA adapter 与 tokenizer，不是完整基座模型；推理时需同时加载基座模型和该 adapter。
    adapter_dir = outdir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = trainer.evaluate()
    record = {
        "base_model": model_id,
        "model_revision_requested": revision,
        "resolved_commit_from_model_config": resolved_commit,
        "lora_config": lora.to_dict(),
        "data_sha256": {key: hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest() for key in ("train_file", "val_file")},
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "train_size": len(rows_train), "val_size": len(rows_val),
        "train_args": training_args.to_dict(),
        "eval_metrics": metrics,
        "command": "python scripts/train.py --config " + args.config,
    }
    (outdir / "experiment_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with (outdir / "trainer_log_history.json").open("w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print("Adapter:", adapter_dir)
    print("Final eval:", metrics)


if __name__ == "__main__":
    main()
