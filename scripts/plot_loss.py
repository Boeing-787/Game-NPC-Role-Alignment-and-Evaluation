"""按训练记录绘制 train / eval loss，勿把验证 loss 视为角色评分。"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = config(args.config)
    path = Path(cfg["output_dir"]) / "trainer_log_history.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    train = [(r["step"], r["loss"]) for r in rows if "loss" in r and "step" in r]
    eval_ = [(r["step"], r["eval_loss"]) for r in rows if "eval_loss" in r and "step" in r]
    if not train:
        raise RuntimeError("训练日志缺少 loss 数据")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([p[0] for p in train], [p[1] for p in train], label="Training loss")
    if eval_:
        ax.plot([p[0] for p in eval_], [p[1] for p in eval_], label="Validation loss", marker="o")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()
    fig.tight_layout()
    output = Path(args.out) if args.out else Path(cfg.get("results_dir", "results")) / "loss.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
