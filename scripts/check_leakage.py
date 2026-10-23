"""用字符相似度提示可能的 train/test 语义近似，仅为人工审查的辅助。"""
import argparse
from difflib import SequenceMatcher
from common import config, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--threshold", type=float, default=0.62)
    args = ap.parse_args()
    cfg = config(args.config)
    train = read_jsonl(cfg["train_file"])
    val = read_jsonl(cfg["val_file"])
    tests = read_jsonl(cfg["test_file"])
    pairs = []
    for t in tests:
        # Current frozen eval rows use ``user``; retain ``question`` support
        # for earlier project templates.
        q = str(t.get("user") or t.get("question") or "").strip()
        if not q:
            raise ValueError(f"测试样本 {t.get('id')} 缺少 user/question 文本")
        for row in train + val:
            prompt = row["messages"][0]["content"].strip()
            sim = SequenceMatcher(None, q, prompt).ratio()
            if sim >= args.threshold:
                pairs.append((sim, t["id"], row["id"], q, prompt))
    if not pairs:
        print("未发现高于阈值的 train/val vs test 字符近似问题。仍需人工排查语义同义和世界观泄漏！")
    else:
        for sim, tid, rid, q, prompt in sorted(pairs, reverse=True)[:100]:
            print(f"SIM={sim:.2f} test={tid} train_or_val={rid}\n  T={q}\n  X={prompt}")
        print("请人工核验上述潜在泄漏，发现实质重复要重设计或移出相关数据。")


if __name__ == "__main__":
    main()
