# 钟离角色对话 LoRA 微调

## 环境

正式训练与生成需要 CUDA GPU。

~~~bash
conda create -n <环境名> python=3.12 -y
conda activate <环境名>

# 先安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖
pip install -r requirements.txt
~~~

将训练配置中的 `model_name` 改为本机的 Qwen3 模型目录。

## 训练

仓库包含基础语料组与边界增强语料组的配置。选择其中一个配置启动训练：

~~~bash
cd <项目根目录>
conda activate <环境名>

CONFIG_FILE=<训练配置文件>
python scripts/train.py --config "$CONFIG_FILE"
~~~

训练完成后，LoRA adapter 位于该配置 `output_dir` 指定目录的 `adapter` 子目录。

## 官方参考集生成与自动评测

~~~bash
MODEL_DIR=<本地基座模型目录>
ADAPTER_DIR=<训练产生的 LoRA_adapter 目录>
RESULT_DIR=<评测结果目录>
REFERENCE_TEST_FILE=<官方参考测试集 JSONL>

# 生成基座模型与微调模型回答
python scripts/generate_reference.py \
  --model-id "$MODEL_DIR" --variant base --out "$RESULT_DIR/base.jsonl"

python scripts/generate_reference.py \
  --model-id "$MODEL_DIR" --variant finetuned --adapter "$ADAPTER_DIR" \
  --out "$RESULT_DIR/finetuned.jsonl"

# DeepSeek 裁判评测：密钥仅通过环境变量提供
export DEEPSEEK_API_KEY='<API_KEY>'

python scripts/score_v5_task_quality.py \
  --test-file "$REFERENCE_TEST_FILE" \
  --predictions "$RESULT_DIR/base.jsonl" \
  --output "$RESULT_DIR/base.judged.jsonl" \
  --summary-output "$RESULT_DIR/base.summary.json"

python scripts/score_v5_task_quality.py \
  --test-file "$REFERENCE_TEST_FILE" \
  --predictions "$RESULT_DIR/finetuned.jsonl" \
  --output "$RESULT_DIR/finetuned.judged.jsonl" \
  --summary-output "$RESULT_DIR/finetuned.summary.json"

python scripts/compare_v5_task_scores.py \
  --base "$RESULT_DIR/base.judged.jsonl" \
  --finetuned "$RESULT_DIR/finetuned.judged.jsonl" \
  --output "$RESULT_DIR/comparison.json"
~~~

## 开放式评测生成

~~~bash
OPEN_EVAL_FILE=<开放式评测集 JSONL>

python eval/scripts/generate.py \
  --model-id "$MODEL_DIR" --variant base \
  --single-file "$OPEN_EVAL_FILE" \
  --categories daily lore_values boundary --skip-multi \
  --output-dir "$RESULT_DIR/open_eval_base"

python eval/scripts/generate.py \
  --model-id "$MODEL_DIR" --variant finetuned --adapter "$ADAPTER_DIR" \
  --single-file "$OPEN_EVAL_FILE" \
  --categories daily lore_values boundary --skip-multi \
  --output-dir "$RESULT_DIR/open_eval_finetuned"
~~~

不要提交模型权重、训练输出、评测输出、API 密钥或上游原始游戏数据。
