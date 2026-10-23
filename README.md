# 重要：Qwen3-14B + H100 80GB 实验版使用说明

此版本在原项目基础上切换为 `Qwen/Qwen3-14B`，使用 BF16 LoRA（并开启 gradient checkpointing）。
Qwen3 默认可能进入 thinking 模式，因此**训练和推理均显式使用 `enable_thinking=False`**。
训练脚本通过 `tokenizer.apply_chat_template(...,enable_thinking=False)` 生成纯文本 prompt，
构造 `prompt`/`completion` 数据，用 `completion_only_loss=True` 只训练 assistant 回复。

**注意：没有在用户的 H100 上实测；先完成两步冒烟测试。**

## 1. 你上传的数据审查结果

- 原始 `curated.jsonl` 89 行；格式全部合法，但包含高相似回复对、问句中混入来源字段、战斗短语占比高和 `boundary=0`。
- `data/AUDIT_ROWS.csv` 逐条写明标志及建议提问；`data/NEAR_DUPLICATES.csv` 是近重复组。
- `data/curated.REVIEW.jsonl` 为**56 条对话优先的候选种子数据**：已经删除高相似样本、低上下文片段、部分战斗音效，并对明显模板化提问做了建议改写。
- 每条保留 `source_ids`；但本包**不包含 verified sources.csv**。候选数据不能宣称已核验，修改的问题也需要人工与答案对应复核。
- `audit_metadata` 只是审核记录，训练脚本只读取 `messages`。
- 强烈建议新增约 80–150 条真正的开放式日常/价值观/边界对话后，再做正式评测。

## 2. 技术冒烟测试（仅检查是否能跑通）

```bash
# 先按原 README 安装 GPU 版 PyTorch 和 requirements.txt
python scripts/quick_split_unverified.py
cp data/test.PROPOSED.jsonl data/test.jsonl
python scripts/generate.py --variant before
python scripts/train.py --max-steps 2
```

以上数据尚未完成来源审核，仅可用于**探索性**跑通流程。不要将其作为最终作业的来源已核验数据。

## 3. 正式数据和训练

根据 `data/AUDIT_ROWS.csv` 审核样本，并提供带真实链接的 `data/sources.csv`，执行：

```bash
cp data/curated.REVIEW.jsonl data/curated.jsonl
python scripts/prepare_data.py
python scripts/check_leakage.py
python scripts/generate.py --variant before
python scripts/train.py --max-steps 2
python scripts/train.py
python scripts/generate.py --variant after
python scripts/make_scoresheet.py
# 人工填写 results/blind_scores.csv 后
python scripts/summarize.py
```

注意 `curated.REVIEW.jsonl` 的 `audit_metadata` 仅供审查，不参与模型输入；只有当
`data/sources.csv` 中的 URL、原文和 `source_ids` 都能对应时才视为满足作业的来源要求。

## 4. 运行配置

- Qwen3-14B, 2 epochs, lr=8e-5, per_device_batch_size=1, grad_accum=8, sequence=768, LoRA r=16/alpha=32。
- H100 80GB 预计可以尝试此方案，但显存占用和速度必须实际确认，遇到 OOM 优先将 `max_length` 改为 512，或者替换模型为 `Qwen/Qwen3-8B`。
- 如果先只有 56 条，则实验是种子级验证，性能变化的统计置信度非常有限。
- Qwen3.5-9B 是更新的模型，但它的官方 Transformers 使用了多模态模型接口；**不能直接将 config 改名后继续使用此纯文本 AutoModelForCausalLM 脚本**。


---

## 原始项目 README（Qwen2.5 版本说明，仅作参考）

# NPC 对话 BF16 LoRA 微调（NVIDIA H100）

> **交付属性**：可运行的项目骨架、数据模板和 24 道评测建议题。**不包含真实游戏语料、已完成模型训练、GPU 实测数据或最终实验结果。**
>
> 研究范围严格对应用户给定 PDF 的 A（角色语料）/B（SFT 数据）/C（LoRA 训练）/D（Before/After 对比）。

## 0. 两天工作的最低完成标准

- **Day 1**：敲定角色及资料来源，形成 150–250 条溯源样本、20 条左右验证样本，准备 24 题独立测试；首次用 2 步冒烟测试验证 H100 + BF16 + loss mask。
- **Day 2**：完成一次 LoRA 训练、保存 adapter；生成 before/after 答案、盲评、汇总表格和 loss 图；填写报告初稿。
- **Day 3（可选）**：复核来源与边界案例，分析退化样本，尝试基于诊断的单项改进，并整理复现包。

**不做**：RAG、Agent、游戏前端、7B/更大模型、多轮复杂网格搜索。

## 1. 角色和资料来源

默认角色为《原神》钟离。这是可替换的实施假设，并非作业强制角色。中文数据建议主要从游戏内角色语音、剧情文本人工核验，并在 `data/sources.csv` 标记来源。可辅助定位的社区页面（**非官方原始发布者**）：

- 中文语音索引：https://genshin-impact.fandom.com/wiki/Zhongli/Voice-Overs/Chinese
- 角色语音分类：https://genshin-impact.fandom.com/wiki/Zhongli/Voice-Overs
- 角色档案：https://genshin-impact.fandom.com/wiki/Zhongli/Profile

**优先在游戏内或权利人发布内容中核对文本；社区站可能存在版本差异。** 每条训练样本必须有真实 `source_ids`，且在 `sources.csv` 中留存来源、原文片段、获取日期和构造方式。仅将这些资料用于本次面试评估，注意源站和游戏内容的使用条款。不要公开大量受版权保护的完整剧情脚本。

例子文件 `data/curated.TEMPLATE.jsonl`、`data/sources.TEMPLATE.csv` 中是**自行构造的演示文本，非原作原文**；脚本会主动拒绝以 `EXAMPLE_` 开头的记录，防止误作为成果提交。

## 2. 环境：建议 Linux + Python 3.11 + H100

先检查驱动和 CUDA：

```bash
nvidia-smi
python3 --version
python3 - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('name:', torch.cuda.get_device_name(0))
    print('BF16:', torch.cuda.is_bf16_supported())
    print('memory GB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
PY
```

创建环境（建议在配套 CUDA 驱动正常的 Linux 机器上运行）：

```bash
conda create -n npc-lora python=3.11 -y
conda activate npc-lora
python -m pip install --upgrade pip
# 依照 https://pytorch.org/get-started/locally/ 为实际驱动/CUDA 选择匹配 CUDA 版 PyTorch。
# 不建议盲目替换服务器当前正常运行的 torch。
python -m pip install -r requirements.txt
python -c "import torch,transformers,trl,peft,datasets; print(torch.__version__,transformers.__version__,trl.__version__,peft.__version__,datasets.__version__)"
```

依赖的主要 API 针对 TRL `0.27.1` 和 Transformers 4.x。若你的机房环境中已有固定版本，**先在新环境中确认可安装的版本，而不要覆盖已有 CUDA 训练环境**。首次训练必须做下方的冒烟测试。

## 3. 数据：人工构建，不凭空声称“已收集”

复制模板并编辑实际来源：

```bash
cp data/curated.TEMPLATE.jsonl data/curated.jsonl
cp data/sources.TEMPLATE.csv data/sources.csv
cp data/test.PROPOSED.jsonl data/test.jsonl
```

⚠️ `curated.jsonl` 和 `sources.csv` 中的演示条目必须替换成真实数据，否则 `prepare_data.py` 会拒绝继续。`test.PROPOSED.jsonl` 是**我们自行设计的候选测试问题**，与真实剧情语料来源无关，可以在训练前审核并冻结；请核对与训练/验证数据是否近似重复。不要根据训练结果选择性修改测试问题。

### 数据 schema

训练原始条目 `data/curated.jsonl` 的单条结构：

```json
{
  "id": "SFT0001",
  "group_id": "voice-zhongli-topic-001",
  "source_ids": ["S001"],
  "category": "daily",
  "construction_method": "grounded_adaptation",
  "messages": [
    {"role": "user", "content": "【替换为依据原始片段构造、已经审核的问题】"},
    {"role": "assistant", "content": "【替换为依据原文或角色设定构造、已经审核的回答】"}
  ]
}
```

- `group_id`：**同一原始剧情段落/语音主题/可明显互相改写的一组样本**统一 ID，防止训练验证集共享同一片段。
- `source_ids`：真实出处 ID，与 `sources.csv` 对应；一个来源可以有多个样本，但过密改写会导致过拟合。
- `category`：`daily`、`lore`、`values`、`boundary`。
- `construction_method`：`direct`（直接取原文配上下文）、`grounded_adaptation`（按原文改写）、`grounded_synthesis`（基于已核查设定生成后人工审核）。
- `messages`：本项目 MVP **仅支持单轮 user + assistant**；训练脚本自动在其前加入和测试阶段一致的短 system prompt。

### 规模建议与质检

训练/验证目标约 200/20；优先确认至少 150 条质量好的样本。源语料拆解为语音、故事、剧情互动等，用人工重写并保留 `source_ids`。不要将无来源自动生成回答当官方台词。

类别可以先按以下比例试做：`daily 35%`、`lore 30%`、`values 20%`、`boundary 15%`。建议对训练样本做人工复核：语气是否一致、剧情事实是否可信、问答是否相关、边界是否合适、是否出现同义重复。`prepare_data.py` 只能查准确匹配和来源索引，**不能自动查角色事实真假和语义级泄漏**。

```bash
python scripts/prepare_data.py
# 生成 data/train.jsonl 与 data/val.jsonl，按 group_id 分组切分，默认约 9:1
```

随后人工检查两个数据集之间的语义重复、group_id 分组合理性。还可以运行 `python scripts/check_leakage.py` 对测试集做字符近似重合提示。`data/test.jsonl` 的 24 条问题分为 8 条 daily、8 条 lore 和 8 条 boundary；别将这些问题或目标回复输入训练集。

## 4. 先运行未微调的 Before

```bash
python scripts/generate.py --variant before
# 输出 results/before.jsonl
```

训练前先跑 Before 可以尽早发现模型加载、推理 chat template 或测试题问题。所有生成操作使用相同 `system_prompt`、`do_sample=False`、`max_new_tokens`。

## 5. 冒烟测试与正式 LoRA 训练

在 `configs/experiment.yaml` 中确认模型、系统提示和超参。在首次全量训练之前：

```bash
python scripts/train.py --max-steps 2
# 注意：这只是冒烟测试！不会输出正式 adapter。
```

关注打印的 `有效 completion tokens`：不能为 0。脚本使用 **prompt/completion 格式 + completion_only_loss=True**，与 Qwen2.5 的 chat template 配合，避免直接启用对部分模板不兼容的 `assistant_only_loss=True`。

通过后执行正式训练：

```bash
python scripts/train.py
# 产物：outputs/zhongli_lora/adapter/
#       outputs/zhongli_lora/experiment_record.json
#       outputs/zhongli_lora/trainer_log_history.json
python scripts/plot_loss.py
```

运行后保存准确环境版本、显卡信息和 PyTorch/CUDA 状态：

```bash
python -m pip freeze > outputs/zhongli_lora/pip_freeze.txt
nvidia-smi > outputs/zhongli_lora/gpu_snapshot.txt
```

需要完全相同的基础模型时，将 `configs/experiment.yaml` 中的 `model_revision` 锁定为具体 commit ID，而不是只写模型名。

## 6. After、A/B 盲评、统计

```bash
python scripts/generate.py --variant after
python scripts/make_scoresheet.py
```

`results/blind_scores.csv` 中回答顺序已随机化。**先不要查看 `results/answer_key.csv`**，以降低人工判分偏差。

- 每条都填 `style_A/B`、`identity_A/B`、`quality_A/B`（整数 1–5）。
- `lore` 类额外填 `lore_A/B`，其他类别留空。
- `boundary` 类额外填 `boundary_A/B`，其他类别留空。
- 评分前固定标准：1 明显失败、3 部分符合、5 明显符合；角色身份和话风不能只靠口头禅判定。

评分后：

```bash
python scripts/summarize.py
# 输出 results/summary.csv 和 results/summary.png
```

人工挑选 3 条 case：提升明显、提升不明显、明显退化/失败。记录完整 Before/After 原文、定性解释和可能归因。指标上升不等于因果已证明：在报告中说明评审主观性、样本数量以及基础模型已有角色知识等局限。

## 7. 报告与交付

在 `REPORT_TEMPLATE.md` 中填好 A/B/C/D、训练曲线、评分汇总表与失败案例。

最小交付清单：

- `REPORT_TEMPLATE.md` 填写后改名为 `report.md`，可另转 PDF。
- `data/curated.jsonl`、`data/sources.csv`、`data/train.jsonl`、`data/val.jsonl`、`data/test.jsonl`（如有分发权利限制，按面试要求提交并注明版权）。
- `configs/experiment.yaml`、`scripts/`、依赖版本与复现命令。
- Adapter 权重或访问路径、训练日志和 `results/before.jsonl`、`results/after.jsonl`、`results/summary.csv`。

## 8. 常见故障（优先排查）

| 现象 | 优先处理 |
|---|---|
| `torch.cuda.is_available=False` | `nvidia-smi` 是否识别 GPU；PyTorch 是否装了 CUDA wheel；驱动是否兼容 |
| `KeyError: qwen2` | Transformers 版本过旧 |
| `CUDA out of memory` | batch 4→2→1；max_length 512→384；必要时关闭其他 GPU 进程 |
| loss 为 NaN | 检查数据有无极长文本；batch 先减半；检查 BF16 与 CUDA 运行状态 |
| target completion tokens=0 | 检查 `messages` 格式和回答长度；确认截断后仍有 assistant 内容 |
| train loss 下降但角色答复机械重复 | 降低 epoch、排查模板化数据、重复样本和泄漏 |
| After 变差 | 检查语料正确性/风格一致性、拆分方式与过拟合；不要只报好的 case |

## 9. 本地可运行的代码质量测试（无需 GPU）

```bash
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
```

**重要限制**：此工程由离线环境编写并进行了静态和纯 Python 测试；不代表已在你的 H100、具体 CUDA 驱动、网络或实际游戏语料上完成训练。请首先执行环境检查与 2 步冒烟训练。
