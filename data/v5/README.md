# Zhongli v5 corpus

This corpus is built afresh by `scripts/build_zhongli_v5.py`. It does **not**
read v3/v4 curated, train, validation, or evaluation files.

## Sources and scope

- Official task-dialogue rows come from explicit `TALK_ROLE_NPC:10232` Zhongli
  nodes in the CHS quest corpus emitted locally from `animegamedata2`.
  Every non-Zhongli context speaker must resolve from `node.speaker`, a
  PLAYER/BLACK_SCREEN role, or `TALK_ROLE_NPC role.id -> NpcExcelConfigData ->
  TextMap_MediumCHS`; unresolved candidates are excluded rather than rendered
  as a generic “某人”.
  Official character-voice rows come from the Zhongli title-to-text dictionary
  in `Genshin-Impact-NPC-Audio-Texts-Dataset/lines.json`. `sources.csv`
  records the upstream commit/Talk node or the voice title for every row.
- `boundary` rows are deliberately labelled `human_authored`. They are an
  independent scenario-authoring ledger, not game quotations; their source URL
  is `manual://review-required` and their origin must be stated as such in the
  report.
- The task categories are `daily`, `lore`, `values`, and `boundary`. For
  formal reporting, combine `lore` and `values` into **Lore & Values**.

The AnimeGameData2 projection does not expose a ready-to-use table of Zhongli's
profile voice lines, so the separate voice dataset supplies these. Its prompt
is a transparent template of the official title (not a claimed in-game player
utterance); its assistant response remains the official voice text. Inspect
`curated.REVIEW.jsonl` before final training and cite this construction rule.

## Files

- `curated.REVIEW.jsonl`: formal fields plus classification and reviewer notes.
- `audio.REVIEW.jsonl`: the 38 title-grounded character-voice rows for focused
  category review.
- `curated.jsonl`: only the formal SFT schema.
- `train.jsonl`, `val.jsonl`: training and early-stopping data.
- `test_reference.jsonl`: frozen official quote pairs for reference-answer
  evaluation. It has no manual Boundary rows.
- `test_reference_eval.jsonl`: the same test with the original message history
  and a separate `reference_answer` field; use this for generation/scoring.
- `sources.csv` and `build_audit.json`: provenance, selection and leakage audit.

## Split rule

Official samples are assigned by whole quest group, approximately 70% / 15% /
15% for train / validation / reference test. No quest can appear in more than
one split. Manual Boundary samples are split only between train and validation,
because there is no official reference answer to compare with.

Do not tune model, prompts, decoding, epoch count, or data selection against
`test_reference.jsonl`. The independent 28/40-item open-question suite remains
the primary role-play evaluation set.

## Reference-answer LLM judge

`test_reference_eval.jsonl` is evaluated separately from `eval/data`. First,
generate one anonymous candidate answer per row with
`scripts/generate_reference.py`. Then use `scripts/score_v5_task_quality.py`.
The script deliberately does not send a `base` or `finetuned` label.

The judge is explicitly told that the target character is Zhongli from Genshin
Impact and receives a compact role card. Daily remains open-ended role-play:
its official reply is shown as a style, identity, and context anchor rather
than a unique semantic target; the score is style 40%, persona 25%, relevance
25%, and factual restraint 10%. Lore receives the official reply as a semantic,
style, and factual/value anchor, not a wording target; its weights are semantic
fidelity 50%, grounding 25%, persona 15%, and style 10%. Boundary has no
official-reference row and is scored only in the frozen 28-item human-blind
suite. The scorer writes every structured judgement, per-category averages,
and 5,000-sample paired bootstrap intervals.

Use the current DeepSeek Flash model identifier `deepseek-flash`; the key is
read from `DEEPSEEK_API_KEY`, never stored in this repository.
