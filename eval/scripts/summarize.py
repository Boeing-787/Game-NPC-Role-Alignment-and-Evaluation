"""Summarize paired blind A/B scores; call only after all scores are filled."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from build_blind import CATEGORY_DIMENSIONS
CORE=['style','persona','relevance']


def avg(xs):return statistics.mean(xs) if xs else float('nan')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sheet',default='results/blind_scores.csv')
    p.add_argument('--key',default='results/blind_key.json')
    p.add_argument('--out',default='results/summary.md')
    a=p.parse_args()
    key=json.loads(Path(a.key).read_text(encoding='utf8'))
    with Path(a.sheet).open('r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    errors=[];items=[]
    for row in rows:
        id=row['id'];cat=row['category']
        if id not in key or cat not in CATEGORY_DIMENSIONS:
            errors.append(f'Unexpected id/category: {id}/{cat}');continue
        vals={}
        for variant in ['base','finetuned']:
            prefix=next((x for x in ['A','B'] if key[id][x]==variant),None)
            if not prefix: errors.append(f'{id}: key mapping missing {variant}');continue
            vals[variant]={}
            for dim in CATEGORY_DIMENSIONS[cat]:
                txt=(row.get(f'{prefix}_{dim}') or '').strip()
                try:score=int(txt)
                except ValueError:
                    errors.append(f'{id}/{variant}/{dim}: missing/noninteger score');continue
                if not 1<=score<=5:
                    errors.append(f'{id}/{variant}/{dim}: score must be 1–5');continue
                vals[variant][dim]=score
        items.append({'id':id,'cat':cat,'values':vals,'notes':row.get('notes',''),'A_failure_tags':row.get('A_failure_tags',''), 'B_failure_tags':row.get('B_failure_tags','')})
    if errors:
        print('\n'.join(errors[:35]));raise SystemExit(f'Cannot summarize: {len(errors)} missing/invalid fields. Finish blind scoring first.')
    def get(cat,variant,dim):
        return [r['values'][variant][dim] for r in items if r['cat']==cat and dim in r['values'][variant]]
    cats=list(CATEGORY_DIMENSIONS)
    lines=['# 角色 NPC 微调评测结果（人工评分）','',f'- 评测场景：{len(items)}（其中多轮场景：{sum(r["cat"]=="multiturn" for r in items)}）','- 评分为 1–5 分，人工盲评；总分不包含多轮场景，后者单独报告。','',
          '## 主测试集：按类别的三项通用指标均分','',
          '| 场景类别 | n | Before | After | Δ (After − Before) | Before胜/平/After胜 |','|---|---:|---:|---:|---:|---|']
    cat_deltas={}
    for cat in cats:
        catrows=[r for r in items if r['cat']==cat]
        b=[avg([r['values']['base'][d] for d in CORE]) for r in catrows]
        f=[avg([r['values']['finetuned'][d] for d in CORE]) for r in catrows]
        wins=sum(x>y for x,y in zip(b,f))
        ties=sum(x==y for x,y in zip(b,f))
        fwins=sum(x<y for x,y in zip(b,f))
        delta=avg(f)-avg(b); cat_deltas[cat]=delta
        lines.append(f'| {cat} | {len(catrows)} | {avg(b):.2f} | {avg(f):.2f} | {delta:+.2f} | {wins}/{ties}/{fwins} |')
    lines+=['','## 各指标均分','','| 指标 | n | Before | After | Δ |','|---|---:|---:|---:|---:|']
    all_dims=['style','persona','relevance','canon','boundary','context','multiturn']
    for dim in all_dims:
        matched=[r for r in items if dim in r['values']['base']]
        b=avg([r['values']['base'][dim] for r in matched]); f=avg([r['values']['finetuned'][dim] for r in matched])
        lines.append(f'| {dim} | {len(matched)} | {b:.2f} | {f:.2f} | {f-b:+.2f} |')
    catmain=[c for c in cats if c!='multiturn']
    lines+=['','## 主结论统计','',
            f'四个主类别等权的通用指标平均变化：{avg([cat_deltas[c] for c in catmain]):+.2f} 分。',
            '该数值用于描述本次固定样本上的差异，不代表统计显著性或普遍泛化效果。',
            '主表按每个场景先计算 style/persona/relevance 均分，再统计同题 Before/After 配对差异；未将不适用指标填为 0。',
            '多轮场景评价针对三轮完整对话，每组计一个场景，不把三轮当作三个独立样本。',
            '', '## 失败与成功案例索引', '']
    changed=[]
    for r in items:
        b=avg([r['values']['base'][d] for d in CORE]);f=avg([r['values']['finetuned'][d] for d in CORE])
        changed.append((f-b,r))
    changed.sort(key=lambda t:t[0],reverse=True)
    for kind, samples in [('明显改善',changed[:3]),('退化或提升最少',changed[-3:])]:
        lines.append(f'### {kind}')
        for delta,r in samples:lines.append(f'- {r["id"]} ({r["cat"]}): 通用指标均分变化 {delta:+.2f}; 备注：{r["notes"] or "待补充归因"}')
        lines.append('')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('saved',out)

if __name__=='__main__': main()
