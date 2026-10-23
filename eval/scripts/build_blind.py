"""Create a double-anonymized (A/B) evaluation sheet and a private mapping key."""
import argparse
import csv
import json
import random
from pathlib import Path

DIMENSIONS=['style','persona','relevance','canon','boundary','context','multiturn']
CATEGORY_DIMENSIONS={
    'daily':['style','persona','relevance'],
    'lore_values':['style','persona','relevance','canon'],
    'boundary':['style','persona','relevance','boundary'],
    'context':['style','persona','relevance','context'],
    'multiturn':['style','persona','relevance','multiturn'],
}


def load_jsonl(file):
    items=[json.loads(l) for l in Path(file).read_text(encoding='utf8').splitlines() if l.strip()]
    ids=[r['id'] for r in items]
    if len(ids)!=len(set(ids)):
        raise ValueError(f'Duplicate IDs in {file}')
    return {r['id']:r for r in items}


def render(item):
    if item['type']=='single':
        return item['response']
    return '\n\n'.join(f'第{i+1}轮\n用户：{q}\n钟离：{a}' for i,(q,a) in enumerate(zip(item['turns'],item['responses'])))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',default='results/base.jsonl')
    ap.add_argument('--finetuned',default='results/finetuned.jsonl')
    ap.add_argument('--single',default='data/eval_single.jsonl')
    ap.add_argument('--multi',default='data/eval_multi.jsonl')
    ap.add_argument('--skip-multi', action='store_true', help='仅构建正式 28 条单轮核心题的盲评表。')
    ap.add_argument('--categories', nargs='+', default=None,
                    help='仅保留给定类别，例如 daily lore_values boundary。')
    ap.add_argument('--out',default='results')
    ap.add_argument('--seed',type=int,default=20261020)
    ap.add_argument('--evidence-ledger',default='data/evidence_ledger.csv',help='Grounded lore source annotations; never included in model input')
    args=ap.parse_args()
    base=load_jsonl(args.base); after=load_jsonl(args.finetuned)
    cases=load_jsonl(args.single)
    if not args.skip_multi:
        cases |= load_jsonl(args.multi)
    if args.categories is not None:
        allowed = set(args.categories)
        cases = {case_id: case for case_id, case in cases.items() if case.get('category') in allowed}
        if not cases:
            raise ValueError('类别过滤后没有待评分样本')
    if set(cases)!=set(base) or set(cases)!=set(after):
        raise SystemExit('Baseline, fine-tuned and test-case ID sets do not match. Check generated files.')
    evidence={}
    if Path(args.evidence_ledger).exists():
        with Path(args.evidence_ledger).open('r',encoding='utf-8-sig',newline='') as ef:
            for ev in csv.DictReader(ef):
                evidence.setdefault(ev['case_id'],[]).append(ev)
    lore_ids=[cid for cid,case in cases.items() if case['category']=='lore_values']
    if set(lore_ids)-set(evidence):
        raise ValueError('Missing evidence entries for lore questions: '+', '.join(sorted(set(lore_ids)-set(evidence))))
    rng=random.Random(args.seed)
    sheet=[];key={}
    for id,case in cases.items():
        if base[id]['category']!=after[id]['category']:
            raise ValueError(f'category mismatch: {id}')
        a_is_base=rng.random()<0.5
        A=base[id] if a_is_base else after[id]
        B=after[id] if a_is_base else base[id]
        key[id]={'A':'base' if a_is_base else 'finetuned',
                 'B':'finetuned' if a_is_base else 'base'}
        prompt=(case['user'] if case['type']=='single' else '\n'.join(f'第{i+1}轮：{t}' for i,t in enumerate(case['turns'])))
        refs=evidence.get(id,[])
        row={'id':id,'category':case['category'],'subtype':case.get('subtype',''),'prompt':prompt,
             'source_sections':'；'.join(f"{x['source_title']} / {x['source_section']}" for x in refs),
             'source_urls':'；'.join(dict.fromkeys(x['source_url'] for x in refs)),
             'grounded_facts':refs[0]['must_include_facts'] if refs else '',
             'grounded_red_flags':refs[0]['must_avoid_errors'] if refs else '',
             'response_A':render(A),'response_B':render(B),
             'reviewer_focus':case.get('reviewer_focus',''), 'notes':'',
             'A_failure_tags':'', 'B_failure_tags':''}
        for prefix in ['A','B']:
            for dim in DIMENSIONS:
                row[f'{prefix}_{dim}']=''
        sheet.append(row)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    columns=list(sheet[0]);score_file=out/'blind_scores.csv'
    with score_file.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(sheet)
    (out/'blind_key.json').write_text(json.dumps(key,ensure_ascii=False,indent=2),encoding='utf8')
    print('Generated',score_file,'and private',out/'blind_key.json')
    print('Score fields: integers 1–5. Fill ONLY the applicable dimensions:')
    for cat,dims in CATEGORY_DIMENSIONS.items(): print(cat,', '.join(dims))
    print('DO NOT share blind_key.json with the evaluator until scores are frozen.')

if __name__=='__main__':main()
