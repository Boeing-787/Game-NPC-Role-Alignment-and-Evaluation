"""Flag potential lexical overlap with training questions before first evaluation run."""
import argparse
import difflib
import json
import re
from pathlib import Path


def records(file):
    return [json.loads(x) for x in Path(file).read_text(encoding='utf8').splitlines() if x.strip()]


def norm(s):
    # Remove role prefix / whitespace / punctuation to compare substantive content.
    s=re.sub(r'请以钟离的身份接续下面的对话，只输出自然回应。','',s)
    s=re.sub(r'[^\w\u4e00-\u9fff]+','',s)
    return s.lower()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True,help='Path to original train JSONL or supplied transcript text')
    p.add_argument('--single',default='data/eval_single.jsonl')
    p.add_argument('--multi',default='data/eval_multi.jsonl')
    p.add_argument('--threshold',type=float,default=0.72)
    a=p.parse_args()
    train=records(a.train)
    trainqs=[(r.get('id',str(i)),norm(next((m['content'] for m in r['messages'] if m['role']=='user'),''))) for i,r in enumerate(train)]
    cases=[]
    for c in records(a.single):cases.append((c['id'],c['user']))
    for c in records(a.multi):
        for i,turn in enumerate(c['turns'],1):cases.append((f'{c["id"]}-T{i}',turn))
    print('train',len(train),'candidate eval inputs',len(cases),'threshold',a.threshold)
    issues=[]
    for id,raw in cases:
        q=norm(raw)
        if not q:continue
        sims=((difflib.SequenceMatcher(None,q,tq,autojunk=False).ratio(),tid) for tid,tq in trainqs if tq)
        sim,train_id=max(sims,default=(0,''))
        if sim>=a.threshold:
            issues.append((id,train_id,sim))
    for id,train_id,sim in sorted(issues,key=lambda item:-item[2]):
        print(f'REVIEW {id} overlaps {train_id} lexical_similarity={sim:.3f}')
    print('flags:',len(issues),'(automatic overlap screening only; semantic review still necessary)')

if __name__=='__main__':main()
