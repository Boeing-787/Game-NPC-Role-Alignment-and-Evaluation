"""Split lore_values canon scores into verified facts vs sourced value applications."""
import csv,json,statistics
from pathlib import Path

def main():
 cases={json.loads(t)['id']:json.loads(t) for t in Path('data/eval_single.jsonl').read_text(encoding='utf8').splitlines() if t.strip()}
 key=json.loads(Path('results/blind_key.json').read_text(encoding='utf8'))
 rows=list(csv.DictReader(Path('results/blind_scores.csv').open(encoding='utf-8-sig',newline='')))
 output=['# 设定题分层统计','', '| 子类型 | n | Before | After | 增量 |','|---|---:|---:|---:|---:|']
 for subtype in ('canonical_fact','value_application'):
  pairs=[]
  for r in rows:
   id=r['id']
   if cases.get(id,{}).get('subtype')!=subtype:continue
   try:
    before=float(r[next(k for k,v in key[id].items() if v=='base')+'_canon'])
    after=float(r[next(k for k,v in key[id].items() if v=='finetuned')+'_canon'])
   except (ValueError,KeyError,StopIteration) as e:raise ValueError(f'请先完成 {id} 两个匿名回复的 canon 分数') from e
   if not (1<=before<=5 and 1<=after<=5):raise ValueError(f'{id} 分数必须在 1-5')
   pairs.append((before,after))
  if not pairs:raise ValueError('找不到 '+subtype+' 评分')
  b=statistics.mean(t[0] for t in pairs);a=statistics.mean(t[1] for t in pairs)
  output.append(f'| {subtype} | {len(pairs)} | {b:.2f} | {a:.2f} | {a-b:+.2f} |')
 path=Path('results/grounded_breakdown.md');path.parent.mkdir(exist_ok=True)
 path.write_text('\n'.join(output)+'\n',encoding='utf8')
 print('Saved',path)

if __name__=='__main__':main()
