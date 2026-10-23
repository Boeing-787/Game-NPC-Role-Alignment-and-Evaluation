"""Generate comparable responses for a baseline model and its LoRA adapter.

Requires: torch, transformers, peft, accelerate.
Run twice, using the *same* --model-id and decoding settings.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Keep this byte-for-byte aligned with configs/experiment_v2.yaml.  Both
# Before and After must use the same condition as the SFT training examples.
SYSTEM = '你正在扮演《原神》中的钟离。请依据角色已有设定与说话方式自然回应；不确定的设定不要编造。'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True, help='Same base model ID for both runs; use a local path if necessary')
    ap.add_argument('--variant', required=True, choices=['base', 'finetuned'])
    ap.add_argument('--adapter', help='LoRA adapter directory; required when variant=finetuned')
    ap.add_argument('--output-dir', default='results')
    ap.add_argument('--single-file', default='data/eval_single.jsonl')
    ap.add_argument('--multi-file', default='data/eval_multi.jsonl')
    ap.add_argument('--categories', nargs='+', default=None,
                    help='仅生成指定 single-turn 类别，例如 daily lore_values boundary；默认不筛选')
    ap.add_argument('--skip-multi', action='store_true', help='不生成 multi-file，用于正式 28 条核心单轮题')
    ap.add_argument('--max-new-tokens', type=int, default=192)
    args = ap.parse_args()
    if args.variant == 'finetuned' and not args.adapter:
        ap.error('For the finetuned variant, provide --adapter pointing to the trained LoRA adapter.')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA not available. This script is intended for your GPU environment.')

    from peft import PeftModel
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=False
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tok.eos_token_id

    def reply(history):
        # Qwen3 thinking-enabled models support this flag. For others the
        # model's own chat template decides whether it is meaningful.
        enc = tok.apply_chat_template(
            history, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors='pt'
        )
        enc = enc.to(next(model.parameters()).device)
        with torch.inference_mode():
            out = model.generate(
                input_ids=enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        generated = out[0, enc.shape[-1]:]
        return tok.decode(generated, skip_special_tokens=True).strip()

    result=[]
    single_cases = read_jsonl(args.single_file)
    if args.categories is not None:
        allowed = set(args.categories)
        single_cases = [case for case in single_cases if case.get('category') in allowed]
        if not single_cases:
            ap.error('--categories 未匹配到任何 single-turn 测试题')
    for case in single_cases:
        history=[{'role':'system','content':SYSTEM},{'role':'user','content':case['user']}]
        response=reply(history)
        result.append({'id':case['id'],'category':case['category'],'type':'single',
                       'user':case['user'],'response':response})
        print(args.variant,case['id'],'done',flush=True)
    if not args.skip_multi:
        for case in read_jsonl(args.multi_file):
            history=[{'role':'system','content':SYSTEM}]
            responses=[]
            for prompt in case['turns']:
                history.append({'role':'user','content':prompt})
                response=reply(history)
                responses.append(response)
                history.append({'role':'assistant','content':response})
            result.append({'id':case['id'],'category':'multiturn','type':'multi',
                           'turns':case['turns'],'responses':responses})
            print(args.variant,case['id'],'done',flush=True)
    path=Path(args.output_dir);path.mkdir(parents=True,exist_ok=True)
    out=path/f'{args.variant}.jsonl'
    out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf8')
    (path/f'{args.variant}_config.json').write_text(json.dumps({
        'model_id':args.model_id,'variant':args.variant,'adapter':args.adapter,
        'system_prompt':SYSTEM,'enable_thinking':False,'do_sample':False,
        'max_new_tokens':args.max_new_tokens,'seed':42,
        'n_single':sum(r['type']=='single' for r in result),
        'n_multi':sum(r['type']=='multi' for r in result)
    },ensure_ascii=False,indent=2),encoding='utf8')
    print('saved:',out)


if __name__=='__main__':
    main()
