#!/usr/bin/env python3
"""Actual installed ExLlama tokenizer + landed template/encode/cache, no weights.

Unlike the earlier literal probe, this tests whole rendered prompts, neighbors,
BOS and the production request-local cache, twice per mode. GPU page allocation
is tested separately by operator-run.sh with EXL3_CACHE_TRACE=1.
"""
import argparse,asyncio,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

def sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
async def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_dir',type=Path);ap.add_argument('--app',type=Path,default=Path('/app'))
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    os.environ['TOOLFIX_APP']=str(a.app);os.environ['TOOLFIX_BASE']=str(a.app)
    # Import installed tokenizer before the harness supplies the logger/sampler seams.
    from exllamav3.tokenizer import Tokenizer
    model_config=json.loads((a.model_dir/'config.json').read_text())
    eos=model_config.get('eos_token_id');eos_list=eos if isinstance(eos,list) else [eos] if eos is not None else []
    cfg=NS(directory=str(a.model_dir),bos_token_id=model_config.get('bos_token_id'),
        pad_token_id=model_config.get('pad_token_id'),eos_token_id=eos_list[0] if eos_list else None,
        eos_token_id_list=list(eos_list))
    tokenizer=Tokenizer(cfg)
    import harness as h
    from backend_seam import load_methods,params
    ns=load_methods();ns['os']=os
    container=NS(tokenizer=tokenizer,hf_model=NS(add_bos_token=lambda:False),max_seq_len=262144,
        cache=NS(max_num_tokens=262144),generator=NS(generator=NS(recurrent_cache=None)),
        job_max_rq_tokens=lambda _:262144)
    import importlib.util
    spec=importlib.util.spec_from_file_location('checker',Path(__file__).parents[1]/'glm53_toolcheck_r4.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    rows=[];failed=False
    for mode in ('legacy','registered','runs'):
        for case in ('hard','think-only','close-only','1000-line'):
            prior=None
            for repeat in (1,2):
                data=h.ChatCompletionRequest(messages=[dict(role='user',content=checker.CASES[case]+
                    ' Use LF line separators and end the file with exactly one LF newline.')],tools=checker.TOOLS)
                mc=h.Model(template=h.environment(Path(__file__).parent/'fixtures/upstream-glm-template.jinja'))
                with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1','TABBY_GLM_LITERAL_ENCODING':mode,
                                           'TABBY_GLM_TAG_TRACE':'0','TABBY_GLM_CACHE_VERIFY':'0'}):
                    prompt,_=await h.namespace(mc)['apply_chat_template'](data)
                    p=params();p.max_tokens=40000;p._glm_prompt_literals=data._glm_prompt_literals
                    ns['validate_context_length'](container,prompt,p)
                    cached=await ns['_encode_prompt'](container,prompt,p,False,[])
                    ns['encode_once_enabled']=lambda:False
                    fresh=await ns['_encode_prompt'](container,prompt,p,False,[])
                    ns['encode_once_enabled']=lambda:True
                ids=cached.flatten().tolist();fresh_ids=fresh.flatten().tolist()
                decoded=tokenizer.decode_(ids,True)
                item=dict(mode=mode,case=case,repeat=repeat,prompt=prompt,decoded_prompt=decoded,
                    decode_exact=decoded==prompt,literal_spans=data._glm_prompt_literals[1],
                    token_ids=ids,fresh_ids=fresh_ids,ids_sha256=sha(ids),cache_equal=ids==fresh_ids,
                    repeats_equal=prior is None or prior==ids)
                rows.append(item);prior=ids
                failed|=not item['cache_equal'] or not item['repeats_equal'] or (mode=='runs' and not item['decode_exact'])
    result=dict(scope='Installed ExLlama tokenizer, real app methods; no model weights or GPU page cache',
                tokenizer_source_sha256=hashlib.sha256(Path(sys.modules[Tokenizer.__module__].__file__).read_bytes()).hexdigest(),
                rows=rows)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:print(json.dumps({k:row[k] for k in ('mode','case','repeat','decode_exact','cache_equal','repeats_equal','ids_sha256')}))
    if failed:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
