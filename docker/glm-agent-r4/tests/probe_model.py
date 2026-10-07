#!/usr/bin/env python3
"""No weights/GPU: collect actual model metadata, tokenizer encodings and source hashes."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

TAGS=('<think>','</think>','<|observation|>','<|user|>','<|assistant|>')

def read(path):
    return json.loads(path.read_text()) if path.exists() else {}

def metadata(directory):
    config=read(directory/'config.json');generation=read(directory/'generation_config.json')
    tokenizer=read(directory/'tokenizer_config.json');tokens=read(directory/'tokenizer.json')
    mapping={t['content']:t['id'] for t in tokens.get('added_tokens',[])}
    for tid,spec in tokenizer.get('added_tokens_decoder',{}).items():mapping[spec['content']]=int(tid)
    vocab=tokens.get('model',{}).get('vocab',{})
    if isinstance(vocab,dict):
        for tag,tid in vocab.items():
            if tag in TAGS:mapping.setdefault(tag,tid)
    def eos(value):return value if isinstance(value,list) else [] if value is None else [value]
    tokenizer_eos=tokenizer.get('eos_token')
    if isinstance(tokenizer_eos,dict):tokenizer_eos=tokenizer_eos.get('content')
    # Show each source independently. This approximates upstream config assembly;
    # the server's [GLM-TAG-R3] stops event is the authoritative runtime set.
    sources={'config':eos(config.get('eos_token_id')),
             'text_config':eos(config.get('text_config',{}).get('eos_token_id')),
             'generation_config':eos(generation.get('eos_token_id')),
             'tokenizer_config_eos_token':eos(mapping.get(tokenizer_eos))}
    return {'tag_ids':{t:mapping.get(t) for t in TAGS},'eos_sources':sources,
            'think_in_eos_sources':{name:[i for i in ids if i in {mapping.get(t) for t in TAGS[:2]}]
                                    for name,ids in sources.items()},
            'tokenizer_eos_token':tokenizer_eos,
            'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (directory/'config.json',directory/'generation_config.json',
                             directory/'tokenizer_config.json',directory/'tokenizer.json') if p.exists()}}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_dir',type=Path);ap.add_argument('--app',type=Path,default=Path('/app'))
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result=metadata(a.model_dir)
    try:
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(a.model_dir/'tokenizer.json'))
        encodings={}
        for tag in TAGS[:2]:
            tokenizer.encode_special_tokens=False
            special=tokenizer.encode(tag,add_special_tokens=False).ids
            tokenizer.encode_special_tokens=True
            ordinary=tokenizer.encode(tag,add_special_tokens=False).ids
            encodings[tag]={'special':special,'ordinary':ordinary,
                            'ordinary_roundtrip':tokenizer.decode(ordinary,skip_special_tokens=False),
                            'ordinary_excludes_tag_id':result['tag_ids'][tag] not in ordinary}
        result['tokenizer_encodings']=encodings
    except (ImportError,OSError,ValueError) as exc:
        result['tokenizer_probe_error']=str(exc)
    result['source_hashes']={}
    for relative in ('backends/exllamav3/model.py','common/glm_tag_safety.py',
                     'endpoints/OAI/utils/stream_parser.py','endpoints/OAI/utils/chat_completion.py'):
        p=a.app/relative
        if p.exists():result['source_hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        distribution=importlib.metadata.distribution('exllamav3')
        result['exllamav3_version']=distribution.version
        for relative in ('exllamav3/tokenizer/tokenizer.py','exllamav3/model/config.py','exllamav3/generator/job.py'):
            p=Path(distribution.locate_file(relative))
            if p.exists():result['source_hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    except importlib.metadata.PackageNotFoundError:
        result['exllamav3_version']='not installed in probe interpreter'
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if 'tokenizer_probe_error' in result or any(not e['ordinary_excludes_tag_id'] or e['ordinary_roundtrip']!=tag
        for tag,e in result.get('tokenizer_encodings',{}).items()):raise SystemExit(1)

if __name__=='__main__':main()
