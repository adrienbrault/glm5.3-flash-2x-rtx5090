#!/usr/bin/env python3
"""CPU probe of the actual model tokenizer JSON; never loads weights."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from tokenizers import Tokenizer

class IDs:
    def __init__(self,ids):self.ids=list(ids);self.shape=(1,len(self.ids))
    def flatten(self):return self
    def tolist(self):return self.ids
    def new_tensor(self,rows):return IDs(rows[0])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_dir',type=Path);ap.add_argument('--app',type=Path,default=Path('/app'))
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    spec=importlib.util.spec_from_file_location('r4_safety',a.app/'common/glm_tag_safety.py')
    safety=importlib.util.module_from_spec(spec);spec.loader.exec_module(safety)
    raw=Tokenizer.from_file(str(a.model_dir/'tokenizer.json'))
    raw.no_truncation();raw.no_padding()
    class Wrapper:
        def single_id(self,text):return raw.token_to_id(text)
        def encode(self,text,add_bos=False,encode_special_tokens=False,embeddings=None):
            raw.encode_special_tokens=not encode_special_tokens
            return IDs(raw.encode(text,add_special_tokens=False).ids)
    tokenizer=Wrapper();rows={}
    for tag in ('<think>','</think>','<|observation|>','<|user|>','<tool_call>','</tool_call>','</fake>'):
        native=tokenizer.encode(tag).ids
        protected=[tid for part in safety._literal_pieces(tokenizer,tag) for tid in part.ids]
        tid=tokenizer.single_id(tag)
        rows[tag]={'added_id':tid,'native_encode_false':native,'r4_ids':protected,
                   'r4_roundtrip':raw.decode(protected,skip_special_tokens=False),
                   'r4_excludes_tag_id':tid is None or tid not in protected}
    config={}
    for filename in ('tokenizer.json','tokenizer_config.json','generation_config.json','config.json'):
        path=a.model_dir/filename
        if path.exists():config[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
    encoded=json.loads(raw.to_str())
    result={'model_files_sha256':config,'tag_encodings':rows,
            'added_tag_metadata':[t for t in encoded.get('added_tokens',[]) if t['content'] in rows],
            'safety_sha256':hashlib.sha256((a.app/'common/glm_tag_safety.py').read_bytes()).hexdigest(),
            'scope':'Real Rust tokenizer with ExLlama-style flag mapping; no ExLlama imports, no weights'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if any(not row['r4_excludes_tag_id'] or row['r4_roundtrip']!=tag for tag,row in rows.items()):raise SystemExit(1)

if __name__=='__main__':main()
