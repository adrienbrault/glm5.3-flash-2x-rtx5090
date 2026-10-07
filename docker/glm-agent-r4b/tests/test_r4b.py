"""R880 wire/trace replay and real production template + request cache regressions."""
import gzip
import unittest
from harness import *
from backend_seam import Tensor, params, load_methods
from tokenizers import Tokenizer as Rust, models, AddedToken, decoders
import threading
import weakref
ROOT=Path(__file__).parents[1]

class RustWrapper:
    bos_token_id=1
    def __init__(self):
        vocab={chr(i):i-32 for i in range(32,127)};vocab['\n']=95
        # A real BPE merge crossing the </fake> boundary. Segmentation loses it.
        for piece in (' <',' </',' </f',' </fa',' </fak',' </fake',' </fake>'):
            vocab[piece]=len(vocab)
        merges=[(' ','<'),(' <','/'),(' </','f'),(' </f','a'),(' </fa','k'),(' </fak','e'),(' </fake','>')]
        self.tokenizer=Rust(models.BPE(vocab=vocab,merges=merges));self.tokenizer.decoder=decoders.Fuse()
        self.tokenizer.add_tokens([AddedToken(t,special=False) for t in ('<think>','</think>','<tool_call>','<|user|>')])
        self.calls=[];self._encode_lock=threading.RLock()
    def single_id(self,s):return self.tokenizer.token_to_id(s)
    def encode(self,text,add_bos=False,encode_special_tokens=True,embeddings=None):
        self.calls.append(text)
        with self._encode_lock:
            self.tokenizer.encode_special_tokens=not encode_special_tokens
            ids=self.tokenizer.encode(text,add_special_tokens=False).ids
        return Tensor(([1] if add_bos else [])+ids)
    def decode_(self,ids,special):return self.tokenizer.decode(ids,skip_special_tokens=not special)

def make_prompt(text):
    codec=glm_tag_safety.PromptLiterals({})
    prompt,spans=codec.restore('<think>'+codec.mask(text))
    p=params();p._glm_prompt_literals=(prompt,spans)
    return prompt,p

class R4BTests(unittest.IsolatedAsyncioTestCase):
    def test_fake_neighbor_retains_real_bpe_merge_and_literal_ids(self):
        t=RustWrapper();prompt,p=make_prompt('a </fake> <think>x</think>')
        original=t.tokenizer.to_str()
        with patch.dict(os.environ,{'TABBY_GLM_LITERAL_ENCODING':'runs'}):
            ids=glm_tag_safety.encode_prompt(t,prompt,p).ids
        self.assertIn(t.single_id(' </fake>'),ids)
        self.assertEqual(t.decode_(ids,True),prompt)
        self.assertEqual(ids.count(t.single_id('<think>')),1)
        self.assertNotIn(t.single_id('</think>'),ids)
        self.assertEqual(t.tokenizer.to_str(),original)

    def test_surviving_added_tokens_keep_original_model_ids(self):
        t=RustWrapper();prompt,p=make_prompt('a </fake> <think>x</think>')
        prompt='<|user|>'+prompt+'<tool_call>'
        p._glm_prompt_literals=(prompt,tuple((a+8,b+8) for a,b in p._glm_prompt_literals[1]))
        with patch.dict(os.environ,{'TABBY_GLM_LITERAL_ENCODING':'runs'}):
            ids=glm_tag_safety.encode_prompt(t,prompt,p).ids
        self.assertEqual(ids[0],t.single_id('<|user|>'))
        self.assertEqual(ids[-1],t.single_id('<tool_call>'))
        self.assertEqual(t.decode_(ids,True),prompt)
        self.assertEqual(ids.count(t.single_id('<think>')),1)

    def test_base_vocab_literal_control_falls_back_without_rejection(self):
        t=RustWrapper();state=json.loads(t.tokenizer.to_str());vocab=state['model']['vocab']
        # Rebuild a real BPE whose ordinary merges can emit the control ID even
        # with the added matcher removed. Subdivision must still protect data.
        merges=[]
        left='<'
        for right in 'think>':
            combined=left+right;vocab[combined]=len(vocab);merges.append([left,right]);left=combined
        state['model']['merges']+=merges
        state['added_tokens']=[]
        t.tokenizer=Rust.from_str(json.dumps(state))
        t.tokenizer.add_tokens([AddedToken('<think>',special=False),AddedToken('</think>',special=False)])
        prompt,p=make_prompt('a </fake> <think>x</think>')
        with patch.dict(os.environ,{'TABBY_GLM_LITERAL_ENCODING':'runs'}):
            ids=glm_tag_safety.encode_prompt(t,prompt,p).ids
        self.assertEqual(ids.count(t.single_id('<think>')),1)
        self.assertNotIn(t.single_id('</think>'),ids)
        self.assertEqual(t.decode_(ids,True),prompt)

    async def test_same_messages_twice_real_template_encode_and_request_cache(self):
        t=RustWrapper();ns=load_methods();ns['os']=os
        exl=ModuleType('exllamav3');exl.cache_trace=NS(encoded=lambda *a,**kw:None)
        self.enterContext(patch.dict('sys.modules',{'exllamav3':exl}))
        container=NS(tokenizer=t,hf_model=NS(add_bos_token=lambda:False),
            max_seq_len=8192,cache=NS(max_num_tokens=8192),
            generator=NS(generator=NS(recurrent_cache=None)),job_max_rq_tokens=lambda _:4096)
        encoded=[]
        for _ in range(2):
            data=ChatCompletionRequest(messages=[dict(role='user',content='a </fake> <think>x</think>')])
            mc=Model(template=environment(ROOT/'tests/fixtures/upstream-glm-template.jinja'))
            with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1','TABBY_GLM_LITERAL_ENCODING':'runs'}):
                prompt,_=await namespace(mc)['apply_chat_template'](data)
                data.add_bos_token=False
                ns['validate_context_length'](container,prompt,data)
                cached=await ns['_encode_prompt'](container,prompt,data,False,[])
                ns['encode_once_enabled']=lambda:False
                fresh=await ns['_encode_prompt'](container,prompt,data,False,[])
                ns['encode_once_enabled']=lambda:True
                with patch.dict(os.environ,{'TABBY_GLM_CACHE_VERIFY':'1'}):
                    verified=await ns['_encode_prompt'](container,prompt,data,False,[])
            self.assertEqual(cached.ids,fresh.ids)
            self.assertEqual(verified.ids,fresh.ids)
            self.assertIsNot(cached,data._prompt_ids[(prompt,False)][1])
            encoded.append((prompt,cached.ids))
        self.assertEqual(encoded[0],encoded[1])
        # This is the actual app request cache; GPU page/recurrent caches aren't simulated.

    async def test_a4_sse_and_tag_trace_replay_is_byte_exact(self):
        capture=json.loads(gzip.decompress((ROOT/'tests/fixtures/a4-hard.json.gz').read_bytes()))
        reason='';args='';finish=None;wire_deltas=[]
        for _,line in capture['raw']:
            if not line.startswith('data: {'):continue
            for ch in json.loads(line[6:]).get('choices',[]):
                d=ch.get('delta',{});reason+=d.get('reasoning_content','')
                for tc in d.get('tool_calls',[]):
                    x=tc.get('function',{}).get('arguments','');args+=x;wire_deltas.append(x)
                finish=ch.get('finish_reason') or finish
        value=json.loads(args);self.assertEqual(finish,'tool_calls')
        self.assertIn('  </fake>  ',value['content'])
        self.assertEqual(value['content'].count('<think>'),200)
        at=next(i for i,x in enumerate(wire_deltas) if '</fake' in x)
        self.assertEqual(wire_deltas[at-2:at],[' ',' '])
        self.assertEqual(wire_deltas[at+2:at+4],[' ',' '])
        traces=capture['events']
        self.assertEqual([e['token_ids'] for e in traces if e['event']=='generator' and not e['eos']],[[154842]])
        controls=next(e for e in traces if e['event']=='control_spans')
        self.assertEqual(controls['controls'],[[0,8,'</think>']]);self.assertFalse(controls['decode_mismatch'])
        # Trace IDs cover only the real closer; replay those IDs with a matching codec.
        from backend_seam import Tokenizer
        t=Tokenizer();t.pieces[154842]='</think>'
        t.single_id=lambda x:154842 if x=='</think>' else None
        spans,mismatch=glm_tag_safety.ReasoningControls(t).feed('</think>',[154842])
        self.assertEqual(spans,[tuple(x) for x in controls['controls']]);self.assertFalse(mismatch)
        wire=call('write_file',list(value.items()))
        class Replay(Model):
            async def stream_generate(self,*a,**kw):
                yield dict(text=reason,token_ids=[],glm_reasoning_controls=[])
                yield dict(text='</think>',token_ids=[154842],glm_reasoning_controls=spans)
                for c in wire:yield dict(text=c,token_ids=[],glm_reasoning_controls=[])
                yield dict(text='',finish_reason='stop',eos_reason='stop_token')
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],tools=[tool('write_file',
            props={'path':{'type':'string'},'content':{'type':'string'}})])
        for streaming,live in ((False,False),(True,False),(True,True)):
            result,frames=await collect('',data,Replay(),streaming=streaming,live=live)
            calls=assembled(frames) if streaming else result['tool_calls']
            self.assertEqual(json.loads(calls[0]['function']['arguments']),value)
            self.assertEqual((frames[-1] if streaming else result)['finish_reason'],'tool_calls')

    def test_registered_and_legacy_probe_arms(self):
        t=RustWrapper();prompt,p=make_prompt('a </fake> <think>x</think>')
        for mode in ('runs','registered','legacy'):
            with patch.dict(os.environ,{'TABBY_GLM_LITERAL_ENCODING':mode}):
                ids=glm_tag_safety.encode_prompt(t,prompt,p).ids
            self.assertEqual(t.decode_(ids,True),prompt)
            self.assertEqual(ids.count(t.single_id('<think>')),1)
            self.assertNotIn(t.single_id('</think>'),ids)

if __name__=='__main__':unittest.main()
