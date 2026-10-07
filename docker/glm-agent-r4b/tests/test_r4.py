"""r4 CPU regressions: added-token matching, ID-aware channels, real SSE replay."""
import unittest
from harness import *
from backend_seam import Tensor, Tokenizer, params, backend_replay
import importlib.util
import gzip

ROOT = Path(__file__).parents[1]
CLIENT_PATH = Path(os.environ.get('R4_CHECKER', ROOT/'glm53_toolcheck_r4.py'))
spec=importlib.util.spec_from_file_location('r4_checker', CLIENT_PATH)
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)

class UnspecialTokenizer(Tokenizer):
    """GLM-like added tokens match even when encode_special_tokens=False."""
    def encode(self,text,**kw):
        kw['encode_special_tokens']=True
        return super().encode(text,**kw)
    def decode_(self,ids,special):return self.decode(ids)

class R4Tests(unittest.IsolatedAsyncioTestCase):
    def test_unsafe_added_token_is_split_instead_of_rejecting_request(self):
        t=UnspecialTokenizer();p=params();codec=glm_tag_safety.PromptLiterals({})
        text,spans=codec.restore('<think>'+codec.mask('literal </think> <think> </fake>'))
        p._glm_prompt_literals=(text,spans)
        ids=glm_tag_safety.encode_prompt(t,text,p,add_bos=True)
        self.assertEqual(t.decode(ids.ids[1:]),text)
        self.assertEqual(ids.ids.count(10),1)
        self.assertNotIn(11,ids.ids)
        self.assertEqual(ids.ids.count(1),1)

    async def test_user_tool_assistant_reasoning_and_schema_literals(self):
        literal='unquoted </think> and <think>x</think> <|observation|> </fake>'
        t=UnspecialTokenizer()
        for role in ('user','assistant','tool'):
            history=[dict(role=role,content=literal)]
            if role=='tool':history[0]['tool_call_id']='call1'
            if role=='assistant':history[0]['reasoning_content']=literal
            history.append(dict(role='user',content='next'))
            data=ChatCompletionRequest(messages=history,
                 tools=[tool(props={'body':{'type':'string','description':literal}})],
                 chat_template_kwargs={'clear_thinking':False})
            mc=Model(template=environment(ROOT/'tests/fixtures/upstream-glm-template.jinja'))
            with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1'}):
                prompt,_=await namespace(mc)['apply_chat_template'](data)
                ids=glm_tag_safety.encode_prompt(t,prompt,data)
            self.assertEqual(t.decode(ids.ids),prompt)
            self.assertIn(literal,prompt)
            structural=prompt
            for start,end in reversed(data._glm_prompt_literals[1]):structural=structural[:start]+structural[end:]
            for tag,tid in (('<think>',10),('</think>',11),('<|observation|>',99)):
                self.assertEqual(ids.ids.count(tid),structural.count(tag))
            self.assertNotIn('__TABBY_LITERAL_',prompt)

    def test_real_and_literal_closers_with_identical_spelling(self):
        p=TagStreamParser(reasoning_start='<think>',reasoning_end='</think>',
                          tool_start='<tool_call>',tool_end='</tool_call>',start_in_reasoning=True,
                          preserve_tool_reasoning_tags=True,preserve_reasoning_literals=True)
        events=[]
        # No quoting/nesting heuristic: ordinary pieces always remain data.
        for c in 'unquoted </think> then <think>':events+=p.feed_tokenized(c,[])
        self.assertTrue(p.in_reasoning)
        events+=p.feed_tokenized('</think>answer </think>',[(0,8,'</think>')])
        events+=p.finish()
        self.assertEqual(''.join(v for ch,v in events if ch==REASONING),'unquoted </think> then <think>')
        self.assertEqual(''.join(v for ch,v in events if ch==CONTENT),'answer </think>')

    def test_controls_aligned_with_mixed_literals_in_merged_chunk(self):
        t=UnspecialTokenizer();tracker=glm_tag_safety.ReasoningControls(t)
        text='literal </think> then </think>answer'
        literal=t.encode('literal ',encode_special_tokens=False).ids
        literal += [1000+ord(c) for c in '</think> then ']
        ids=literal+[11]+[1000+ord(c) for c in 'answer']
        spans,mismatch=tracker.feed(text,ids)
        self.assertFalse(mismatch)
        at=text.index('</think>',text.index('</think>')+1)
        self.assertEqual(spans,[(at,at+8,'</think>')])

    def test_multibyte_prefix_across_engine_chunks(self):
        class BytesTokenizer(UnspecialTokenizer):
            def decode_(self,ids,special):
                return bytes(i-2000 for i in ids if i!=11).decode(errors='replace') + ('</think>' if 11 in ids else '')
        tracker=glm_tag_safety.ReasoningControls(BytesTokenizer())
        self.assertEqual(tracker.feed('',[2000+0xc3]),([],False))
        self.assertEqual(tracker.feed('é</think>',[2000+0xa9,11]),([(1,9,'</think>')],False))

    async def test_1000_line_exact_body_round_trip_through_collector(self):
        body=client.expected_body('1000-line')
        raw='ordinary </think> mention</think>'+call('write_file',[('path','/tmp/long.py'),('content',body)])
        # Scripted backend emits provenance. Real delimiter is only the second close.
        real=raw.index('</think>',raw.index('</think>')+1)
        class ProvenanceModel(Model):
            async def stream_generate(self,*a,**kw):
                yield dict(text=raw,token_ids=[1],glm_reasoning_controls=[(real,real+8,'</think>')])
                yield dict(text='',finish_reason='stop',eos_reason='stop_token')
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],
             tools=[tool('write_file',props={'path':{'type':'string'},'content':{'type':'string'}})])
        for streaming,live in ((False,False),(True,False),(True,True)):
            result,frames=await collect('',data,ProvenanceModel(),streaming=streaming,live=live)
            calls=assembled(frames) if streaming else result['tool_calls']
            self.assertEqual(json.loads(calls[0]['function']['arguments']),{'path':'/tmp/long.py','content':body})
            reason=''.join(f.get('delta_reasoning_content','') for f in frames) if streaming else result['reasoning_content']
            self.assertEqual(reason,'ordinary </think> mention')

    def test_checker_byte_contract(self):
        body=client.expected_body('hard')
        def check(b):
            msg={'choices':[{'message':{'tool_calls':[{'function':{'name':'write_file','arguments':json.dumps({'path':'/tmp/y.py','content':b})}}]},'finish_reason':'tool_calls'}]}
            return client.summarize([[0,json.dumps(msg)]],False,'hard')
        self.assertTrue(check(body)['passed'])
        for bad in (body.replace('\n','\r\n'),body.rstrip('\n'),body+'\n'):
            r=check(bad);self.assertFalse(r['passed']);self.assertIsNotNone(r['calls'][0]['byte_diff'])

    def test_a2_wire_replay_does_not_manufacture_missing_tokens(self):
        fixture=ROOT/'tests/fixtures/a2-sse.json.gz'
        cases=json.loads(gzip.decompress(fixture.read_bytes()))
        for name,raw in cases.items():
            reasoning=content='';arguments='';finish=None
            p=TagStreamParser(reasoning_start='<think>',reasoning_end='</think>',start_in_reasoning=True)
            events=[]
            for _,line in raw:
                if not line.startswith('data: {'):continue
                for ch in json.loads(line[6:]).get('choices',[]):
                    d=ch.get('delta') or {};r=d.get('reasoning_content') or '';c=d.get('content') or ''
                    reasoning+=r;content+=c
                    # SSE already has channels; IDs are absent. Explicitly replay
                    # visible bytes as data and inject only the observed channel switch.
                    events+=p.feed_tokenized(r,[])
                    if c and p.in_reasoning:events+=p.feed_tokenized('</think>',[(0,8,'</think>')])
                    events+=p.feed_tokenized(c,[])
                    finish=ch.get('finish_reason') or finish
            events+=p.finish()
            self.assertEqual(''.join(v for ch,v in events if ch==REASONING),reasoning,name)
            self.assertEqual(''.join(v for ch,v in events if ch==CONTENT),content,name)
            self.assertEqual(client.summarize(raw,True,name.split('-stream')[0])['finish'],finish)

    async def test_a2_recorded_tool_arguments_replayed_byte_exact(self):
        cases=json.loads(gzip.decompress((ROOT/'tests/fixtures/a2-sse.json.gz').read_bytes()))
        for name,raw in cases.items():
            arguments=''
            for _,line in raw:
                if not line.startswith('data: {'):continue
                for ch in json.loads(line[6:]).get('choices',[]):
                    for tc in ch.get('delta',{}).get('tool_calls',[]):
                        arguments+=tc.get('function',{}).get('arguments') or ''
            if not arguments:continue
            recorded=json.loads(arguments)
            # SSE has JSON, not original model XML. This is explicitly a
            # reconstructed tool grammar replay of those captured arguments.
            wire=call('write_file',list(recorded.items()))
            class Replay(Model):
                async def stream_generate(self,*a,**kw):
                    for ch in wire:yield dict(text=ch,token_ids=[1],glm_reasoning_controls=[])
                    yield dict(text='',finish_reason='stop')
            data=ChatCompletionRequest(messages=[dict(role='user',content='go')],tools=[
                 tool('write_file',props={'path':{'type':'string'},'content':{'type':'string'}})])
            result,frames=await collect('',data,Replay(),streaming=True,live=True)
            parsed=json.loads(assembled(frames)[0]['function']['arguments'])
            self.assertEqual(parsed,recorded,name)
            self.assertEqual(parsed['content'].encode(),recorded['content'].encode(),name)

    def test_default_sampling_matrix_is_unmodified(self):
        a=NS(model='m',cases=['hard','think-only','close-only','1000-line'],
             modes=['stream'],temperatures=['zero','default'],repeats=2,max_tokens=12000,long_max_tokens=40000)
        rows=list(client.request_matrix(a));self.assertEqual(len(rows),16)
        for tag,case,body in rows:
            self.assertEqual(body.get('temperature'),0 if '-zero-' in tag else None)
            self.assertNotIn('top_p',body)
            self.assertNotIn('min_p',body)

    def test_vision_aliases_are_excluded_and_positions_preserved(self):
        codec=glm_tag_safety.PromptLiterals({},excluded=['<image_alias>'])
        prompt,spans=codec.restore(codec.mask('<image_alias> literal </think>'))
        self.assertEqual(prompt,'<image_alias> literal </think>')
        self.assertEqual(len(spans),1)
        self.assertEqual(prompt[slice(*spans[0])],'</think>')
        visits=[]
        embedding=NS(text_alias='<image_alias>',token_list_at=lambda pos:visits.append(pos) or [9])
        shifted=glm_tag_safety._shift_embeddings([embedding],[Tensor([1,2,3])])[0]
        self.assertEqual(shifted.text_alias,embedding.text_alias)
        self.assertEqual(shifted.token_list_at(2),[9])
        self.assertEqual(visits,[5])
        t=UnspecialTokenizer();p=params()
        untouched=glm_tag_safety.encode_prompt(t,'ordinary vision',p,embeddings=[embedding])
        self.assertEqual(t.calls[-1],('ordinary vision',False,True))

    def test_actual_rust_added_token_matching(self):
        try:from tokenizers import Tokenizer as Rust, models, AddedToken, decoders
        except ImportError:self.skipTest('tokenizers unavailable; required in image')
        raw=Rust(models.BPE(vocab={chr(i):i for i in range(32,127)},merges=[]))
        raw.decoder=decoders.Fuse()
        raw.add_tokens([AddedToken(t,special=False) for t in glm_tag_safety.TAGS])
        class Wrapper(UnspecialTokenizer):
            def encode(self,text,add_bos=False,encode_special_tokens=False,embeddings=None):
                return Tensor(([1] if add_bos else [])+raw.encode(text,add_special_tokens=False).ids)
            def single_id(self,text):return raw.token_to_id(text) if text in glm_tag_safety.TAGS else None
        t=Wrapper();p=params();text='<think>x</think>';p._glm_prompt_literals=(text,((0,7),(8,16)))
        self.assertEqual(raw.encode('</think>',add_special_tokens=False).ids,[t.single_id('</think>')])
        ids=glm_tag_safety.encode_prompt(t,text,p).ids
        self.assertEqual(raw.decode(ids,skip_special_tokens=False),text)
        self.assertFalse(set(ids)&{t.single_id(tag) for tag in glm_tag_safety.TAGS})

if __name__=='__main__':unittest.main()
