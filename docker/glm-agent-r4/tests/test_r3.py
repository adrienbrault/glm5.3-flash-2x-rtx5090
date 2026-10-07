"""Actual parser/collector regressions; scripted IDs are synthetic, not GPU samples."""
import unittest
from harness import *
from backend_seam import Tokenizer, Tensor, params, backend_replay, load_methods
import weakref

class StopAndPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_1000_line_special_ids_in_arguments(self):
        body=''.join(f'row_{i} = "a\\\\b \\"q\\" </fake> <think>x</think>"\n' for i in range(1,1001))
        raw='plan</think>'+call(pairs=[('body',body)])
        parts=re.split('(<think>|</think>)',raw)
        script=[(10 if piece=='<think>' else 11 if piece=='</think>' else 20,piece)
                for piece in parts if piece]+[(99,'<|observation|>')]
        with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1'}):
            chunks,_=await backend_replay(script,params(),hf_eos=(10,11,99))
        class ReplayModel(Model):
            async def stream_generate(self,*a,**kw):
                for c in chunks:yield c
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],
             tools=[tool(props={'body':{'type':'string'}})])
        for streaming,live in ((False,False),(True,False),(True,True)):
            result,frames=await collect('',data,ReplayModel(),streaming=streaming,live=live)
            calls=assembled(frames) if streaming else result['tool_calls']
            self.assertEqual(json.loads(calls[0]['function']['arguments']),{'body':body})
            self.assertEqual((frames[-1] if streaming else result)['finish_reason'],'tool_calls')

    def test_opt_in_trace_contains_trigger_fields_in_console_message(self):
        messages=[];logger=NS(info=lambda message,extra:messages.append((message,extra)))
        for flag in ('0','1'):
            with patch.dict(os.environ,{'TABBY_GLM_TAG_TRACE':flag}):
                glm_tag_safety.trace(logger,'test','generator',eos_reason='stop_token',eos_triggering_token_id=10)
        self.assertEqual(len(messages),1)
        payload=json.loads(messages[0][0].removeprefix('[GLM-TAG-R4] '))
        self.assertEqual(payload['eos_triggering_token_id'],10)
        self.assertEqual(payload,messages[0][1])

    def test_real_hf_configuration_union_does_not_filter_think_ids(self):
        ns=dict(BaseModel=BaseModel,Optional=Optional,Union=Union,List=List,Dict=Dict,Set=Set)
        functions(APP/'common/transformers_utils.py',set(),ns,
                  classes={'GenerationConfig','TextConfig','HuggingFaceConfig','TokenizerConfig','HFModel'})
        for name in ('GenerationConfig','TextConfig','HuggingFaceConfig','TokenizerConfig'):
            ns[name].model_rebuild(_types_namespace=ns)
        hf=ns['HFModel']();hf.hf_config=ns['HuggingFaceConfig'](eos_token_id=[10,99])
        hf.generation_config=ns['GenerationConfig'](eos_token_id=11)
        hf.tokenizer_config=ns['TokenizerConfig'](eos_token='</think>',eos_token_id=42)
        self.assertEqual(set(hf.eos_tokens()),{10,11,99})
        self.assertEqual(hf.tokenizer_config.model_dump(),{'add_bos_token':True})

    async def test_actual_backend_job_receives_filtered_stops_and_special_decode(self):
        prefix,suffix=call(pairs=[('body','PLACEHOLDER')]).split('PLACEHOLDER')
        script=[(20,'plan `'),(10,'<think>'),(21,'x'),(11,'</think>'),(22,'` done'),
                (11,'</think>'),(23,prefix),(10,'<think>'),(21,'x'),(11,'</think>'),
                (24,suffix),(99,'<|observation|>')]
        for request,hf,backend in (([10,11],(99,),()),([],(10,11,99),()),
                                   ([],(99,),(10,11)),(['<think>','</think>'],(99,),())):
            for enabled in ('0','1'):
                p=params(request);before=list(p.stop)
                with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':enabled}):
                    chunks,job=await backend_replay(script,p,hf_eos=hf,backend_eos=backend)
                self.assertEqual(p.stop,before,'A backend must not mutate a reused request')
                self.assertTrue(job['decode_special_tokens'])
                raw=''.join(c.get('text','') for c in chunks)
                if enabled=='0':
                    self.assertEqual(raw,'plan `')
                    self.assertEqual(chunks[-1]['stop_str'],'<think>')
                else:
                    self.assertEqual(job['stop_conditions'],[99])
                    data=ChatCompletionRequest(messages=[dict(role='user',content='go')],
                         tools=[tool(props={'body':{'type':'string'}})])
                    class ReplayModel(Model):
                        async def stream_generate(self,*a,**kw):
                            for c in chunks:yield c
                    for streaming,live in ((False,False),(True,False),(True,True)):
                        result,frames=await collect('',data,ReplayModel(),streaming=streaming,live=live)
                        reasoning=''.join(f.get('delta_reasoning_content','') for f in frames) if streaming else result['reasoning_content']
                        self.assertEqual(reasoning,'plan `<think>x', 'Real closing ID ends reasoning even in quotes')
                        calls=assembled(frames) if streaming else result['tool_calls']
                        self.assertEqual(json.loads(calls[0]['function']['arguments']),{'body':'<think>x</think>'})

    def test_stop_alias_zero_id_scope_and_unrelated_stops(self):
        t=Tokenizer();t.pieces[0]='<think>'
        self.assertEqual(glm_tag_safety.build_stops(t,[0,'END'],[11,99],[42],fixes_on=True),
                         (['END',99,42],[0,11]))
        with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1'}):
            self.assertFalse(glm_tag_safety.enabled('qwen3_coder'))
        self.assertEqual(glm_tag_safety.build_stops(t,[0,'END'],[11,99],[42],fixes_on=False),
                         ([0,'END',11,99,42],[]))

    async def test_history_exact_text_and_ordinary_ids_for_data_tags(self):
        literal='before </think> middle <think>x</think> after'
        reasoning='quoted `</think>` and nested <think>x</think>'
        histories=[
            [dict(role='assistant',content=literal),dict(role='user',content='next '+literal)],
            [dict(role='user',content=literal),dict(role='assistant',content=literal,
                  reasoning_content=reasoning,tool_calls=[dict(id='a',function=dict(name='f',arguments=json.dumps({'body':literal})))]),
             dict(role='tool',tool_call_id='a',content=literal)],
        ]
        for history in histories:
            for clear in (True,False):
                data=ChatCompletionRequest(messages=history,tools=[tool(props={'body':{'type':'string'}})],
                     chat_template_kwargs={'clear_thinking':clear})
                mc=Model(template=environment(Path(__file__).parents[1]/'glm53-chat_template.jinja'))
                with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1'}):
                    prompt,_=await namespace(mc)['apply_chat_template'](data)
                self.assertIn(literal,prompt)
                self.assertIn('<|assistant|><think>',prompt)
                self.assertTrue(prompt.endswith('<|assistant|><think>'))
                self.assertNotIn('__TABBY_LITERAL_',prompt)
                json.dumps(data.model_dump()) # No codecs/callables/markers in requests.
                t=Tokenizer();encoded=glm_tag_safety.encode_prompt(t,prompt,data,add_bos=True)
                self.assertEqual(t.decode(encoded.ids[1:]),prompt)
                self.assertEqual(encoded.ids.count(1),1)
                special=[i for i in encoded.ids if i in (10,11)]
                self.assertEqual(len(special),3,'Only the past outer pair and current opener are special')
                self.assertTrue(all(prompt[a:b] in glm_tag_safety.TAGS for a,b in data._glm_prompt_literals[1]))
                # Per-choice deep copies and forced continuations keep the spans.
                copied=data.model_copy(deep=True)
                extended=glm_tag_safety.encode_prompt(t,prompt+'</think>',copied)
                self.assertEqual(extended.ids[-1],11)

    async def test_actual_context_check_and_generation_encode_match(self):
        t=Tokenizer();codec=glm_tag_safety.PromptLiterals({})
        masked='<|assistant|><think>'+codec.mask('`<think>x</think>`')+'</think>'
        prompt,spans=codec.restore(masked);p=params();p._glm_prompt_literals=(prompt,spans)
        ns=load_methods();ns['encode_once_enabled']=lambda:False
        self_obj=NS(tokenizer=t,hf_model=NS(add_bos_token=lambda:False),max_seq_len=8192,
            cache=NS(max_num_tokens=8192),generator=NS(generator=NS(recurrent_cache=None)),
            job_max_rq_tokens=lambda _:4096)
        ns['validate_context_length'](self_obj,prompt,p)
        expected=glm_tag_safety.encode_prompt(t,prompt,p).ids
        generated=await ns['_encode_prompt'](self_obj,prompt,p,False,[])
        self.assertEqual(expected,generated.ids)
        # Cached path uses the exact protected IDs, without re-encoding.
        ns['encode_once_enabled']=lambda:True
        p._prompt_ids={(prompt,False):(weakref.ref(t),generated)}
        count=len(t.calls)
        cached=await ns['_encode_prompt'](self_obj,prompt,p,False,[])
        self.assertEqual(len(t.calls),count)
        self.assertEqual(cached.ids,expected)
        class UnsafeTokenizer(Tokenizer):
            def encode(self,text,**kw):
                kw['encode_special_tokens']=True
                return super().encode(text,**kw)
        unsafe=UnsafeTokenizer()
        protected=glm_tag_safety.encode_prompt(unsafe,prompt,p)
        self.assertEqual(unsafe.decode(protected.ids),prompt)
        self.assertEqual(protected.ids.count(10),1)
        self.assertEqual(protected.ids.count(11),1)

    async def test_history_prefix_and_continued_final_message(self):
        for continued in (False,True):
            mc=Model(template=environment(Path(__file__).parents[1]/'glm53-chat_template.jinja'))
            data=ChatCompletionRequest(messages=[dict(role='assistant',content='literal </think> tail')],
                 continue_final_message=continued,add_generation_prompt=not continued,response_prefix='prefix')
            with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1'}):
                prompt,_=await namespace(mc)['apply_chat_template'](data)
            self.assertIn('literal </think> tail',prompt)
            self.assertTrue(prompt.endswith('prefix'))
            self.assertEqual(data._glm_prompt_literals[0],prompt)

class LiteralReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_and_quoted_tags_survive_all_collectors(self):
        plans = [
            'Nested <think>x</think> then act.',
            'Each line is `row_3 = "a\\b \\"q\\" </fake> <think>x</think>"`.',
            'Only `<think>` in the file.',
            'Only `</think>` in the file.',
            'A quoted "</think>" and "<think>" string.',
            'Fence ```\n<think>x</think>\n``` ends here.',
        ]
        value='<think>x</think> only <think> and only </think>'
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],
             tools=[tool(props={'body':{'type':'string'}})])
        for plan in plans:
            raw=plan+'</think>'+call(pairs=[('body',value)])
            for width in (1,2,7,len(raw)):
                for streaming,live in ((False,False),(True,False),(True,True)):
                    result,frames=await collect(raw,data,streaming=streaming,live=live,width=width)
                    got=(''.join(f.get('delta_reasoning_content','') for f in frames)
                         if streaming else result['reasoning_content'])
                    self.assertEqual(got,plan,(plan,width,streaming,live))
                    calls=assembled(frames) if streaming else result['tool_calls']
                    self.assertEqual(json.loads(calls[0]['function']['arguments']),{'body':value})
                    self.assertEqual((frames[-1] if streaming else result)['finish_reason'],'tool_calls')

    async def test_scripted_special_ids_mid_reasoning_and_arguments(self):
        # Distinct IDs for both tags, plus actual ordinary text and real EOS.
        # The fake backend models stop-before-decode. The real collector parses its output.
        ids={10:'<think>',11:'</think>',12:'`',13:'plan ',14:'x',15:' done',
             16:call(pairs=[('body','PLACEHOLDER')]).split('PLACEHOLDER')[0],
             17:call(pairs=[('body','PLACEHOLDER')]).split('PLACEHOLDER')[1],99:'<|observation|>'}
        script=[13,12,10,14,11,12,15,11,16,10,14,11,17,99]
        class TokenModel(Model):
            async def stream_generate(self,*a,**kw):
                for tid in script:
                    if tid==99:
                        yield dict(text='',finish_reason='stop',eos_reason='stop_token',stop_str=ids[tid]);return
                    yield dict(text=ids[tid],token_ids=[tid],
                               glm_reasoning_controls=[(0,len(ids[tid]),ids[tid])] if tid in (10,11) else [])
        mc=TokenModel()
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],
             tools=[tool(props={'body':{'type':'string'}})])
        for streaming,live in ((False,False),(True,False),(True,True)):
            result,frames=await collect('',data,mc,streaming=streaming,live=live)
            reasoning=''.join(f.get('delta_reasoning_content','') for f in frames) if streaming else result['reasoning_content']
            self.assertEqual(reasoning,'plan `<think>x', 'Real closing ID ends reasoning even in quotes')
            calls=assembled(frames) if streaming else result['tool_calls']
            self.assertEqual(json.loads(calls[0]['function']['arguments']),{'body':'<think>x</think>'})

    async def test_content_tags_remain_text_after_real_boundary(self):
        result,_=await collect('plan</think>Example <think>x</think> done')
        self.assertEqual(result['reasoning_content'],'plan')
        self.assertEqual(result['content'],'Example <think>x</think> done')

if __name__=='__main__':unittest.main(verbosity=2)
