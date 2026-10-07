import asyncio
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from harness import *

TEMPLATE=Path(os.environ.get('TOOLFIX_TEMPLATE',Path(__file__).parents[1]/'glm53-chat_template.jinja'))
BASE=Path(os.environ.get('TOOLFIX_BASE','/tmp/tabby-toolfix-base'))

class HistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self): self.mc=Model(template=environment(TEMPLATE))
    async def render(self,messages,enabled=True,**kwargs):
        data=ChatCompletionRequest(messages=messages,tools=[tool()],**kwargs)
        with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1' if enabled else '0'}):
            prompt,_=await namespace(self.mc)['apply_chat_template'](data)
        return prompt
    def history(self,content=None,reasoning=None,args='{}'):
        assistant=dict(role='assistant',content=content,tool_calls=[dict(id='call_a',function=dict(name='f',arguments=args))])
        if reasoning is not None: assistant['reasoning_content']=reasoning
        return [dict(role='user',content='go'),assistant,dict(role='tool',tool_call_id='call_a',content='done')]
    async def test_null_not_a_tabby_bug(self):
        for content in (None,''):
            for flag in (False,True): self.assertNotIn('None<tool_call>',await self.render(self.history(content),flag))
        raw=self.history();raw[1]["tool_calls"][0]["function"]["arguments"]={}
        direct=await environment(TEMPLATE).render_async(messages=raw,tools=[])
        self.assertIn("None<tool_call>",direct)
    async def test_reasoning_and_clear_thinking(self):
        for clear in (True,False):
            prompt=await self.render(self.history(reasoning='plan'),chat_template_kwargs={'clear_thinking':clear})
            self.assertIn('<think>plan</think>',prompt)
            msgs=self.history(reasoning='plan')+[dict(role='user',content='next')]
            prompt=await self.render(msgs,chat_template_kwargs={'clear_thinking':clear})
            self.assertEqual('<think>plan</think>' in prompt,not clear)
    async def test_empty_arguments_red_on_baseline(self):
        with self.assertRaises(HTTPException): await self.render(self.history(args=''),False)
        self.assertIn('<tool_call>f</tool_call>',await self.render(self.history(args='')))
    async def test_cached_reasoning_real_two_turns(self):
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],tools=[tool()])
        self.mc.chunks=['plan</think>'+call()]
        result,_=await collect('',data,self.mc)
        hist=self.history();hist[1]['tool_calls']=result['tool_calls']
        # Client renumbers ids; canonical args and full visible prefix still match.
        hist[1]['tool_calls'][0]['id']='client_0';hist[2]['tool_call_id']='client_0'
        self.assertIn('<think>plan</think>',await self.render(hist))
        hist[1]['reasoning_content']='own'
        self.assertIn('<think>own</think>',await self.render(hist))
        hist[1]['reasoning_content']=''
        self.assertNotIn('<think>plan</think>',await self.render(hist))
        del hist[1]['reasoning_content'];hist[0]['content']='different'
        self.assertNotIn('<think>plan</think>',await self.render(hist))
    async def test_streaming_reasoning_cache(self):
        self.mc.chunks=list('plan'+call())
        _,frames=await collect('',mc=self.mc,streaming=True,live=True)
        hist=self.history();hist[1]['tool_calls']=assembled(frames)
        hist[2]['tool_call_id']=hist[1]['tool_calls'][0]['id']
        self.assertIn('<think>plan</think>',await self.render(hist))
    async def test_inline_and_alias_reasoning(self):
        self.assertIn('<think>inline</think>',await self.render(self.history('<think>inline</think>visible')))
        h=self.history();h[1]['reasoning']='alias'
        self.assertIn('<think>alias</think>',await self.render(h))
        self.assertNotIn('<think>alias</think>',await self.render(h,False))
        self.assertNotIn('reasoning',ChatCompletionMessage(role='assistant',reasoning='alias').model_dump())
    async def test_tool_results_no_loss_and_order(self):
        h=self.history();h[1]['tool_calls'].append(dict(id='call_b',function=dict(name='f',arguments='{}')))
        h[2:]=[dict(role='tool',tool_call_id='call_b',content='B'),dict(role='tool',tool_call_id='call_a',content='A')]
        prompt=await self.render(h);self.assertLess(prompt.index('<tool_response>A'),prompt.index('<tool_response>B'))
        h[2]['tool_call_id']='unknown'
        self.assertNotIn('<tool_response>B',await self.render(h,False))
        prompt=await self.render(h);self.assertIn('<tool_response>B',prompt);self.assertIn('<tool_response>A',prompt)
    async def test_none_named_thinking_prompt(self):
        h=[dict(role='user',content='go')]
        self.assertIn('# Tools',await self.render(h,False,tool_choice='none'))
        self.assertNotIn('# Tools',await self.render(h,tool_choice='none'))
        self.assertTrue((await self.render(h,enable_thinking=False)).endswith('<think></think>'))
        self.assertTrue((await self.render(h,False,enable_thinking=False)).endswith('<think>'))
    async def test_named_prompt_filters_signatures(self):
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],tools=[tool(),tool('g')],
                                   tool_choice={'function':{'name':'f'}})
        with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1'}):
            prompt,_=await namespace(self.mc)['apply_chat_template'](data)
        self.assertIn('"name": "f"',prompt);self.assertNotIn('"name": "g"',prompt)

    async def test_off_prompt_equivalence(self):
        if not BASE.exists():self.skipTest('Provide TOOLFIX_BASE')
        for h in (self.history(), self.history(reasoning='own'),self.history('<think>inline</think>v')):
            data=ChatCompletionRequest(messages=h,tools=[tool()])
            with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'0'}):
                base,_=await namespace(self.mc,BASE)['apply_chat_template'](data.model_copy(deep=True))
                result,_=await namespace(self.mc)['apply_chat_template'](data.model_copy(deep=True))
            self.assertEqual(base,result)
        # Unknown message reasoning was ignored in the original; accepting its
        # arbitrary shape must not introduce a default-off validation failure.
        self.assertEqual(ChatCompletionMessage(reasoning={'opaque':True}).model_dump(exclude_none=True),{'role':'user'})

    def test_memory_bounds_and_model_isolation(self):
        m=fixes.ReasoningMemory(entries=2,byte_limit=5)
        m.put('a','é');m.put('b','b');self.assertEqual(m.get('a'),'é');m.put('c','cc')
        self.assertIsNone(m.get('b'));self.assertLessEqual(m.size,5)
        m.put('oversized','abcdef');self.assertIsNone(m.get('oversized'))
        self.assertIsNot(fixes.memory(Model()),fixes.memory(Model()))

class ParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_unions_and_string(self):
        props={'s':{'type':'string'},'i':{'type':['integer','null']},
               'n':{'anyOf':[{'type':'integer'},{'type':'null'}]},
               'b':{'oneOf':[{'type':'boolean'},{'type':'null'}]},'o':{'type':'object'}}
        raw=call(pairs=[('s','123'),('i','null'),('n','"5"'),('b','true'),('o','{"x":[1]}')])
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],tools=[tool(props=props)])
        expected={'s':'123','i':None,'n':5,'b':True,'o':{'x':[1]}}
        for width in (1,7,len(raw)):
            result,_=await collect(raw,data,width=width)
            self.assertEqual(json.loads(result['tool_calls'][0]['function']['arguments']),expected)
            for live in (False,True):
                _,frames=await collect(raw,data,streaming=True,live=live,width=width)
                self.assertEqual(json.loads(assembled(frames)[0]['function']['arguments']),expected)
                self.assertEqual(frames[-1]['finish_reason'],'tool_calls')
    def test_type_values(self):
        cases=[('123',{'type':['string','integer']},'123'),('false',{'type':['null','boolean']},False),
               ('3.0',{'type':['integer','null']},3),('true',{'type':'integer'},'true'),
               ('null',{'anyOf':[{'type':'string'},{'type':'null'}]},None),
               ('NaN',{},'NaN'),('x',{'enum':['x','y']},'x'),('null',{'const':None},None)]
        for raw,schema,want in cases:self.assertEqual(fixes.typed_value(raw,schema),want)
    async def test_string_whitespace_and_live_body(self):
        body='  # file\n    line \n' * 200
        raw=call(pairs=[('body',body)])
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],tools=[tool(props={'body':{'type':'string'}})])
        result,_=await collect(raw,data)
        self.assertEqual(json.loads(result['tool_calls'][0]['function']['arguments'])['body'],body)
        _,frames=await collect(raw,data,streaming=True,live=True)
        self.assertEqual(json.loads(assembled(frames)[0]['function']['arguments'])['body'],body)
        close_at=raw.index('</arg_value>')
        before=''.join(d['function'].get('arguments','') for f in frames[:close_at] for d in f.get('delta_tool_calls') or [])
        self.assertGreater(len(before),len(body))

    async def test_choice_call_count_names_and_no_tools(self):
        raw=call('g')+call('f')+call('f')
        for streaming in (False,True):
            for live in (False,True):
                for which in ('auto','none',{'type':'function','function':{'name':'f'}}):
                    data=ChatCompletionRequest(messages=[dict(role='user',content='go')],tools=[tool(),tool('g')],
                                               tool_choice=which,parallel_tool_calls=False)
                    result,frames=await collect(raw,data,streaming=streaming,live=live)
                    calls=assembled(frames) if streaming else result['tool_calls']
                    self.assertEqual([c['function']['name'] for c in calls],[] if which=='none' else ['g' if which=='auto' else 'f'])
        result,_=await collect(call(),ChatCompletionRequest(messages=[dict(role='user',content='go')]))
        self.assertEqual(result['tool_calls'],[]);self.assertEqual(result['finish_reason'],'stop')
    async def test_unclosed_thinking_and_truncation(self):
        for enabled in (False,True):
            for streaming in (False,True):
                for live in (False,True):
                    raw='plan'+call()
                    result,frames=await collect(raw,streaming=streaming,live=live,enabled=enabled)
                    calls=assembled(frames) if streaming else result['tool_calls']
                    self.assertEqual(len(calls),1)
        raw='<tool_call>f<arg_key>x</arg_key><arg_value>unfinished'
        for live in (False,True):
            _,frames=await collect(raw,mc=Model(list(raw),eos='max_new_tokens'),streaming=True,live=live)
            self.assertEqual(frames[-1]['finish_reason'],'length')
        mc=Model(list('example '+call()));mc.tool_calls_in_reasoning=False
        result,_=await collect('',mc=mc);self.assertEqual(result['tool_calls'],[])
    async def test_malformed_and_unknown(self):
        raw=call()+ '<tool_call>f<arg_key>x</arg_key></tool_call>'
        result,_=await collect(raw);self.assertEqual(result['tool_calls'],[])
        self.assertEqual(result['content'],raw);self.assertEqual(result['finish_reason'],'stop')
        for raw in (call('unknown'),'<tool_call>f<arg_value>x</arg_value></tool_call>',
                    '<tool_call>f<arg_key>x</tool_call>',
                    '<tool_call>f<arg_key>x</arg_key><arg_value>missing end</tool_call>'):
            result,_=await collect(raw);self.assertFalse(result['tool_calls']);self.assertEqual(result['finish_reason'],'stop')
    async def test_off_matches_overlay(self):
        if not BASE.exists():self.skipTest('Provide TOOLFIX_BASE for differential test')
        for raw in ('plan</think>answer',call(),call(pairs=[('s','123')]),'<tool_call>f<arg_key>x</arg_key>'):
            for live in (False,):
                for streaming in (False,True):
                    base,bframes=await collect(raw,enabled=False,app=BASE,streaming=streaming,live=live)
                    result,frames=await collect(raw,enabled=False,streaming=streaming,live=live)
                    def scrub(value):
                        if isinstance(value,dict):return {k:([] if k=='delta_tool_calls' and not v else scrub(v)) for k,v in value.items() if k!='id'}
                        if isinstance(value,(list,tuple)):return [scrub(v) for v in value]
                        return value
                    self.assertEqual(scrub((result,frames)),scrub((base,bframes)))

class ChoiceTests(unittest.TestCase):
    def test_glm_forcing_and_gate(self):
        with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1','TABBY_GLM_FORCING':'1'}):
            data=ChatCompletionRequest(messages=[],tools=[tool(),tool('g')],tool_choice={'function':{'name':'f'}})
            forced=choice['prepare_tool_choice_forcing'](data,'glm4_5',lambda t:None,'test')
            self.assertIn('"f"',forced.grammar);self.assertNotIn('"g"',forced.grammar)
            self.assertIn('param:',forced.grammar);self.assertNotIn('(WS? call)*',forced.grammar)
            forced=choice['prepare_tool_choice_forcing'](data,'glm4_5',lambda t: {'<tool_call>':10,'</tool_call>':11}.get(t),'test')
            self.assertIn('<[10]>',forced.grammar);self.assertIsNotNone(forced.call_grammar)
            data.tool_choice='required';data.parallel_tool_calls=True
            forced=choice['prepare_tool_choice_forcing'](data,'glm4_5',lambda t:None,'test')
            self.assertIn('(WS? call)*',forced.grammar)
            data.parallel_tool_calls=False
            self.assertNotIn('(WS? call)*',choice['prepare_tool_choice_forcing'](data,'glm4_5',lambda t:None,'test').grammar)
        with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'0'}):
            self.assertIsNone(choice['prepare_tool_choice_forcing'](data,'glm4_5',lambda t:None,'test'))
    def test_invalid_named_choice(self):
        data=ChatCompletionRequest(messages=[],tools=[tool()],tool_choice={'function':{'name':'missing'}})
        with self.assertRaises(HTTPException) as e:choice['validate_tool_choice'](data)
        self.assertEqual(e.exception.status_code,400)
    def test_exact_scope(self):
        for value in ('','0','true','01','1'):
            with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':value}):
                self.assertEqual(fixes.enabled('glm4_5'),value=='1')
                self.assertFalse(fixes.enabled('qwen3_coder'))
                self.assertFalse(fixes.enabled('glm4_7'))

if __name__=='__main__':unittest.main(verbosity=2)
