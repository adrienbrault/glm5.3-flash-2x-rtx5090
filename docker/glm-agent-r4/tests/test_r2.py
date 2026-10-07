"""Revision-2 regressions through actual models, channel parser and collector."""
import unittest
from harness import *

class R2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_literal_reasoning_tags_exact_in_all_modes(self):
        value=' \n# <think>x</think> then </think><think>\n  true null 123 "é" \\ \t\x00\n '
        raw=call(pairs=[('body',value)])
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],
            tools=[tool(props={'body':{'type':'string'}})])
        for width in (1,2,7,len(raw)):
            for streaming,live in ((False,False),(True,False),(True,True)):
                result,frames=await collect(raw,data,streaming=streaming,live=live,width=width)
                calls=assembled(frames) if streaming else result['tool_calls']
                self.assertEqual(json.loads(calls[0]['function']['arguments']),{'body':value})
                end=frames[-1] if streaming else result
                self.assertEqual(end['finish_reason'],'tool_calls')
        # Default-off retains the inherited deletion and string trimming.
        result,_=await collect(raw,data,enabled=False)
        self.assertNotIn('<think>',result['tool_calls'][0]['function']['arguments'])

    async def test_literal_tags_do_not_switch_outer_reasoning_state(self):
        raw='plan'+call(pairs=[('body','<think>x</think>')])+'still thinking</think>Answer'
        data=ChatCompletionRequest(messages=[dict(role='user',content='go')],
            tools=[tool(props={'body':{'type':'string'}})])
        result,_=await collect(raw,data)
        self.assertEqual(result['reasoning_content'],'planstill thinking')
        self.assertEqual(result['content'],'Answer')
        self.assertEqual(json.loads(result['tool_calls'][0]['function']['arguments']),{'body':'<think>x</think>'})

    async def test_raw_fallback_matches_nonstream_and_buffered_stream(self):
        invalids=[
            '<tool_call>f<arg_key>body</arg_key><arg_value>half <',
            call()+'<tool_call>f<arg_key>body</arg_key></tool_call>',
            call(pairs=[('body','first'),('body','second')]),
            '<tool_call>f<arg_key>body</arg_key><arg_value>missing close</tool_call>',
            '<tool_call',
            call('unknown'),
            '<tool_call>f<arg_value>unpaired</arg_value></tool_call>',
        ]
        for raw in invalids:
            for eos in ('eos','max_new_tokens'):
                for width in (1,len(raw)):
                    want='length' if eos=='max_new_tokens' else 'stop'
                    for streaming,live in ((False,False),(True,False),(True,True)):
                        result,frames=await collect(raw,mc=Model([raw[i:i+width] for i in range(0,len(raw),width)],eos=eos),
                            streaming=streaming,live=live)
                        if streaming:
                            self.assertEqual(frames[-1]['finish_reason'],want,(raw,streaming,live))
                            content=''.join(f.get('delta_content','') for f in frames)
                        else:
                            self.assertEqual(result['finish_reason'],want)
                            self.assertEqual(result['tool_calls'],[])
                            content=(result['content'] or '')+(result['reasoning_content'] or '')
                        # A partial start tag remains in the original channel.
                        self.assertEqual(content,raw)

    async def test_valid_then_partial_start_tag_is_not_success(self):
        raw=call()+'<tool_call'
        _,frames=await collect(raw,streaming=True,live=True)
        self.assertEqual(frames[-1]['finish_reason'],'stop')
        self.assertEqual(''.join(f.get('delta_content','') for f in frames),raw)

    async def test_duplicate_turn_never_cached(self):
        mc=Model(list('plan'+call()+call(pairs=[('x','one'),('x','two')])))
        _,frames=await collect('',mc=mc,streaming=True,live=True)
        self.assertEqual(frames[-1]['finish_reason'],'stop')
        memory=getattr(mc,'_glm_tool_reasoning',None)
        self.assertTrue(memory is None or not memory.items)

    async def test_all_flags_off_matches_pristine(self):
        base=Path(os.environ['TOOLFIX_BASE'])
        cases=['plan</think>answer',call(),call(pairs=[('body','<think>x</think>')]),
            call(pairs=[('null','x')]),call(pairs=[('body','1'),('body','2')]),
            call()+'<tool_call>f<arg_key>x</arg_key>',
            '<tool_call>f<arg_key>null</arg_key><arg_value>x</arg_value><arg_value>y</arg_value></tool_call>']
        def scrub(value):
            if isinstance(value,dict):
                return {k:([] if k=='delta_tool_calls' and not v else scrub(v)) for k,v in value.items() if k!='id'}
            if isinstance(value,(list,tuple)):return [scrub(v) for v in value]
            return value
        with patch.dict(os.environ,{},clear=True):
            for raw in cases:
                for streaming in (False,True):
                    a=await collect(raw,enabled=False,live=False,streaming=streaming)
                    b=await collect(raw,enabled=False,live=False,streaming=streaming,app=base)
                    self.assertEqual(scrub(a),scrub(b))

class ForcingGateTests(unittest.TestCase):
    def test_independent_forcing_gate(self):
        data=ChatCompletionRequest(messages=[],tools=[tool()],tool_choice='required')
        for fixes_on in ('0','1'):
            for forcing_on in ('','0','true','01','1'):
                with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':fixes_on,'TABBY_GLM_FORCING':forcing_on}):
                    want=fixes_on=='1' and forcing_on=='1'
                    self.assertEqual(fixes.forcing_enabled('glm4_5'),want)
                    self.assertEqual(choice['supports_forcing']('glm4_5'),want)
                    result=choice['prepare_tool_choice_forcing'](data,'glm4_5',lambda t:None,'test')
                    self.assertEqual(result is not None,want)
                    self.assertFalse(fixes.forcing_enabled('qwen3_coder'))
        with patch.dict(os.environ,{},clear=True):
            self.assertFalse(fixes.enabled('glm4_5'))
            self.assertFalse(fixes.forcing_enabled('glm4_5'))
            self.assertFalse(stream_toolcalls_enabled('glm4_5',True))

if __name__=='__main__':unittest.main(verbosity=2)
