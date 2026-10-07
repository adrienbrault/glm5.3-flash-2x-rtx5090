"""CPU validation of the GLM grammar against the actual production tokenizer.

Requires tokenizers + llguidance (already used by TabbyAPI's engine). This does
not load model weights or a GPU. A nonzero exit means do not enable this patch.
"""
import argparse
import ctypes
from harness import *
from llguidance import LLMatcher, LLTokenizer, grammar_from
from tokenizers import Tokenizer

p=argparse.ArgumentParser()
p.add_argument('tokenizer_json');p.add_argument('--eos-id',type=int,required=True)
a=p.parse_args()
tok=Tokenizer.from_file(a.tokenizer_json)
ll=LLTokenizer(tok.to_str(),eos_token=[a.eos_id])
def ids(text):return tok.encode(text,add_special_tokens=False).ids
def single_id(text):
    encoded=ids(text)
    return encoded[0] if len(encoded)==1 else None
for tag in ('<tool_call>','</tool_call>','<arg_key>','</arg_key>','<arg_value>','</arg_value>','</think>'):
    print(tag,ids(tag))
assert single_id('</think>') is not None,'Backend cannot arm the filter without a single reasoning end token'
def allowed(m,token):
    bitmap=(ctypes.c_uint32*((ll.vocab_size+31)//32))()
    m.unsafe_compute_mask_ptr(ctypes.addressof(bitmap),ctypes.sizeof(bitmap))
    return bool(bitmap[token>>5] & (1<<(token&31)))
def accepts(g,text,eos=True):
    m=LLMatcher(ll,g,log_level=0)
    for token in ids(text):
        if not allowed(m,token) or not m.consume_token(token):return False
    return allowed(m,a.eos_id) if eos else True
with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1','TABBY_GLM_FORCING':'1'}):
    for parallel in (False,True):
        for selected in ('required',{'function':{'name':'f'}}):
            data=ChatCompletionRequest(messages=[],tools=[tool(),tool('g')],tool_choice=selected,parallel_tool_calls=parallel)
            forced=choice['prepare_tool_choice_forcing'](data,'glm4_5',single_id,'grammar probe')
            for grammar in (forced.grammar,forced.call_grammar):
                if grammar is None:continue
                g=grammar_from('lark',grammar)
                error=LLMatcher.validate_grammar(g,ll)
                assert not error,error
                assert not accepts(g,''),'EOS admitted without call'
                for raw in (call(),call(pairs=[('s','')]),call(pairs=[('s','Paris')]),
                            call(pairs=[('s','a < b && c > d, "quoted"\nline')])):
                    assert accepts(g,raw),(grammar,raw)
                assert not accepts(g,call('unknown'))
                assert not accepts(g,'<tool_call>f<arg_key>x</arg_key><arg_value>unclosed')
                assert accepts(g,call()+call('g'))==(parallel and selected=='required')
                assert accepts(g,call('g'))==(selected=='required')
                assert not accepts(g,call()+'visible answer after call')
print('production tokenizer grammar probe passed')
