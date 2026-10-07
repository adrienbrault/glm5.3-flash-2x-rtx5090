"""Client matrix, trace reconstruction, and metadata probes; no HTTP/model needed."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import tempfile

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
g=module('toolcheck',Path(__file__).parents[1]/'glm53_toolcheck.py')
probe=module('model_probe',Path(__file__).parent/'probe_model.py')

class ToolcheckTests(unittest.TestCase):
    def test_metadata_keeps_eos_provenance_and_special_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for name,data in {
                'config.json':{'eos_token_id':[99]},
                'generation_config.json':{'eos_token_id':[10,99]},
                'tokenizer_config.json':{'eos_token':{'content':'</think>'},
                    'added_tokens_decoder':{'0':{'content':'<think>','special':True},'11':{'content':'</think>','special':True}}},
                'tokenizer.json':{'model':{'vocab':{'<think>':10}},'added_tokens':[]},
            }.items():(directory/name).write_text(json.dumps(data))
            result=probe.metadata(directory)
        self.assertEqual(result['tag_ids']['<think>'],0)
        self.assertEqual(result['think_in_eos_sources']['tokenizer_config_eos_token'],[11])
        self.assertEqual(result['eos_sources']['generation_config'],[10,99])
        self.assertEqual(len(result['files_sha256']),4)

    def test_complete_matrix_and_default_temperature_omission(self):
        a=NS(model='synthetic',cases=list(g.CASES),modes=['stream','plain'],temperatures=['zero','default'],
             repeats=5,max_tokens=12000,long_max_tokens=40000)
        requests=list(g.request_matrix(a));self.assertEqual(len(requests),100)
        self.assertEqual(len({tag for tag,_,_ in requests}),100)
        for tag,case,body in requests:
            self.assertEqual(body.get('temperature'),0 if '-zero-' in tag else None)
            self.assertEqual(body['max_tokens'],40000 if case=='1000-line' else 12000)
            self.assertEqual(len(g.expected_lines(case)),20 if case=='simple' else 1000 if case=='1000-line' else 200)

    def test_single_tag_cases_and_exact_escaping(self):
        opened=g.expected_lines('think-only')[0];closed=g.expected_lines('close-only')[0]
        self.assertIn('<think>',opened);self.assertNotIn('</think>',opened)
        self.assertIn('</think>',closed);self.assertNotIn('<think>',closed)
        self.assertIn('a\\\\b \\"q\\"',g.expected_lines('hard')[0])

    def test_stream_exact_body_and_fragmented_json(self):
        content='\n'.join(g.expected_lines('close-only'))+'\n'
        args=json.dumps({'path':'/tmp/y.py','content':content})
        lines=[]
        for i in range(0,len(args),3):
            lines.append([0,'data: '+json.dumps({'choices':[{'index':0,'delta':{'tool_calls':[
                {'index':0,'function':{'name':'write_file' if i==0 else None,'arguments':args[i:i+3]}}]}}]})])
        lines.extend([[0,'data: '+json.dumps({'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]})],
                      [0,'data: [DONE]']])
        summary=g.summarize(lines,True,'close-only')
        self.assertTrue(summary['passed']);self.assertTrue(summary['calls'][0]['exact_body'])
        self.assertEqual(summary['calls'][0]['close_think_count'],200)

    def test_valid_json_with_substituted_tags_fails(self):
        content='\n'.join(g.expected_lines('hard')).replace('<think>x</think>','⓾x')
        raw=[[0,json.dumps({'choices':[{'message':{'tool_calls':[{'function':{
            'name':'write_file','arguments':json.dumps({'path':'/tmp/y.py','content':content})}}]},'finish_reason':'tool_calls'}]})]]
        result=g.summarize(raw,False,'hard')
        self.assertTrue(result['calls'][0]['args_json_ok']);self.assertFalse(result['passed'])
        self.assertEqual(result['calls'][0]['first_mismatch_line'],1)

    def test_stop_without_call_does_not_pass(self):
        raw=[[0,'data: '+json.dumps({'choices':[{'index':0,'delta':{'reasoning_content':'`literal '}}]})],
             [0,'data: '+json.dumps({'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})]]
        result=g.summarize(raw,True,'hard');self.assertFalse(result['passed']);self.assertEqual(result['calls'],[])

if __name__=='__main__':unittest.main(verbosity=2)
