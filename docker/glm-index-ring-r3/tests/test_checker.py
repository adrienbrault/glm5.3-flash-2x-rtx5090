#!/usr/bin/env python3
import unittest
from check_shadow import check,REQUIRED,KINDS
import json

def summary(reason='idle',mismatches=0,omit=None):
    cov={k:1 for k in REQUIRED if k!=omit};ks={k:1 for k in KINDS}
    layers={f'model.layers.{i}.self_attn':1 for i in range(11)}
    layers['model.layers.45.self_attn']=1
    encode=lambda d:json.dumps(d,separators=(',',':'))
    return f'[RING-SHADOW] calls=123 compared_rows=234 mismatches={mismatches} layers=12 coverage={encode(cov)} kinds={encode(ks)} layer_calls={encode(layers)} reason={reason}'

class CheckerTests(unittest.TestCase):
    def test_pass_requires_all_paths(self):self.assertTrue(check([('engine',summary())])['passed'])
    def test_no_comparisons(self):self.assertFalse(check([('engine','healthy server')])['passed'])
    def test_missing_path(self):
        for k in REQUIRED:self.assertFalse(check([('engine',summary(omit=k))])['passed'],k)
    def test_mismatch_even_if_last_summary_clean(self):
        self.assertFalse(check([('engine',summary(mismatches=1)+'\n'+summary())])['passed'])
    def test_periodic_is_not_final(self):self.assertFalse(check([('engine',summary('periodic'))])['passed'])
    def test_missing_workload(self):self.assertFalse(check([('engine',summary())],[])['passed'])

if __name__=='__main__':unittest.main()
