"""Explicit CPU-only test adapter; never imported by production or default operator tests.

Supports only the tensor operations used in state/byte-observer tests. It is not a
PyTorch numerical or extension substitute; the Docker build reruns tests with real torch.
"""
from types import SimpleNamespace
import numpy as np

def raw(value):
    if isinstance(value,Tensor):return value.a
    if isinstance(value,tuple):return tuple(raw(v) for v in value)
    return value

class Tensor:
    def __init__(self,a,device='cpu'):self.a=np.asarray(a);self.device=SimpleNamespace(type=str(device))
    @property
    def dtype(self):return self.a.dtype
    @property
    def shape(self):return self.a.shape
    @property
    def ndim(self):return self.a.ndim
    def __len__(self):return len(self.a)
    def __getitem__(self,key):return Tensor(self.a[raw(key)])
    def __setitem__(self,key,value):self.a[raw(key)]=raw(value)
    def __add__(self,v):return Tensor(self.a+raw(v))
    __radd__=__add__
    def __sub__(self,v):return Tensor(self.a-raw(v))
    def __mul__(self,v):return Tensor(self.a*raw(v))
    def __mod__(self,v):return Tensor(self.a%raw(v))
    def __floordiv__(self,v):return Tensor(self.a//raw(v))
    def __eq__(self,v):return Tensor(self.a==raw(v))
    def __ne__(self,v):return Tensor(self.a!=raw(v))
    def __bool__(self):return bool(self.a)
    def numel(self):return self.a.size
    def element_size(self):return self.a.dtype.itemsize
    def cpu(self):return self
    def detach(self):return self
    def clone(self):return Tensor(self.a.copy(),self.device.type)
    def zero_(self):self.a.fill(0);return self
    def fill_(self,value):self.a.fill(value);return self
    def view(self,*shape):
        if len(shape)==1 and isinstance(shape[0],np.dtype):return Tensor(self.a.view(shape[0]))
        return Tensor(self.a.reshape(*shape))
    reshape=view
    def flatten(self):return Tensor(self.a.flatten())
    def contiguous(self):return Tensor(np.ascontiguousarray(self.a))
    def float(self):return Tensor(self.a.astype(np.float32))
    def long(self):return Tensor(self.a.astype(np.int64))
    def int(self):return Tensor(self.a.astype(np.int32))
    def to(self,dtype):
        return Tensor(self.a.astype(dtype)) if isinstance(dtype,np.dtype) else self
    def abs(self):return Tensor(np.abs(self.a))
    def max(self):return Tensor(self.a.max())
    def argmax(self):return Tensor(self.a.argmax())
    def sum(self,dim=None):return Tensor(self.a.sum(axis=dim))
    def item(self):return self.a.item()
    def unsqueeze(self,dim):return Tensor(np.expand_dims(self.a,dim))
    def dim(self):return self.a.ndim
    def data_ptr(self):return self.a.__array_interface__['data'][0]
    def numpy(self):return self.a

class Torch:
    Tensor=Tensor
    half=np.dtype('float16');int16=np.dtype('int16');uint8=np.dtype('uint8');int32=np.dtype('int32')
    float16=half
    def zeros(self,shape,dtype=half,device='cpu'):return Tensor(np.zeros(shape,dtype=dtype),device)
    def empty(self,shape,dtype=half):return Tensor(np.empty(shape,dtype=dtype))
    def zeros_like(self,t,device=None):return Tensor(np.zeros_like(t.a),device or t.device.type)
    def arange(self,*args,device=None):return Tensor(np.arange(*map(raw,args)))
    def randn(self,shape,dtype=np.dtype('float32'),device='cpu'):return Tensor(np.random.randn(*shape).astype(dtype),device)
    def randperm(self,n):return Tensor(np.random.permutation(n))
    def tensor(self,v,dtype=None,device='cpu'):return Tensor(np.array(v,dtype=dtype),device)
    def frombuffer(self,buf,dtype,count,offset):return Tensor(np.frombuffer(buf,dtype=dtype,count=count,offset=offset))
    def manual_seed(self,n):np.random.seed(n)
    def equal(self,a,b):return bool(np.array_equal(a.a,b.a))
    def all(self,t):return bool(np.all(t.a))
    def count_nonzero(self,t):return Tensor(np.count_nonzero(t.a))
    def nan_to_num(self,t,**kw):return Tensor(np.nan_to_num(t.a,**kw))
    def softmax(self,t,dim):
        v=t.a-t.a.max(axis=dim,keepdims=True);e=np.exp(v)
        return Tensor(e/e.sum(axis=dim,keepdims=True))

torch=Torch()
