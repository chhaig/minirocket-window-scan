"""Bare-metal timing on the Mac (run outside any VM). 2026-11-24.
Compares, single thread, on all 19 HAPT test recordings, three axes, W=256:
  aeon  : extracted aeon 1.3.0 MiniRocket numba core on copied crops (the manuscript's N)   [needs numba]
  cnat  : C re-implementation of the same crop algorithm (kern.c: native)
  scan  : new boundary-free causal-context scan (kern.c: scan) + level features
Each timing runs from in-memory samples to class scores (random fixed 12-class linear head, same for all).
Usage: python bench_mac.py PROCESS_INDEX      -> writes mac-results/process-<i>.json
       python bench_mac.py summarize          -> writes mac-results/summary.json
Build first: python benchmark/build.py (both C kernels use strict floating point).
"""
import os,sys,json,time,re,ctypes,platform,statistics
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
H=Path(__file__).resolve().parent; BASE=H.parent;sys.path.insert(0,str(BASE));import padconv
OUT=H/'random-results';OUT.mkdir(exist_ok=True);W=256;HOPS=[int(v) for v in os.environ.get("BENCH_HOPS","8,64").split(",")];NLEV=32
def load():
    root=Path(os.environ.get('HAPT_DIR',str(BASE/'data/hapt')));test=set(np.loadtxt(root/'Test/subject_id_test.txt',dtype=int).tolist());recs=[]
    for p in sorted((root/'RawData').glob('acc_exp*_user*.txt')):
        if int(re.search(r'user(\d+)',p.name).group(1)) in test:recs.append(np.loadtxt(p,dtype=np.float32))
    assert len(recs)==19;return recs
def summarize():
    runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('process-*.json'))];assert runs
    s=dict(processes=len(runs),environment=runs[0]['environment'],rows=[])
    for h in HOPS:
        med={m:statistics.median(r['hops'][str(h)]['median'][m] for r in runs) for m in runs[0]['hops'][str(h)]['median']}
        row=dict(hop=h,windows=runs[0]['hops'][str(h)]['windows'],seconds=med,equal_counts=all(r['hops'][str(h)]['cnat_equals_aeon'] is True for r in runs))
        for a in ('aeon','cnat'):
            if a in med:row[f'{a}/scan']=med[a]/med['scan'];row[f'{a}/scan_range']=[min(r['hops'][str(h)]['median'][a]/r['hops'][str(h)]['median']['scan'] for r in runs),max(r['hops'][str(h)]['median'][a]/r['hops'][str(h)]['median']['scan'] for r in runs)]
        s['rows'].append(row)
    (OUT/'summary.json').write_text(json.dumps(s,indent=1));print(json.dumps(s,indent=1))
if sys.argv[1]=='summarize':summarize();sys.exit(0)
proc=int(sys.argv[1]);
if proc>=0:
    validation=json.loads((H/'validation.json').read_text()); assert validation['passed'] is True
    for f,digest in validation['source_sha256'].items():
        import hashlib
        assert hashlib.sha256((H.parent/f).read_bytes()).hexdigest()==digest, ('changed since validation',f)
recs=load()
L=ctypes.CDLL(str(H/'kern.so'));fp=np.ctypeslib.ndpointer(np.float32,flags='C');ip=np.ctypeslib.ndpointer(np.int32,flags='C')
L.native.argtypes=[fp,ctypes.c_int,ctypes.c_int,ip,ip,ctypes.c_int,fp,ip,fp,ctypes.c_int]
L.scan.argtypes=[fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,ip,ctypes.c_int,fp,ip,fp,ctypes.c_int]
dil,nfd=padconv.fit_dilations(W);dil=dil.astype(np.int32);nfd=nfd.astype(np.int32);F=int(84*nfd.sum());idx2=padconv.INDICES.astype(np.int32);idx=idx2.ravel().copy()
# fixed parameters: biases from real responses (one random window per kernel/dilation), level thresholds, linear head
rng=np.random.RandomState(0);bias=[]
for c in range(3):
    x0=np.ascontiguousarray(recs[0][:,c]);_,_,b=padconv.transform([x0],[np.arange(0,len(x0)-W+1,128)],W,dil,nfd,None,False,rng);bias.append(b)
thr=[np.quantile(recs[0][:,c],(np.arange(NLEV)+0.5)/NLEV).astype(np.float32) for c in range(3)]
coefN=rng.standard_normal((3*F,12)).astype(np.float32);coefS=np.concatenate([coefN,rng.standard_normal((3*NLEV,12)).astype(np.float32)])
try:
    from numba import njit,prange,vectorize,set_num_threads
    set_num_threads(1);ns=dict(njit=njit,prange=prange,vectorize=vectorize,np=np)
    exec((H/'native_core.py').read_text(),ns);aeon_core=ns['_static_transform_uni']
    params=[(None,None,dil,nfd,bias[c]) for c in range(3)]
except ImportError:
    aeon_core=None
def crops(x,ss):return np.array([x[s:s+W].T for s in ss],np.float32)          # copy, as in the manuscript's N
def run_aeon(x,h):
    ss=np.arange(0,len(x)-W+1,h);win=crops(x,ss)
    z=np.concatenate([aeon_core(np.ascontiguousarray(win[:,c,:]),params[c],idx2) for c in range(3)],axis=1);return z,z@coefN
def run_cnat(x,h):
    ss=np.arange(0,len(x)-W+1,h);win=crops(x,ss);z=np.empty((len(ss),3*F),np.float32)
    for c in range(3):
        o=np.empty((len(ss),F),np.float32);L.native(np.ascontiguousarray(win[:,c,:]),len(ss),W,dil,nfd,len(dil),bias[c],idx,o,F);z[:,c*F:(c+1)*F]=o
    return z,z@coefN
def run_scan(x,h):
    M=(len(x)-W)//h+1;ss=np.arange(M)*h;z=np.empty((M,3*F+3*NLEV),np.float32)
    for c in range(3):
        xc=np.ascontiguousarray(np.concatenate([x[:,c],np.zeros(4,np.float32)]));o=np.empty((M,F),np.float32)
        L.scan(xc,len(x),M,h,W,dil,nfd,len(dil),bias[c],idx,o,F);z[:,c*F:(c+1)*F]=o
        pre=np.zeros((len(x)+1,NLEV),np.int32);np.cumsum(x[:,c:c+1]>thr[c][None,:],0,out=pre[1:]);z[:,3*F+c*NLEV:3*F+(c+1)*NLEV]=(pre[ss+W]-pre[ss])/np.float32(W)
    return z,z@coefS
methods={'cnat':run_cnat,'scan':run_scan}
if aeon_core is not None:methods['aeon']=run_aeon
names=list(methods);res=dict(process=proc,environment=dict(python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),numpy=np.__version__,numba=(__import__('numba').__version__ if aeon_core else None),load=os.getloadavg()),hops={})
for h in HOPS:
    for m in names:methods[m](recs[0][:2000],h)                               # warm-up / JIT
    eq=None
    if aeon_core is not None:
        za,_=run_aeon(recs[3][:6000],h);zc,_=run_cnat(recs[3][:6000],h);eq=bool(np.array_equal(za,zc)); assert eq, 'Exact aeon/cnat feature mismatch'
    totals={m:[0.,0.,0.] for m in names};nw=0
    for ri,x in enumerate(recs):
        for rep in range(3):
            sh=(ri+rep+proc)%len(names)
            for m in names[sh:]+names[:sh]:
                t=time.perf_counter();z,s=methods[m](x,h);totals[m][rep]+=time.perf_counter()-t
                assert np.isfinite(z).all() and np.isfinite(s).all(), (m,ri,h,'nonfinite output')
        nw+=len(s)
    res['hops'][str(h)]=dict(windows=nw,seconds=totals,median={m:statistics.median(v) for m,v in totals.items()},cnat_equals_aeon=eq)
    print(h,res['hops'][str(h)]['median'],flush=True)
(OUT/f'process-{proc}.json').write_text(json.dumps(res,indent=1))
