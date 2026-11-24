"""End-to-end timing with TRAINED models on HAPT (run on the bare-metal host). 2026-11-24.
 aeonN : copied crops -> aeon 1.3.0 MiniRocket numba core with the trained N biases -> trained N classifier   [needs numba]
 scanPL: C scan (kern.so, symbol `scan`) with the trained P biases + level features -> trained P+L classifier
Both run on ALL windows of all 19 test recordings, one thread, from in-memory samples to predicted classes.
After timing, accuracy on labelled endpoints is computed from the timed predictions and must equal test-hop<H>.json.
Usage: KERN_SO=/path/kern.so python bench_trained.py PROCESS ; python bench_trained.py summarize
"""
import os,sys,json,time,re,ctypes,platform,statistics
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from pathlib import Path
D=Path(__file__).resolve().parent;H=D.parent/'models';sys.path.insert(0,str(H.parent));import padconv
OUT=D/'trained-results';OUT.mkdir(exist_ok=True);W=256;NLEV=32;NC=12;HOPS=[int(v) for v in os.environ.get('BENCH_HOPS','8,64').split(',')]
if sys.argv[1]=='summarize':
    runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('process-*.json'))];assert runs;rows=[]
    for h in HOPS:
        r0=runs[0]['hops'][str(h)];med={m:statistics.median(r['hops'][str(h)]['median'][m] for r in runs) for m in r0['median']};row=dict(hop=h,windows=r0['windows'],seconds=med,accuracy=r0['accuracy'],matches_reference=all(r['hops'][str(h)]['matches_reference'] for r in runs))
        if 'aeonN' in med:row['aeonN/scanPL']=med['aeonN']/med['scanPL'];row['range']=[min(r['hops'][str(h)]['median']['aeonN']/r['hops'][str(h)]['median']['scanPL'] for r in runs),max(r['hops'][str(h)]['median']['aeonN']/r['hops'][str(h)]['median']['scanPL'] for r in runs)]
        rows.append(row)
    s=dict(processes=len(runs),environment=runs[0]['environment'],rows=rows);(OUT/'summary.json').write_text(json.dumps(s,indent=1));print(json.dumps(s,indent=1));sys.exit(0)
validation=json.loads((Path(__file__).resolve().parent/'validation.json').read_text()); assert validation['passed'] is True
for filename,digest in validation['source_sha256'].items():
    import hashlib
    assert hashlib.sha256((Path(__file__).resolve().parent.parent/filename).read_bytes()).hexdigest()==digest, ('changed since validation',filename)
proc=int(sys.argv[1]);root=Path(os.environ.get('HAPT_DIR',str(D.parent/'data/hapt')));M={k:v for k,v in np.load(H/'model.npz').items()}
labels=np.loadtxt(root/'RawData/labels.txt',dtype=np.int64);test=set(np.loadtxt(root/'Test/subject_id_test.txt',dtype=int).tolist());R=[]
for p in sorted((root/'RawData').glob('acc_exp*_user*.txt')):
    e,u=map(int,re.search(r'exp(\d+)_user(\d+)',p.name).groups())
    if u in test:
        x=np.loadtxt(p,dtype=np.float32);y=np.zeros(len(x),np.int64)
        for _,_,lab,a,b in labels[(labels[:,0]==e)&(labels[:,1]==u)]:y[a-1:b]=lab
        R.append((x,y))
assert len(R)==19
dil=M['dil'].astype(np.int32);nfd=M['nfd'].astype(np.int32);F=int(84*nfd.sum());idx2=padconv.INDICES.astype(np.int32);idx=idx2.ravel().copy()
L=ctypes.CDLL(os.environ['KERN_SO']);fp=np.ctypeslib.ndpointer(np.float32,flags='C');ip=np.ctypeslib.ndpointer(np.int32,flags='C')
L.scan.argtypes=[fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,ip,ctypes.c_int,fp,ip,fp,ctypes.c_int]
# classifier in single-matrix form: score = z @ A + c   (A = w/scale, c = ym - (mu) @ w), float32 for both pipelines
def head(k):
    A=(M[f'{k}_w']/M[f'{k}_scale'][:,None].astype(np.float64));c=M[f'{k}_ym']-M[f'{k}_mu'].astype(np.float64)@M[f'{k}_w'];return A.astype(np.float32),c.astype(np.float32)
AN,cN=head('N');AP,cP=head('P');bP=[np.ascontiguousarray(M['P_bias'][c]) for c in range(3)];thr=M['thr']
try:
    from numba import njit,prange,vectorize,set_num_threads
    set_num_threads(1);ns=dict(njit=njit,prange=prange,vectorize=vectorize,np=np)
    exec((D/'native_core.py').read_text(),ns);core=ns['_static_transform_uni']
    params=[(None,None,dil,nfd,np.ascontiguousarray(M['N_bias'][c])) for c in range(3)]
except ImportError:core=None
def aeonN(x,h):
    ss=np.arange(0,len(x)-W+1,h);win=np.array([x[s:s+W].T for s in ss],np.float32)
    z=np.concatenate([core(np.ascontiguousarray(win[:,c,:]),params[c],idx2) for c in range(3)],axis=1);return (z@AN+cN).argmax(1)
def scanPL(x,h):
    Mw=(len(x)-W)//h+1;ss=np.arange(Mw)*h;z=np.empty((Mw,3*F+3*NLEV),np.float32)
    for c in range(3):
        xc=np.ascontiguousarray(np.concatenate([x[:,c],np.zeros(4,np.float32)]));o=np.empty((Mw,F),np.float32);L.scan(xc,len(x),Mw,h,W,dil,nfd,len(dil),bP[c],idx,o,F);z[:,c*F:(c+1)*F]=o
        pre=np.zeros((len(x)+1,NLEV),np.int32);np.cumsum(x[:,c:c+1]>thr[c][None,:],0,out=pre[1:]);z[:,3*F+c*NLEV:3*F+(c+1)*NLEV]=(pre[ss+W]-pre[ss])/np.float32(W)
    return (z@AP+cP).argmax(1)
methods={'scanPL':scanPL}
if core is not None:methods['aeonN']=aeonN
names=list(methods);res=dict(process=proc,environment=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,numba=(__import__('numba').__version__ if core else None),load=os.getloadavg()),hops={})
for h in HOPS:
    for m in names:methods[m](R[0][0][:2000],h)
    totals={m:[0.,0.,0.] for m in names};preds={m:[] for m in names};Y=[];nw=0
    for ri,(x,y) in enumerate(R):
        ss=np.arange(0,len(x)-W+1,h);lab=y[ss+W-1];Y.append(lab);nw+=len(ss)
        for rep in range(3):
            sh=(ri+rep+proc)%len(names)
            for m in names[sh:]+names[:sh]:
                t=time.perf_counter();p=methods[m](x,h);totals[m][rep]+=time.perf_counter()-t
                if rep==0:preds[m].append(p)
    Y=np.concatenate(Y);mask=Y>0;acc={}
    for m in names:
        p=np.concatenate(preds[m]);acc[m]=dict(accuracy=float((p[mask]==Y[mask]-1).mean()),macro_f1=padconv.macro_f1(Y[mask]-1,p[mask],NC))
    ref=json.loads((H/f'test-hop{h}.json').read_text());ok=abs(acc['scanPL']['accuracy']-ref['PL_scan']['accuracy'])<1e-12  # aeonN may differ slightly from the NumPy N reference (float32 summation order at threshold ties); both are reported
    res['hops'][str(h)]=dict(windows=nw,scored=int(mask.sum()),seconds=totals,median={m:statistics.median(v) for m,v in totals.items()},accuracy=acc,reference={'N':ref['N'],'PL_scan':ref['PL_scan']},matches_reference=bool(ok))
    print(h,res['hops'][str(h)]['median'],acc,'matches_reference',ok,flush=True)
(OUT/f'process-{proc}.json').write_text(json.dumps(res,indent=1))
