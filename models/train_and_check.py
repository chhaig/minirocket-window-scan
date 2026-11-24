"""Trained-model end-to-end check on HAPT (2026-11-24).
 train   : fit N (zero-padded crops) and P+L (padding-free + level) models with padconv, save model.npz
 test H  : per test recording at hop H: N features (NumPy), P features (C scan, kern.so) + level features;
           report accuracy/macro-F1 of both models on labelled endpoints, and (hop 64 only) whether the C scan
           reproduces the NumPy P features used in the accuracy experiments.
KERN_SO env var: path of benchmark/kern.so built with benchmark/build.py.
"""
import numpy as np, json, sys, os, re, ctypes, time
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent));import padconv
from padconv import transform, level_features, Ridge, macro_f1, padded_mask, fit_dilations, INDICES
W=256;NC=12;NLEV=32;root=Path(os.environ.get('HAPT_DIR',str(H.parent/'data/hapt')))
def load():
    labels=np.loadtxt(root/'RawData/labels.txt',dtype=np.int64);test=set(np.loadtxt(root/'Test/subject_id_test.txt',dtype=int).tolist());out=[]
    for p in sorted((root/'RawData').glob('acc_exp*_user*.txt')):
        e,u=map(int,re.search(r'exp(\d+)_user(\d+)',p.name).groups());x=np.loadtxt(p,dtype=np.float32);y=np.zeros(len(x),np.int64)
        for _,_,lab,a,b in labels[(labels[:,0]==e)&(labels[:,1]==u)]:y[a-1:b]=lab
        out.append(dict(x=x,y=y,subject=u,split='test' if u in test else ('val' if u%5==0 else 'train')))
    return out
dil,nfd=fit_dilations(W);pm=padded_mask(dil,nfd);F=int(84*nfd.sum())
def starts(r,h):
    ss=np.arange(0,len(r['x'])-W+1,h);return ss[r['y'][ss+W-1]>0]
def feats(recs,ss,kind,bias=None,rng=None,thr=None):
    cols=[];B=[];T=[]
    for c in range(3):
        xs=[np.ascontiguousarray(r['x'][:,c]) for r in recs]
        r_=transform(xs,ss,W,dil,nfd,None if bias is None else bias[c],kind=='P',rng[c] if rng else None)
        cols.append(np.where(pm,r_[2] if kind=='P' else r_[0],r_[1]));B.append(r_[-1])
    if kind=='P':
        for c in range(3):
            xs=[np.ascontiguousarray(r['x'][:,c]) for r in recs]
            if thr is None or len(T)<3 and thr is None:pass
            t=thr[c] if thr is not None else np.quantile(np.concatenate([np.lib.stride_tricks.sliding_window_view(x,W)[s].ravel() for x,s in zip(xs,ss) if len(s)]),(np.arange(NLEV)+0.5)/NLEV).astype(np.float32)
            T.append(t);cols.append(level_features(xs,ss,W,t))
    return np.concatenate(cols,1),B,T
if sys.argv[1]=='train':
    R=load();tr=[r for r in R if r['split']=='train'];va=[r for r in R if r['split']=='val'];M={}
    for kind in 'NP':
        rng=[np.random.RandomState(11+c) for c in range(3)]
        st=[starts(r,128) for r in tr];sv=[starts(r,128) for r in va]
        ztr,B,T=feats(tr,st,kind,rng=rng);zva,_,_=feats(va,sv,kind,bias=B,thr=T if kind=='P' else None)
        ytr=np.concatenate([r['y'][s+W-1]-1 for r,s in zip(tr,st)]);yva=np.concatenate([r['y'][s+W-1]-1 for r,s in zip(va,sv)])
        m=Ridge().prep(ztr,ytr,NC);sel=[(macro_f1(yva,m.solve(a).predict(zva),NC),-a) for a in (0.1,10,1000)];a=-max(sel)[1]
        m=Ridge().prep(np.concatenate([ztr,zva]),np.concatenate([ytr,yva]),NC).solve(a)
        M.update({f'{kind}_bias':np.stack(B),f'{kind}_scale':m.scale,f'{kind}_mu':m.mu,f'{kind}_w':m.w,f'{kind}_ym':m.ym,f'{kind}_alpha':a})
        if kind=='P':M['thr']=np.stack(T)
        print(kind,'alpha',a,'val',sel,flush=True)
    np.savez(H/'model.npz',dil=dil,nfd=nfd,**M);print('saved')
else:
    h=int(sys.argv[2]);M=np.load(H/'model.npz');R=[r for r in load() if r['split']=='test']
    L=ctypes.CDLL(os.environ['KERN_SO']);fp=np.ctypeslib.ndpointer(np.float32,flags='C');ip=np.ctypeslib.ndpointer(np.int32,flags='C')
    L.scan.argtypes=[fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,ip,ctypes.c_int,fp,ip,fp,ctypes.c_int]
    d32=dil.astype(np.int32);n32=nfd.astype(np.int32);idx=INDICES.astype(np.int32).ravel().copy()
    def pred(z,k):return (((z/M[f'{k}_scale']-M[f'{k}_mu']).astype(np.float64))@M[f'{k}_w']+M[f'{k}_ym']).argmax(1)
    Y=[];PN=[];PP=[];G=[];mism=0;cmpn=0;pdiff=0;t0=time.time()
    for r in R:
        x=r['x'];ss=starts(r,h)
        if len(ss)==0:continue
        zn,_,_=feats([r],[ss],'N',bias=M['N_bias']);PN.append(pred(zn,'N'));del zn
        Mall=(len(x)-W)//h+1;z=np.empty((len(ss),3*F+3*NLEV),np.float32)
        for c in range(3):
            xc=np.ascontiguousarray(np.concatenate([x[:,c],np.zeros(4,np.float32)]));o=np.empty((Mall,F),np.float32)
            L.scan(xc,len(x),Mall,h,W,d32,n32,len(d32),np.ascontiguousarray(M['P_bias'][c]),idx,o,F);z[:,c*F:(c+1)*F]=o[ss//h]
            z[:,3*F+c*NLEV:3*F+(c+1)*NLEV]=level_features([np.ascontiguousarray(x[:,c])],[ss],W,M['thr'][c])
        pp=pred(z,'P');PP.append(pp)
        if h==64:
            zr,_,_=feats([r],[ss],'P',bias=M['P_bias'],thr=M['thr']);mism+=int((np.abs(zr-z)>1/(2*W)).sum());cmpn+=zr.size;pdiff+=int((pred(zr,'P')!=pp).sum())
        Y.append(r['y'][ss+W-1]-1);G.append(np.full(len(ss),r['subject']))
    Y=np.concatenate(Y);PN=np.concatenate(PN);PP=np.concatenate(PP)
    out=dict(hop=h,scored_windows=int(len(Y)),N=dict(accuracy=float((PN==Y).mean()),macro_f1=macro_f1(Y,PN,NC)),PL_scan=dict(accuracy=float((PP==Y).mean()),macro_f1=macro_f1(Y,PP,NC)),
             alpha=dict(N=float(M['N_alpha']),P=float(M['P_alpha'])),seconds=time.time()-t0)
    if h==64:out['scan_vs_numpy_P']=dict(feature_values=cmpn,count_mismatches=mism,prediction_mismatches=pdiff)
    (H/f'test-hop{h}.json').write_text(json.dumps(out,indent=1));print(json.dumps(out))
