"""Padding-convention experiment (2026-11-23).
Question: does a MiniRocket-style model need independently zero-padded crops,
or does it reach the same accuracy when trained AND evaluated with
(C) recording-context responses, or (V) valid-only pooling?
Pure NumPy re-implementation (no aeon/numba); not bit-identical to aeon.
Arms: N native crop zero padding (alternating padded/valid pooling)
      C context padding: taps outside the crop read the recording (zeros outside the recording)
      V valid-only pooling for every feature
      U mismatch: model trained with N, evaluated on C features (the manuscript's diagnostic)
"""
import numpy as np, itertools, json, sys, time
from numpy.lib.stride_tricks import sliding_window_view
INDICES=np.array(list(itertools.combinations(range(9),3)),np.int64)
def fit_dilations(W,per_kernel=40,maxd=32):
    tm=min(per_kernel,maxd);mult=per_kernel/tm
    dil,counts=np.unique(np.logspace(0,np.log2((W-1)/8),tm,base=2).astype(np.int32),return_counts=True)
    nfd=(counts*mult).astype(np.int32);rem=per_kernel-nfd.sum();i=0
    while rem>0:
        nfd[i]+=1;rem-=1;i=(i+1)%len(nfd)
    return dil,nfd
def quantiles(n):
    phi=(np.sqrt(5)+1)/2
    return np.array([(i*phi)%1 for i in range(1,n+1)],np.float32)
def ext_windows(recs,starts,W,d,context):
    """recs: list of 1-D float32 arrays; starts: list of start arrays. Returns M x (W+8d)."""
    p=4*d;out=[]
    for x,ss in zip(recs,starts):
        if len(ss)==0:continue
        if context:
            xp=np.concatenate([np.zeros(p,np.float32),x,np.zeros(p,np.float32)])
            out.append(sliding_window_view(xp,W+2*p)[ss])
        else:
            e=np.zeros((len(ss),W+2*p),np.float32);e[:,p:p+W]=sliding_window_view(x,W)[ss];out.append(e)
    return np.concatenate(out)
def ext_causal(recs,starts,W,d):
    """M x (W+8d): each crop preceded by its 8d past samples (zeros before the recording start)."""
    p=8*d;out=[]
    for x,ss in zip(recs,starts):
        if len(ss)==0:continue
        xp=np.concatenate([np.zeros(p,np.float32),x]);out.append(sliding_window_view(xp,W+p)[ss])
    return np.concatenate(out)
def responses(ext,W,d,chunk=512):
    """yield (k, C[M,W]) lazily per chunk is awkward; instead return closure computing kernel k for a chunk."""
    taps=[ext[:,j*d:j*d+W] for j in range(9)]
    return taps
def transform(recs,starts,W,dil,nfd,biases,context,fit_rng=None,chunk=1024):
    """Returns full-pool counts fraction F_all and interior-pool fraction F_in for every feature.
    If fit_rng is given, biases are fitted (returned) from one random window per (dilation,kernel)."""
    M=sum(len(s) for s in starts);nf_total=84*int(nfd.sum())
    fit=biases is None
    if fit:biases=np.zeros(nf_total,np.float32);q=quantiles(nf_total)
    Fa=np.zeros((M,nf_total),np.float32);Fi=np.zeros((M,nf_total),np.float32);Fc=np.zeros((M,nf_total),np.float32) if context else None;fs=0
    for di,d in enumerate(dil):
        d=int(d);p=4*d;nf=int(nfd[di]);ext=ext_causal(recs,starts,W,d) if context=='causal' else ext_windows(recs,starts,W,d,context)
        if fit:
            rows=fit_rng.randint(0,M,size=84)
            for k in range(84):
                e=ext[rows[k]];a,b,c=INDICES[k]
                C=-sum(e[j*d:j*d+W] for j in range(9))+3*(e[a*d:a*d+W]+e[b*d:b*d+W]+e[c*d:c*d+W])
                biases[fs+k*nf:fs+(k+1)*nf]=np.quantile(C,q[fs+k*nf:fs+(k+1)*nf])
        for lo in range(0,M,chunk):
            e=ext[lo:lo+chunk];taps=[e[:,j*d:j*d+W] for j in range(9)]
            alpha=-taps[0]
            for j in range(1,9):alpha=alpha-taps[j]
            g=[3*t for t in taps]
            for k in range(84):
                a,b,c=INDICES[k];C=alpha+g[a]+g[b]+g[c]
                for qi in range(nf):
                    f=fs+k*nf+qi;cmp=C>biases[f]
                    Fa[lo:lo+chunk,f]=cmp.mean(1)
                    Fi[lo:lo+chunk,f]=cmp[:,p:W-p].mean(1)
                    if context:Fc[lo:lo+chunk,f]=cmp[:,:W-p].mean(1)
        fs+=84*nf
    return (Fa,Fi,Fc,biases) if context else (Fa,Fi,biases)
def padded_mask(dil,nfd):
    m=[]
    for di in range(len(dil)):
        for k in range(84):m.extend([(di+k)%2==0]*int(nfd[di]))
    return np.array(m)
def macro_f1(y,p,nc):
    cm=np.zeros((nc,nc),np.int64);np.add.at(cm,(y,p),1);den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(nc),where=den>0)
    present=cm.sum(1)>0
    return float(f[present].mean())
class Ridge:
    """One-vs-rest ridge classifier with intercept, solved exactly in the dual."""
    def prep(self,Z,y,nc):
        self.scale=Z.std(0).astype(np.float32);self.scale[self.scale==0]=1
        X=Z/self.scale;self.mu=X.mean(0);X-=self.mu;self.X=X
        Y=-np.ones((len(y),nc));Y[np.arange(len(y)),y]=1;self.ym=Y.mean(0);self.Y=Y-self.ym
        self.K=(X@X.T).astype(np.float64);return self
    def solve(self,alpha):
        K=self.K.copy();K[np.diag_indices_from(K)]+=alpha
        self.w=(self.X.T.astype(np.float64)@np.linalg.solve(K,self.Y));return self
    def predict(self,Z):
        return (((Z/self.scale-self.mu).astype(np.float64))@self.w+self.ym).argmax(1)
def level_features(recs,starts,W,thr):
    """Fraction of window samples above each fixed threshold, via prefix counts (O(1) per window and threshold)."""
    out=[]
    for x,ss in zip(recs,starts):
        if len(ss)==0:continue
        pre=np.zeros((len(x)+1,len(thr)),np.int32);pre[1:]=np.cumsum(x[:,None]>thr[None,:],0)
        out.append((pre[ss+W]-pre[ss]).astype(np.float32)/W)
    return np.concatenate(out)
def demean(recs,starts,W):
    """Return per-window mean-removed crops as pseudo-records (one per window) for the N0 diagnostic."""
    r=[];s=[]
    for x,ss in zip(recs,starts):
        for a in ss:
            w=x[a:a+W];r.append((w-w.mean()).astype(np.float32));s.append(np.array([0]))
    return r,s
import os as _os
KINDS=_os.environ.get('PAD_KINDS','N,C,N0,L').split(',')
ARMS=_os.environ.get('PAD_ARMS','N,C,V,N0,C+L,V+L,Cc,Cc+L').split(',')  # Cc: causal context (past samples only; pooled centres [0,W-4d))
def run(data,W,seeds,alphas,nc,log,cache,budget=140,nlev=int(_os.environ.get('PAD_NLEV','32'))):
    """Resumable: caches features and per-arm predictions under `cache`; returns None if out of time."""
    import os
    t0=time.time();os.makedirs(cache,exist_ok=True)
    dil,nfd=fit_dilations(W,int(_os.environ.get('PAD_PER_KERNEL','40')));pm=padded_mask(dil,nfd);ch=data['train'][0][0].shape[1]
    for c in range(ch):
        for kind in KINDS:
            if os.path.exists(f'{cache}/f-{c}-{kind}.done'):continue
            if time.time()-t0>budget*0.6:return None
            rng=np.random.RandomState(seeds[c]);t=time.time();b=None
            for s in ['train','val','test']:
                recs=[np.ascontiguousarray(r[:,c]) for r in data[s][0]];st=data[s][1]
                if kind=='L':
                    if s=='train':
                        samp=np.concatenate([sliding_window_view(x,W)[ss].ravel() for x,ss in zip(recs,st) if len(ss)])
                        b=np.quantile(samp,(np.arange(nlev)+0.5)/nlev).astype(np.float32)
                    np.save(f'{cache}/L-{s}-{c}.npy',level_features(recs,st,W,b));continue
                if kind=='N0':recs,st=demean(recs,st,W)
                r_=transform(recs,st,W,dil,nfd,b,{'C':True,'A':'causal'}.get(kind,False),rng if s=='train' else None);Fa,Fi,b=r_[0],r_[1],r_[-1]
                alt=Fa if kind=='A' else np.where(pm,Fa,Fi);np.save(f'{cache}/{kind}-{s}-{c}.npy',alt)
                if kind=='C':np.save(f'{cache}/Cc-{s}-{c}.npy',np.where(pm,r_[2],Fi))
                if kind=='N':np.save(f'{cache}/V-{s}-{c}.npy',Fi)
            open(f'{cache}/f-{c}-{kind}.done','w').close()
            log(f'channel {c} {kind} {time.time()-t:.0f}s')
    def Z(arm,s):return np.concatenate([np.load(f'{cache}/{a}-{s}-{c}.npy') for a in arm.split('+') for c in range(ch)],1)
    ytr,yva,yte=data['train'][2],data['val'][2],data['test'][2];gte=data['test'][3]
    for arm in ARMS:
        if os.path.exists(f'{cache}/pred-{arm}.npz'):continue
        if time.time()-t0>budget*0.5:return None
        ztr,zva,zte=Z(arm,'train'),Z(arm,'val'),Z(arm,'test')
        m=Ridge().prep(ztr,ytr,nc);sel=[]
        for a in alphas:sel.append((macro_f1(yva,m.solve(a).predict(zva),nc),-a))
        a=-max(sel)[1];m=Ridge().prep(np.concatenate([ztr,zva]),np.concatenate([ytr,yva]),nc).solve(a)
        extra={}
        if arm=='N':extra['pu']=m.predict(Z('C','test'))
        np.savez(f'{cache}/pred-{arm}.npz',p=m.predict(zte),alpha=a,val=[s[0] for s in sel],**extra)
        log(f'{arm} fitted alpha={a} {time.time()-t0:.0f}s')
    out={};P={arm:np.load(f'{cache}/pred-{arm}.npz') for arm in ARMS}
    for arm in ARMS:
        p=P[arm]['p']
        out[arm]=dict(alpha=float(P[arm]['alpha']),val_macro_f1=P[arm]['val'].tolist(),accuracy=float((p==yte).mean()),macro_f1=macro_f1(yte,p,nc),
                      changed_vs_N=float((p!=P['N']['p']).mean()) if 'N' in P else None,per_group={str(g):float((p[gte==g]==yte[gte==g]).mean()) for g in np.unique(gte)})
        log(f'{arm:4s} alpha={out[arm]["alpha"]:<7} acc={out[arm]["accuracy"]:.4f} f1={out[arm]["macro_f1"]:.4f}')
    if 'N' in P:pu=P['N']['pu'];out['U']=dict(accuracy=float((pu==yte).mean()),macro_f1=macro_f1(yte,pu,nc),changed_vs_N=float((pu!=P['N']['p']).mean()))
    cnt=np.bincount(np.concatenate([ytr,yva]),minlength=nc);maj=np.full_like(yte,cnt.argmax())
    out['majority']=dict(accuracy=float((maj==yte).mean()),macro_f1=macro_f1(yte,maj,nc))
    out['n']={s:int(len(data[s][2])) for s in data};out['dilations']=dil.tolist();out['features_per_arm']=int(84*nfd.sum()*ch);out['level_features']=nlev*ch
    import shutil;shutil.rmtree(cache)
    return out
