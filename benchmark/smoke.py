"""Small quantized-input check against the NumPy reference, without provider data."""
from pathlib import Path
import ctypes,sys,json
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));import padconv
L=ctypes.CDLL(str(R/'benchmark/kern.so'));fp=np.ctypeslib.ndpointer(np.float32,flags='C');ip=np.ctypeslib.ndpointer(np.int32,flags='C')
L.scan.argtypes=[fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,ip,ctypes.c_int,fp,ip,fp,ctypes.c_int]
rng=np.random.RandomState(317);x=(rng.randint(-12,13,512)/8).astype(np.float32);W=64;dil,nfd=padconv.fit_dilations(W,4);dil=dil.astype(np.int32);nfd=nfd.astype(np.int32);F=84*int(nfd.sum());idx=padconv.INDICES.astype(np.int32).ravel().copy();pm=padconv.padded_mask(dil,nfd)
den=[]
for di,d in enumerate(dil):
 for k in range(84):den.extend([W-4*int(d) if (di+k)%2==0 else W-8*int(d)]*int(nfd[di]))
den=np.array(den,np.float32);bias=rng.choice(np.arange(-6,7,dtype=np.float32)/4,size=F).astype(np.float32)
checks=[]
for hop in (4,16):
 ss=np.arange(0,len(x)-W+1,hop);out=np.empty((len(ss),F),np.float32);xx=np.concatenate([x,np.zeros(4,np.float32)])
 L.scan(xx,len(x),len(ss),hop,W,dil,nfd,len(dil),bias,idx,out,F)
 fa,fi,fc,_=padconv.transform([x],[ss],W,dil,nfd,bias,True);ref=np.where(pm,fc,fi)
 mismatch=int(np.count_nonzero(np.rint(out*den)!=np.rint(ref*den)));assert mismatch==0,mismatch
 checks.append({'hop':hop,'features':out.size,'count_mismatches':mismatch})
print(json.dumps({'passed':True,'quantized_input_checks':checks},indent=2))
