"""Full 19-record, BOTH-hop gate; all windows, integer counts and predictions."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,ctypes,time,re,warnings
import numpy as np
D=Path(__file__).resolve().parent;BASE=D.parent;sys.path.insert(0,str(BASE));import padconv
M={k:v for k,v in np.load(BASE/'models/model.npz').items()};W=256;dil=M['dil'].astype(np.int32);nfd=M['nfd'].astype(np.int32);F=84*int(nfd.sum());pm=padconv.padded_mask(dil,nfd)
den=[]
for di,d in enumerate(dil):
 for k in range(84):den.extend([W-4*int(d) if (di+k)%2==0 else W-8*int(d)]*int(nfd[di]))
den=np.concatenate([np.tile(np.array(den,dtype=np.float32),3),np.full(96,W,np.float32)]);assert len(den)==3*F+96 and np.all(den>0)
L=ctypes.CDLL(str(D/'kern.so'));fp=np.ctypeslib.ndpointer(np.float32,flags='C');ip=np.ctypeslib.ndpointer(np.int32,flags='C');L.scan.argtypes=[fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,ip,ctypes.c_int,fp,ip,fp,ctypes.c_int]
idx=padconv.INDICES.astype(np.int32).ravel().copy();bias=[np.ascontiguousarray(M['P_bias'][c]) for c in range(3)]
A=(M['P_w']/M['P_scale'][:,None].astype(np.float64)).astype(np.float32);intercept=(M['P_ym']-M['P_mu'].astype(np.float64)@M['P_w']).astype(np.float32)
root=Path(os.environ.get('HAPT_DIR',str(BASE/'data/hapt')));labels=np.loadtxt(root/'RawData/labels.txt',dtype=np.int64);subjects=set(np.loadtxt(root/'Test/subject_id_test.txt',dtype=int).tolist());records=[]
for p in sorted((root/'RawData').glob('acc_exp*_user*.txt')):
 exp,subject=map(int,re.search(r'exp(\d+)_user(\d+)',p.name).groups())
 if subject in subjects:
  x=np.loadtxt(p,dtype=np.float32);y=np.zeros(len(x),np.int64)
  for _,_,lab,a,b in labels[(labels[:,0]==exp)&(labels[:,1]==subject)]:y[a-1:b]=lab
  records.append((p.name,x,y))
assert len(records)==19
result={'passed':False,'scope':'all windows of all 19 HAPT test recordings at hops 8 and 64, trained P biases and trained P+L classifier','source_sha256':json.loads((D/'source-sha256.json').read_text()),'hops':{},'started':time.time()}
with warnings.catch_warnings(record=True) as caught:
 warnings.simplefilter('always')
 for hop in [8,64]:
  rows=[];ys=[];ps=[];pns=[];total_counts=total_preds=total_head=0;maxdiff=0.;rawdiff=0;total_values=0;total_windows=0
  for name,x,y in records:
   ss=np.arange(0,len(x)-W+1,hop);n=len(ss);zs=np.empty((n,3*F+96),np.float32);cols=[];levs=[]
   for c in range(3):
    xc=np.ascontiguousarray(np.concatenate([x[:,c],np.zeros(4,np.float32)]));out=np.empty((n,F),np.float32)
    L.scan(xc,len(x),n,hop,W,dil,nfd,len(dil),bias[c],idx,out,F);zs[:,c*F:(c+1)*F]=out
    lev=padconv.level_features([np.ascontiguousarray(x[:,c])],[ss],W,M['thr'][c]);zs[:,3*F+c*32:3*F+(c+1)*32]=lev;levs.append(lev)
    fa,fi,fc,b=padconv.transform([np.ascontiguousarray(x[:,c])],[ss],W,dil,nfd,M['P_bias'][c],True);cols.append(np.where(pm,fc,fi))
   zn=np.concatenate(cols+levs,axis=1);assert np.isfinite(zs).all() and np.isfinite(zn).all()
   cs=np.rint(zs*den).astype(np.int32);cn=np.rint(zn*den).astype(np.int32)
   assert np.all(cs>=0) and np.all(cs<=den) and np.all(cn>=0) and np.all(cn<=den)
   count_mismatch=int(np.count_nonzero(cs!=cn));scores=zs@A+intercept;score_np=zn@A+intercept
   score_reference=((zn/M['P_scale']-M['P_mu']).astype(np.float64))@M['P_w']+M['P_ym']
   assert np.isfinite(scores).all() and np.isfinite(score_np).all() and np.isfinite(score_reference).all()
   pred=scores.argmax(1);pn=score_reference.argmax(1);pred_mismatch=int(np.count_nonzero(pred!=pn));head_mismatch=int(np.count_nonzero(score_np.argmax(1)!=pn));diff=np.abs(zs-zn)
   row={'record':name,'windows':n,'feature_values':int(zs.size),'count_mismatches':count_mismatch,'prediction_mismatches':pred_mismatch,'numpy_head_conversion_mismatches':head_mismatch,'raw_feature_unequal':int(np.count_nonzero(diff)),'max_abs_feature_diff':float(diff.max())};rows.append(row)
   total_counts+=count_mismatch;total_preds+=pred_mismatch;total_head+=head_mismatch;maxdiff=max(maxdiff,row['max_abs_feature_diff']);rawdiff+=row['raw_feature_unequal'];total_values+=zs.size;total_windows+=n;ys.append(y[ss+W-1]);ps.append(pred);pns.append(pn)
   print(hop,name,'windows',n,'count mismatch',count_mismatch,'prediction mismatch',pred_mismatch,flush=True)
  yy=np.concatenate(ys);pp=np.concatenate(ps);pnp=np.concatenate(pns);mask=yy>0;ref=json.loads((BASE/f'models/test-hop{hop}.json').read_text())['PL_scan']
  metrics={'accuracy':float((pp[mask]==yy[mask]-1).mean()),'macro_f1':padconv.macro_f1(yy[mask]-1,pp[mask],12)}
  result['hops'][str(hop)]={'windows':total_windows,'scored_windows':int(mask.sum()),'feature_values':int(total_values),'count_mismatches':total_counts,'prediction_mismatches':total_preds,'numpy_head_conversion_mismatches':total_head,'raw_feature_unequal':rawdiff,'max_abs_feature_diff':maxdiff,'scan_metrics':metrics,'reference_metrics':ref,'records':rows}
 result['warnings']=sorted(set(str(w.message) for w in caught))
result['passed']=all(x['count_mismatches']==0 and x['prediction_mismatches']==0 and x['numpy_head_conversion_mismatches']==0 and x['scan_metrics']==x['reference_metrics'] for x in result['hops'].values());result['finished']=time.time()
(D/'validation.json').write_text(json.dumps(result,indent=2));print('VALIDATION',result['passed'],flush=True)
if not result['passed']:raise SystemExit('Full count/prediction/reference gate failed; DO NOT TIME')
