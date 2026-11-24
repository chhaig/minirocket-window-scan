"""Frozen subject splits and preprocessing from CODEX task two; no model code changes."""
from pathlib import Path
import numpy as np,json,re,hashlib
ROOT=Path(__file__).resolve().parent
PROTOCOL={
'pamap2':{'W':512,'channels_zero_based':[4,5,6,21,22,23,38,39,40],'label_column':1,'classes_original':[1,2,3,4,5,6,7,12,13,16,17,24],'train':[101,102,103,104,108],'val':[107],'test':[105,106],'subjects':list(range(101,109))},
'mhealth':{'W':256,'channels_zero_based':[0,1,2,5,6,7,14,15,16],'label_column':23,'classes_original':list(range(1,13)),'train':[1,2,3,4,5,6],'val':[7],'test':[8,9,10],'subjects':list(range(1,11))},
'daphnet':{'W':256,'channels_zero_based':list(range(1,10)),'label_column':10,'classes_original':[1,2],'train':[1,2,3,4,5,6],'val':[7],'test':[8,9,10],'subjects':list(range(1,11))}}

def files(name):
 folder=ROOT/'datasets'/name
 if name=='pamap2':
  ps=sorted(p for p in folder.rglob('subject*.dat') if p.parent.name=='Protocol')
  return [(int(re.search(r'subject(\d+)',p.name).group(1)),p) for p in ps if int(re.search(r'subject(\d+)',p.name).group(1)) in PROTOCOL[name]['subjects']]
 if name=='mhealth':return [(int(re.search(r'subject(\d+)',p.name).group(1)),p) for p in sorted(folder.rglob('mHealth_subject*.log'))]
 return [(int(re.search(r'S(\d+)R',p.name).group(1)),p) for p in sorted(folder.rglob('S*R*.txt')) if re.fullmatch(r'S\d\dR\d\d.txt',p.name)]

def prepare(name):
 spec=PROTOCOL[name];out=ROOT/'prepared'/name;out.mkdir(parents=True,exist_ok=True)
 manifest_path=out/'manifest.json'
 if manifest_path.exists():
  m=json.loads(manifest_path.read_text());assert m['protocol']==spec;return m
 source=json.loads((ROOT/'datasets'/name/'SOURCE.txt').read_text());assert source['status']=='complete'
 found=files(name);assert found and set(s for s,p in found)==set(spec['subjects']),(name,found)
 assert not (set(spec['train'])&set(spec['val']) or set(spec['train'])&set(spec['test']) or set(spec['val'])&set(spec['test']))
 m={'dataset':name,'protocol':spec,'source_sha256':source['sha256'],'records':[],'splits':{s:{'records':0,'windows':0,'subjects':spec[s],'class_distribution':{str(c):0 for c in spec['classes_original']}} for s in ['train','val','test']},'preprocessing':'Selected channels only; NaN per-channel linear interpolation within each file, nearest endpoints; all-zero-labelled samples retained as context; discard only zero-labelled window endpoints; no scaling or normalization'}
 mapping={lab:i for i,lab in enumerate(spec['classes_original'])}
 for subject,p in found:
  raw=np.loadtxt(p,dtype=np.float64);assert raw.ndim==2
  assert raw.shape[1]=={'pamap2':54,'mhealth':24,'daphnet':11}[name], (p,raw.shape)
  labels=raw[:,spec['label_column']];assert np.isfinite(labels).all() and np.array_equal(labels,np.rint(labels))
  labels=labels.astype(np.int64);assert set(np.unique(labels))<=set(mapping)|{0},(p,np.unique(labels))
  x=raw[:,spec['channels_zero_based']].copy();missing=np.isnan(x).sum(axis=0).tolist();assert not np.isinf(x).any()
  for c in range(9):
   valid=~np.isnan(x[:,c]);assert valid.any(),(p,c,'all NaN')
   if not valid.all():x[:,c]=np.interp(np.arange(len(x)),np.flatnonzero(valid),x[valid,c])
  x=x.astype(np.float32);assert np.isfinite(x).all();split=next(s for s in ['train','val','test'] if subject in spec[s]);w=spec['W'];hop=w//4 if split=='test' else w//2
  starts=np.arange(0,max(0,len(x)-w+1),hop,dtype=np.int64);candidates=len(starts);starts=starts[labels[starts+w-1]!=0]
  y=np.array([mapping[int(v)] for v in labels[starts+w-1]],dtype=np.int64)
  entry={'subject':subject,'source_relative':str(p.relative_to(ROOT/'datasets'/name)),'samples':len(x),'split':split,'hop':hop,'windows':len(starts),'discarded_endpoint_zero':candidates-len(starts),'nan_counts_selected_channels':missing,'raw_columns':raw.shape[1],'class_distribution':{str(lab):int(np.count_nonzero(y==i)) for lab,i in mapping.items()},'prepared':f'{subject}-{p.stem}.npz'}
  np.savez(out/entry['prepared'],x=x,starts=starts,y=y)
  m['records'].append(entry);sp=m['splits'][split];sp['records']+=1;sp['windows']+=len(starts)
  for label,n in entry['class_distribution'].items():sp['class_distribution'][label]+=n
  print(name,p.name,split,len(x),len(starts),flush=True)
 for sp in m['splits'].values(): assert sp['windows']>0
 manifest_path.write_text(json.dumps(m,indent=2));return m

def load(name):
 m=prepare(name);out=ROOT/'prepared'/name;data={s:([],[],[],[]) for s in ['train','val','test']}
 for r in m['records']:
  z=np.load(out/r['prepared']);split=r['split'];data[split][0].append(z['x']);data[split][1].append(z['starts']);data[split][2].append(z['y']);data[split][3].append(np.full(len(z['y']),r['subject'],dtype=np.int64))
 return {s:(v[0],v[1],np.concatenate(v[2]),np.concatenate(v[3])) for s,v in data.items()},m
if __name__=='__main__':
 import sys
 if len(sys.argv)>1:prepare(sys.argv[1])
 else:
  (ROOT/'frozen-protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
