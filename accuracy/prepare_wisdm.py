from pathlib import Path
import numpy as np,json,hashlib,re,collections
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'data/wisdm';ROOT.mkdir(parents=True,exist_ok=True)
PROTO=json.loads((REPO/'protocols/wisdm-preparation.json').read_text());W=PROTO['W'];LABELS=['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,v):(ROOT/name).write_text(json.dumps(v,indent=2))
def emit(**v):print(json.dumps(v))
def split(u):return 'test' if u%6==0 else ('validation' if u%6==1 else 'train')
def prepare():
 source=ROOT/'WISDM_ar_v1.1_raw.txt';manifest=json.loads((REPO/'protocols/wisdm-source.json').read_text())
 assert sha(source)==manifest['files'][source.name]['sha256']
 bad=collections.Counter();boundaries=collections.Counter();rows=[];records=[];previous=None;current=[];meta=[];nonempty=0;valid=0
 def flush():
  if current:
   a=np.array([v[3:] for v in current],np.float32);y=np.array([v[1] for v in current],np.int64);ts=np.array([v[2] for v in current],np.int64)
   records.append((int(current[0][0]),a,y,ts));current.clear()
 for idx,chunk in enumerate(re.split(r'[;\r\n]+',source.read_text())):
  text=chunk.strip()
  if not text:continue
  nonempty+=1
  try:
   s=[v.strip() for v in text.split(',')]
   while s and not s[-1]:s.pop()
   if len(s)!=6:raise ValueError('field_count')
   try:u=int(s[0]);t=int(s[2]);xyz=[float(v) for v in s[3:]]
   except ValueError:raise ValueError('numeric_parse')
   if u<1 or u>36:raise ValueError('subject_range')
   if s[1] not in LABELS:raise ValueError('label')
   if t<0:raise ValueError('negative_timestamp')
   if not np.isfinite(xyz).all():raise ValueError('nonfinite_acceleration')
   row=(u,LABELS.index(s[1]),t,*xyz)
  except ValueError as e:
   bad[str(e)]+=1;flush();previous=None;continue
  valid+=1
  if t==0:boundaries['retained_zero_timestamp']+=1
  if previous:
   reason='subject_change' if u!=previous[0] else ('gap_over_1s' if t>0 and previous[2]>0 and t-previous[2]>10**9 else None)
   if u==previous[0] and t<=previous[2]:boundaries['retained_nonincreasing_time']+=1
   if reason:boundaries[reason]+=1;flush()
  current.append(row);previous=row
 flush();arrays={};short=collections.Counter();stats={k:dict(subjects=set(),records=0,samples=0,train_windows=0) for k in ['train','validation','test']}
 for index,(u,x,y,ts) in enumerate(records):
  if len(x)<W:short['records']+=1;short['samples']+=len(x);continue
  j=len(meta);name=split(u)
  arrays.update({f'x{j}':x,f'y{j}':y,f't{j}':ts})
  rr=dict(index=j,source_segment=index,subject=u,split=name,samples=len(x),label_changes=int(np.sum(y[1:]!=y[:-1])),x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),time_sha256=hashlib.sha256(ts.tobytes()).hexdigest())
  meta.append(rr);st=stats[name];st['subjects'].add(u);st['records']+=1;st['samples']+=len(x);st['train_windows']+=len(range(0,len(x)-W+1,PROTO['train_hop']))
 for st in stats.values():st['subjects']=sorted(st['subjects'])
 assert sum(st['samples'] for st in stats.values())+short['samples']==valid
 for a,b in [('train','test'),('train','validation'),('validation','test')]:assert not set(stats[a]['subjects'])&set(stats[b]['subjects'])
 np.savez_compressed(ROOT/'records.npz',**arrays)
 result=dict(protocol_sha256=sha(REPO/'protocols/wisdm-preparation.json'),raw_sha256=sha(source),nonempty_observations=nonempty,accepted_observations=valid,rejected_observations=dict(bad),breaks=dict(boundaries),short_segments=dict(short),stats=stats,records=meta)
 save('preparation.json',result);emit(event='prepared',**{k:v for k,v in result.items() if k!='records'})

if __name__=='__main__':prepare()
