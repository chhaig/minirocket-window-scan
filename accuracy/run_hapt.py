import numpy as np, json, sys, re, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));import padconv
TAG=os.environ.get('PAD_TAG','v2')
HERE=Path(__file__).resolve().parent;root=Path(os.environ.get('HAPT_DIR',str(HERE.parent/'data/hapt')))
(Path(__file__).resolve().parents[1]/'outputs').mkdir(exist_ok=True);(Path(__file__).resolve().parents[1]/'cache').mkdir(exist_ok=True)
rep=int(sys.argv[1]);test_hop=int(sys.argv[2]) if len(sys.argv)>2 else 64;W=256
cache=HERE.parent/'cache';cache.mkdir(exist_ok=True);rc=cache/'hapt-records.npz'
labels=np.loadtxt(root/'RawData/labels.txt',dtype=np.int64)
test_ids=set(np.loadtxt(root/'Test/subject_id_test.txt',dtype=int).tolist());train_ids=set(np.loadtxt(root/'Train/subject_id_train.txt',dtype=int).tolist());assert not train_ids&test_ids
if rc.exists():R=dict(np.load(rc))
else:
    R={}
    for p in sorted((root/'RawData').glob('acc_exp*_user*.txt')):R[p.name]=np.loadtxt(p,dtype=np.float32)
    np.savez(rc,**R)
data={s:([],[],[],[]) for s in ['train','val','test']}
for name in sorted(R):
    exp,subject=map(int,re.search(r'exp(\d+)_user(\d+)',name).groups());x=R[name];y=np.zeros(len(x),np.int64)
    for _,_,lab,a,b in labels[(labels[:,0]==exp)&(labels[:,1]==subject)]:y[a-1:b]=lab
    s='test' if subject in test_ids else ('val' if subject%5==0 else 'train')
    ss=np.arange(0,len(x)-W+1,test_hop if s=='test' else 128);ss=ss[y[ss+W-1]>0]   # labelled endpoints only
    data[s][0].append(x);data[s][1].append(ss);data[s][2].append(y[ss+W-1]-1);data[s][3].append(np.full(len(ss),subject))
data={s:(v[0],v[1],np.concatenate(v[2]),np.concatenate(v[3])) for s,v in data.items()}
seeds=[11+100*rep,12+100*rep,13+100*rep]
logf=open(HERE.parent/f'outputs/hapt-rep{rep}-hop{test_hop}-{TAG}.log','a')
def log(m):logf.write(m+'\n');logf.flush()
out=padconv.run(data,W,seeds,[0.1,10,1000],12,log,str(HERE.parent/f'cache/hapt-{rep}-{test_hop}-{TAG}'))
if out is None:print('MORE');sys.exit(0)
out.update(dataset='HAPT',rep=rep,seeds=seeds,test_hop=test_hop,W=W)
(HERE.parent/f'outputs/hapt-rep{rep}-hop{test_hop}-{TAG}.json').write_text(json.dumps(out,indent=1))
print('DONE')
