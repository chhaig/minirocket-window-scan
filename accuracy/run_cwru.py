import numpy as np, json, sys, os
from pathlib import Path
os.environ['PAD_PER_KERNEL']='119'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));import padconv
TAG=os.environ.get('PAD_TAG','v2');HERE=Path(__file__).resolve().parent
(Path(__file__).resolve().parents[1]/'outputs').mkdir(exist_ok=True);(Path(__file__).resolve().parents[1]/'cache').mkdir(exist_ok=True)
rep=int(sys.argv[1]);mode=sys.argv[2]            # mode: load (train loads 0,1; val 2; test 3) or size (manifest split)
W=1024;test_hop=512;D=np.load(HERE.parent/'data/cwru/cwru-de.npz');man=json.loads((HERE.parent/'protocols/cwru-manifest.json').read_text())
data={s:([],[],[],[]) for s in ['train','val','test']}
for r in man['records']:
    x=D[str(r['id'])].astype(np.float32)[:,None]
    s={'train':'train','validation':'val','test':'test'}[r['split']] if mode=='size' else ('train' if r['load'] in (0,1) else 'val' if r['load']==2 else 'test')
    ss=np.arange(0,len(x)-W+1,test_hop if s=='test' else W)
    data[s][0].append(x);data[s][1].append(ss);data[s][2].append(np.full(len(ss),r['label']));data[s][3].append(np.full(len(ss),r['id']))
data={s:(v[0],v[1],np.concatenate(v[2]),np.concatenate(v[3])) for s,v in data.items()}
seeds=[11+100*rep];name=f'cwru{mode}'
logf=open(HERE.parent/f'outputs/{name}-rep{rep}-hop{test_hop}-{TAG}.log','a')
def log(m):logf.write(m+'\n');logf.flush()
out=padconv.run(data,W,seeds,[0.1,10,1000],3,log,str(HERE.parent/f'cache/{name}-{rep}-{TAG}'),budget=100)
if out is None:print('MORE');sys.exit(0)
out.update(dataset='CWRU-'+mode,rep=rep,seeds=seeds,test_hop=test_hop,W=W)
(HERE.parent/f'outputs/{name}-rep{rep}-hop{test_hop}-{TAG}.json').write_text(json.dumps(out,indent=1))
print('DONE')
