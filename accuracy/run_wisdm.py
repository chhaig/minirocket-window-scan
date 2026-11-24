import numpy as np, json, sys, time, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));import padconv
TAG=os.environ.get('PAD_TAG','v2')
HERE=Path(__file__).resolve().parent;EXT=HERE.parent/'data/wisdm'
(Path(__file__).resolve().parents[1]/'outputs').mkdir(exist_ok=True);(Path(__file__).resolve().parents[1]/'cache').mkdir(exist_ok=True)
rep=int(sys.argv[1]);test_hop=int(sys.argv[2]) if len(sys.argv)>2 else 64;W=256
meta=json.loads((EXT/'preparation.json').read_text());D=np.load(EXT/'records.npz')
def split(u):return 'test' if u%6==0 else ('val' if u%6==1 else 'train')
data={s:([],[],[],[]) for s in ['train','val','test']}
for r in meta['records']:
    i=r['index'];x=D[f'x{i}'].astype(np.float32);y=D[f'y{i}'];s=split(r['subject'])
    ss=np.arange(0,len(x)-W+1,test_hop if s=='test' else 128)
    data[s][0].append(x);data[s][1].append(ss);data[s][2].append(y[ss+W-1]);data[s][3].append(np.full(len(ss),r['subject']))
data={s:(v[0],v[1],np.concatenate(v[2]),np.concatenate(v[3])) for s,v in data.items()}
seeds=[11+100*rep,12+100*rep,13+100*rep]
import os
logf=open(HERE.parent/f'outputs/wisdm-rep{rep}-hop{test_hop}-{TAG}.log','a')
def log(m):logf.write(m+'\n');logf.flush()
out=padconv.run(data,W,seeds,[0.1,10,1000],6,log,str(HERE.parent/f'cache/wisdm-{rep}-{test_hop}-{TAG}'))
if out is None:print('MORE');sys.exit(0)
out.update(dataset='WISDM',rep=rep,seeds=seeds,test_hop=test_hop,W=W)
(HERE.parent/f'outputs/wisdm-rep{rep}-hop{test_hop}-{TAG}.json').write_text(json.dumps(out,indent=1))
print('DONE')
