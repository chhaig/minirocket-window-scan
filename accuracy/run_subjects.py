from pathlib import Path
import os,sys,json,time,hashlib,platform,traceback
ROOT=Path(__file__).resolve().parent
# Set before importing NumPy; reserve modest resources for shared server.
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='4'
os.environ['PAD_PER_KERNEL']='12';os.environ.pop('PAD_KINDS',None);os.environ.pop('PAD_ARMS',None)
sys.path.insert(0,str(ROOT.parent))
import numpy as np,padconv
from load_datasets import load
name=sys.argv[1];r=int(sys.argv[2]);result_dir=ROOT/'results';result_dir.mkdir(exist_ok=True)
result_file=result_dir/f'{name}-r{r}.json'
assert not result_file.exists(), 'Refuse overwriting completed result'
expected=json.loads((ROOT.parent/'SHA256SUMS.json').read_text())['padconv.py'];assert hashlib.sha256(Path(padconv.__file__).read_bytes()).hexdigest()==expected
frozen=json.loads((ROOT.parent/'protocols/subject-splits.json').read_text())
start=time.time();data,manifest=load(name);assert manifest['protocol']==frozen[name];w=manifest['protocol']['W'];dil,nfd=padconv.fit_dilations(w,12);assert 84*int(nfd.sum())==1008
seeds=[11+100*r+c for c in range(9)];cache=ROOT/'feature-cache'/f'{name}-r{r}'
with (result_dir/f'{name}-r{r}.log').open('a',buffering=1) as f:
 def log(message):f.write(time.strftime('%Y-%m-%d %H:%M:%S %z ')+message+'\n');print(name,r,message,flush=True)
 log('START '+json.dumps({'seeds':seeds,'features_per_channel':1008,'splits':manifest['splits']}))
 out=padconv.run(data,w,seeds,[0.1,10,1000],len(manifest['protocol']['classes_original']),log,str(cache),budget=1e9)
 assert out is not None
 assert out['features_per_arm']==9072 and out['level_features']==288
 out.update(dataset=name,rep=r,seeds=seeds,W=w,protocol=manifest['protocol'],split_statistics=manifest['splits'],runtime_seconds=time.time()-start,padconv_sha256=expected,loader_sha256=hashlib.sha256((ROOT/'load_datasets.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT.parent/'protocols/subject-splits.json').read_bytes()).hexdigest(),environment={'python':platform.python_version(),'numpy':np.__version__,'threads':4,'PAD_PER_KERNEL':12})
 result_file.write_text(json.dumps(out,indent=2));log('DONE '+str(out['runtime_seconds']))
