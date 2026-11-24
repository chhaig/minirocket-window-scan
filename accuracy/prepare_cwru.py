"""Convert the manifest's 36 official MAT files, preserving record identity."""
from pathlib import Path
import json,hashlib,numpy as np
from scipy.io import loadmat
R=Path(__file__).resolve().parents[1];D=R/'data/cwru';manifest=json.loads((R/'protocols/cwru-manifest.json').read_text());arrays={}
for r in manifest['records']:
 p=D/f"{r['id']}.mat";assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],p
 m=loadmat(p);keys=[k for k in m if k.endswith('_DE_time')];assert len(keys)==1
 x=np.asarray(m[keys[0]],dtype=np.float32).reshape(-1);assert np.isfinite(x).all();arrays[str(r['id'])]=x
assert len(arrays)==36
np.savez(D/'cwru-de.npz',**arrays);print('Converted 36 verified DE_time arrays.')
