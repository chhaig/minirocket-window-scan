"""Download HAPT, WISDM or CWRU from official provider hosts only."""
from pathlib import Path
import sys,json,hashlib,subprocess,zipfile,tarfile
R=Path(__file__).resolve().parents[1]
def fetch(url,p,digest=None):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  part=p.with_suffix(p.suffix+'.part');subprocess.run(['curl','--fail','--location','--retry','2','--output',str(part),url],check=True);part.replace(p)
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if digest:assert h==digest,(p,'Provider content differs from the study version')
 return h
name=sys.argv[1];D=R/'data'/name;D.mkdir(parents=True,exist_ok=True)
if name=='hapt':
 url='https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip';p=D/'official-original.zip';h=fetch(url,p)
 def unpack(p):
  with zipfile.ZipFile(p) as z:
   for m in z.infolist():assert (D/m.filename).resolve().is_relative_to(D.resolve())
   z.extractall(D)
 unpack(p)
 for q in list(D.rglob('*.zip')):
  if q!=p:unpack(q)
 candidates=[q.parent for q in D.rglob('labels.txt') if q.parent.name=='RawData'];assert len(candidates)==1
 source=candidates[0].parent
 if source!=D:
  import shutil
  for folder in ['RawData','Train','Test']:shutil.copytree(source/folder,D/folder,dirs_exist_ok=True)
 (D/'SOURCE.json').write_text(json.dumps({'url':url,'sha256':h},indent=2))
elif name=='wisdm':
 m=json.loads((R/'protocols/wisdm-source.json').read_text());p=D/'WISDM_ar_v1.1.tar.gz';fetch(m['source'],p,m['files'][p.name]['sha256'])
 with tarfile.open(p) as t:
  for member in t.getmembers():
   filename=Path(member.name).name
   if member.isfile() and filename in m['files'] and filename!=p.name:
    content=t.extractfile(member).read();assert hashlib.sha256(content).hexdigest()==m['files'][filename]['sha256'];(D/filename).write_bytes(content)
elif name=='cwru':
 m=json.loads((R/'protocols/cwru-manifest.json').read_text())
 for row in m['records']:fetch(row['url'],D/f"{row['id']}.mat",row['sha256'])
else:raise SystemExit('Choose hapt, wisdm or cwru; use download_uci.py for the other three datasets.')
print(name,'download complete')
