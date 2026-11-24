from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,json,hashlib,zipfile,datetime,traceback
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'datasets';DATA.mkdir(exist_ok=True)
SPECS={'pamap2':(231,'pamap2%2Bphysical%2Bactivity%2Bmonitoring','pamap2+physical+activity+monitoring'),'mhealth':(319,'mhealth%2Bdataset','mhealth+dataset'),'daphnet':(245,'daphnet%2Bfreezing%2Bof%2Bgait','daphnet+freezing+of+gait')}
def download(name):
 number,slug,title=SPECS[name];folder=DATA/name;folder.mkdir(exist_ok=True)
 url=f'https://archive.ics.uci.edu/static/public/{number}/{slug}.zip';archive=folder/'official-original.zip'
 info={'dataset':name,'url':url,'official_page':f'https://archive.ics.uci.edu/dataset/{number}/{title}','date':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'license':'CC BY 4.0; attribution and citation required; see official UCI page and bundled notices','status':'running'}
 try:
  if not archive.exists():
   part=archive.with_suffix('.zip.part')
   cmd=['curl','--fail','--location','--retry','2','--retry-delay','3','--connect-timeout','20','--max-time','900','--output',str(part),url];info['command']=cmd
   with (folder/'download.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=f,check=True)
   part.rename(archive)
  info['sha256']=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(archive.read_bytes()).hexdigest()
  info['archive_bytes']=archive.stat().st_size
  def extract(p):
   with zipfile.ZipFile(p) as z:
    for x in z.infolist(): assert (p.parent/x.filename).resolve().is_relative_to(folder.resolve()),x.filename
    assert z.testzip() is None
    z.extractall(p.parent)
  extract(archive)
  nested=[p for p in folder.rglob('*.zip') if p!=archive]
  for p in nested:extract(p)
  info['status']='complete';info['files']=sum(p.is_file() for p in folder.rglob('*'))
 except Exception as e:info['status']='failed';info['error']=str(e);info['traceback']=traceback.format_exc()
 (folder/'SOURCE.txt').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
 return info
if __name__=='__main__':
 results=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(download,n) for n in SPECS]):
   x=f.result();results.append(x);(ROOT/'download-status.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(x['dataset'],x['status'],flush=True)
