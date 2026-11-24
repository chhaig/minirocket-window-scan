"""Print the paper's means/deltas and trained timing from archived per-run data."""
from pathlib import Path
import json,statistics
R=Path(__file__).resolve().parent
for name,pattern in [('HAPT','hapt-rep*-hop64-v2.json'),('WISDM','wisdm-rep*-hop64-v2.json'),('PAMAP2','pamap2-r*.json'),('MHEALTH','mhealth-r*.json'),('Daphnet','daphnet-r*.json'),('CWRU-load','cwruload-rep*-v2.json'),('CWRU-size','cwrusize-rep*-v2.json')]:
 runs=[json.loads(p.read_text()) for p in sorted((R/'results/accuracy').glob(pattern))];assert runs,name
 means={a:100*statistics.mean(r[a]['accuracy'] for r in runs) for a in ['N','U','N0','Cc','Cc+L']}
 print(name,'runs',len(runs),json.dumps(means),'P+L - N',means['Cc+L']-means['N'])
for host in ['mac','intel']:
 s=json.loads((R/f'results/timing/{host}/trained/summary.json').read_text());print(host,json.dumps(s['rows']))
