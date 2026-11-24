"""Build both C kernels with strict floating point and bind validation inputs."""
from pathlib import Path
import subprocess,sys,hashlib,json,os
D=Path(__file__).resolve().parent;R=D.parent
flags=['-O3','-mcpu=native' if sys.platform=='darwin' else '-march=native','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']
cmd=[os.environ.get('CC','cc'),*flags,'-o',str(D/'kern.so'),str(D/'kern-native.c'),str(D/'kern-scan.c')]
subprocess.run(cmd,check=True)
(D/'build-command.json').write_text(json.dumps(cmd,indent=2))
files=[R/'padconv.py',*D.glob('*.py'),D/'kern-native.c',D/'kern-scan.c',D/'kern.so',*sorted((R/'models').glob('*.npz')),*sorted((R/'models').glob('test-hop*.json'))]
(D/'source-sha256.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('Built strict kernels; run benchmark/validate_scan.py before timing.')
