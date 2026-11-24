from pathlib import Path
import os,sys,runpy
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'accuracy'))
import level_padconv
sys.modules['padconv']=level_padconv
name=sys.argv[1];rep=sys.argv[2];assert name in ('hapt','wisdm')
os.environ['PAD_TAG']='L'+os.environ.get('PAD_NLEV','32')
# Separate the original resumable feature caches for each level count.
s=(R/'accuracy'/f'run_{name}.py').read_text().replace("-hop{test_hop}-{TAG}","-hop{test_hop}-{TAG}")
s=s.replace("-{rep}-{test_hop}-{TAG}","-{rep}-{test_hop}-{TAG}")
sys.argv=[f'run_{name}.py',rep,'64']
exec(compile(s,str(R/'accuracy'/f'run_{name}.py'),'exec'),{'__name__':'__main__','__file__':str(R/'accuracy'/f'run_{name}.py')})
