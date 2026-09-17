"""Run package checks. Retrain only in a new directory."""
import argparse,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(root,script,*args):subprocess.run([sys.executable,str(root/'code'/script),*args],check=True)
def main():
    p=argparse.ArgumentParser(description='Reproduce the fixed analysis.')
    p.add_argument('action',choices=['verify','evaluate','replay-r01','retrain']);p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.action=='retrain':
        if a.out is None or a.out.exists():raise ValueError('Supply a new output directory.')
        a.out.mkdir(parents=True)
        for part in ['code','data','reference']:shutil.copytree(ROOT/part,a.out/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc','node_modules'))
        (a.out/'provenance').mkdir();shutil.copy2(ROOT/'provenance/protocol.json',a.out/'provenance/protocol.json')
        for d in ['results','logs','qa','models']:(a.out/d).mkdir()
        run(a.out,'analysis.py','train');run(a.out,'analysis.py','evaluate');run(a.out,'audit_and_summarize.py')
    elif a.action=='verify':run(ROOT,'analysis.py','replay-r01');run(ROOT,'audit_and_summarize.py')
    else:run(ROOT,'analysis.py',a.action)
if __name__=='__main__':main()
