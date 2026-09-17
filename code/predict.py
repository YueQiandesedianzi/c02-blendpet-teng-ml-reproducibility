"""Predict with saved controls. Do not load target labels."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from analysis import ROOT,FAMILIES,SEEDS,MODELS,WaveOrdinal,evidence,physical_terms,apply_head
from preprocessing import BASE

def predict(waves,base,ids,family,context='final'):
    directory=ROOT/'models'/context
    bank=np.load(directory/f'{family.lower()}-templates.npz')['templates']
    terms,names,_=physical_terms(base,waves,None,None,family,bank=bank)
    rows=[]
    for seed in SEEDS:
        payload=torch.load(directory/f'{family.lower()}-seed-{seed}-cnn.pt',weights_only=False,map_location='cpu')
        model=WaveOrdinal();model.load_state_dict(payload['state_dict'])
        ev=evidence(model,waves,payload['amplitude_scale'])
        for variant in MODELS:
            head=json.loads((directory/f'{family.lower()}-{variant.lower()}-seed-{seed}-head.json').read_text())
            value,route=apply_head(ev,terms,names,base.vpp_v.to_numpy(),head)
            rows.append(pd.DataFrame(dict(analysis_cycle_id=ids,family=family,context=context,model=variant,seed=seed,preclip_pct=value,interval_index=route)))
    seeds=pd.concat(rows,ignore_index=True)
    ensemble=seeds.groupby(['analysis_cycle_id','family','context','model'],as_index=False).preclip_pct.mean()
    ensemble['predicted_pct']=ensemble.preclip_pct.clip(0,100)
    return ensemble,seeds

def main():
    p=argparse.ArgumentParser(description='Predict from target-free cycle inputs.')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=FAMILIES,required=True);p.add_argument('--context',default='final')
    a=p.parse_args();z=np.load(a.input,allow_pickle=False)
    frame=pd.DataFrame(z['base'],columns=z['base_names'].astype(str))[BASE]
    out,_=predict(z['waves'],frame,z['analysis_cycle_id'].astype(str),a.family,a.context)
    a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False,encoding='utf-8-sig')
    print(json.dumps(dict(rows=len(out),target_labels_read=False)))
if __name__=='__main__':main()
