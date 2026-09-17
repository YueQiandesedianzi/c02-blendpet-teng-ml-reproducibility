"""Run R01 with shared base quantities and a folded CNN output layer."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from preprocessing import BASE,P6,P16,GROUPS,P6_GROUP,P16_GROUP,SEEDS,width_fraction,prepare_base,mapped_terms,prepare_inference

def interval_router(vpp, anchor_vpp):
    """Apply the frozen R01 routing rule."""
    vpp, anchor_vpp = np.asarray(vpp,float), np.asarray(anchor_vpp,float)
    diffs = np.diff(anchor_vpp)
    if np.all(diffs>=0) or np.all(diffs<=0):
        boundaries=np.array([.5*(anchor_vpp[0]+anchor_vpp[1]),anchor_vpp[2],.5*(anchor_vpp[3]+anchor_vpp[4])])
        route=np.sum(vpp[:,None]>=boundaries,axis=1) if anchor_vpp[0]<=anchor_vpp[-1] else np.sum(vpp[:,None]<boundaries,axis=1)
        return np.clip(route,0,3).astype(int)
    costs=np.empty((len(vpp),4))
    scale=max(float(np.ptp(anchor_vpp)),1e-9)
    for r in range(4):
        low,high=sorted(anchor_vpp[r:r+2])
        outside=np.maximum(low-vpp,0)+np.maximum(vpp-high,0)
        costs[:,r]=outside+.05*abs(vpp-.5*(low+high))/scale
    return np.clip(np.argmin(costs,axis=1),0,3).astype(int)

class FoldedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv1d(2,16,9,padding=4),nn.GroupNorm(4,16),nn.ReLU(),nn.MaxPool1d(2))
        self.conv2=nn.Sequential(nn.Conv1d(16,32,5,padding=2),nn.GroupNorm(8,32),nn.ReLU(),nn.MaxPool1d(2))
        self.output=nn.Linear(32,1,dtype=torch.float64)

    def forward(self,channels):
        values=self.conv2(self.conv1(channels))
        pooled=torch.nn.functional.adaptive_avg_pool1d(values,1).flatten(1)
        return self.output(pooled.to(torch.float64)).reshape(-1)

def convert_checkpoint(state, family, anchor_vpp, source_hash):
    """Fold adjacent linear layers. Keep all active calibration coefficients."""
    model=FoldedCNN()
    s={k.removeprefix('wave_encoder.'):v.clone() for k,v in state.items() if k.startswith('wave_encoder.conv')}
    wp=state['wave_encoder.wave_projection.weight'].double()
    bp=state['wave_encoder.wave_projection.bias'].double()
    wh=state['wave_encoder.wave_head.weight'].double()
    bh=state['wave_encoder.wave_head.bias'].double()
    s['output.weight']=wh@wp
    s['output.bias']=wh@bp+bh
    model.load_state_dict(s,strict=True)
    columns=[0,*range(1,7)] if family=='PGD-PGS' else [0,*range(7,23)]
    inactive=[i for i in range(23) if i not in columns]
    if torch.count_nonzero(state['coefficient'][:,inactive]):
        raise ValueError('The omitted coefficients must be zero.')
    return dict(format_version='r01-equivalent-v01',family=family,cnn_state_dict=model.state_dict(),
                coefficient=state['coefficient'][:,columns].double(),intercept=state['intercept'].double(),
                amplitude_scale=float(state['amplitude_scale']),anchor_vpp=np.asarray(anchor_vpp,float).tolist(),
                base_columns=BASE,term_names=['wave_evidence']+(P6 if family=='PGD-PGS' else P16),
                original_checkpoint_sha256=source_hash,cnn_parameters=sum(p.numel() for p in model.parameters()))

@torch.no_grad()
def predict_prepared_seed(prepared, payload):
    """Run the model on prepared data. Do not compute physical features or channels."""
    seed=payload['seed'];seed_index=SEEDS.index(seed)
    if str(prepared['family'])!=payload['family'] or list(prepared['seeds'])!=SEEDS:
        raise ValueError('Match the family and the seed order.')
    if str(prepared['original_checkpoint_hashes'][seed_index])!=payload['original_checkpoint_sha256']:
        raise ValueError('Use the checkpoint paired with these prepared channels.')
    if float(prepared['amplitude_scales'][seed_index])!=payload['amplitude_scale']:
        raise ValueError('Use the frozen amplitude scale.')
    terms=np.asarray(prepared['physical_terms'],dtype=np.float64)
    names=list(prepared['term_names']);groups=list(prepared['term_groups'])
    if names!=payload['term_names'][1:]:raise ValueError('Use the saved physical term order.')
    expected_groups=P6_GROUP if payload['family']=='PGD-PGS' else P16_GROUP
    if groups!=expected_groups:raise ValueError('Use the saved contribution groups.')
    data=np.asarray(prepared[f'channels_seed_{seed}'],dtype=np.float32)
    vpp=np.asarray(prepared['vpp'],dtype=np.float64)
    if data.shape!=(len(vpp),2,256) or terms.shape!=(len(vpp),len(names)):
        raise ValueError('Match the prepared channel and feature rows.')
    if not np.isfinite(data).all() or not np.isfinite(terms).all() or not np.isfinite(vpp).all():
        raise ValueError('Supply finite prepared values.')
    channels=torch.as_tensor(data)
    model=FoldedCNN();model.load_state_dict(payload['cnn_state_dict'],strict=True);model.eval()
    evidence=np.concatenate([model(c).numpy() for c in channels.split(256)])
    route=interval_router(vpp,payload['anchor_vpp'])
    coef=np.asarray(payload['coefficient'])[route]
    contributions=terms*coef[:,1:]*25
    out=pd.DataFrame({'interval_index':route,'wave_evidence':evidence,
                      'intercept_pp':route*25+np.asarray(payload['intercept'])[route]*25,
                      'cnn_pp':evidence*coef[:,0]*25})
    for group in GROUPS:
        out[group+'_pp']=contributions[:,[i for i,g in enumerate(groups) if g==group]].sum(axis=1)
    out['preclip_pct']=out[['intercept_pp','cnn_pp',*[g+'_pp' for g in GROUPS]]].sum(axis=1)
    out['predicted_pct']=np.clip(out.preclip_pct,0,100)
    out['clip_correction_pp']=out.predicted_pct-out.preclip_pct
    return out

def predict_prepared_ensemble(prepared, payloads):
    outputs=[predict_prepared_seed(prepared,p) for p in payloads]
    if len(outputs)!=3 or any(not np.array_equal(outputs[0].interval_index,o.interval_index) for o in outputs[1:]):
        raise ValueError('Use the three paired seeds with the same routing statistics.')
    columns=['intercept_pp','cnn_pp',*[g+'_pp' for g in GROUPS]]
    out=pd.DataFrame({c:np.mean([o[c].to_numpy() for o in outputs],axis=0) for c in columns})
    out['interval_index']=outputs[0].interval_index
    out['preclip_pct']=np.mean([o.preclip_pct.to_numpy() for o in outputs],axis=0)
    out['predicted_pct']=np.clip(out.preclip_pct,0,100)
    out['clip_correction_pp']=out.predicted_pct-out.preclip_pct
    return out,outputs

def predict_ensemble(waves, base, payloads):
    """Run preprocessing, then model inference. Use the prepared API to separate both stages."""
    return predict_prepared_ensemble(prepare_inference(waves,base,payloads),payloads)

def main():
    parser=argparse.ArgumentParser(description='Run the frozen model on prepared R01 inputs.')
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();torch.set_num_threads(2)
    prepared=dict(np.load(a.prepared,allow_pickle=False));family=str(prepared['family'])
    if family not in ['PGD-PGS','PGT-PGD']:raise ValueError('Use PGD-PGS or PGT-PGD.')
    payloads=[torch.load(a.package/'models/equivalent'/f'{family.lower()}-seed-{s}.pt',weights_only=False,map_location='cpu') for s in SEEDS]
    out,_=predict_prepared_ensemble(prepared,payloads)
    out.insert(0,'analysis_cycle_id',prepared['analysis_cycle_id'])
    a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False,encoding='utf-8-sig')
    print(f'已输出{len(out)}条预测：{a.output}')

if __name__=='__main__':main()
