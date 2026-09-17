"""Prepare fixed physical terms and CNN channels before model inference."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE=['vpp_v','positive_peak_v','rms_v','mean_abs_v','positive_auc_vs','negative_auc_abs_vs','signed_area_balance',
      'duration_s','half_peak_width_s','half_peak_width_fraction','skewness','kurtosis','zero_crossings','template_corr','snr_proxy_db']
P6=['log1p_vpp','signed_dominant_peak_over_vpp','peak_balance','rms_over_vpp','signed_area_balance','half_peak_width_fraction']
P16=['vpp_v','positive_peak_v','negative_peak_abs_v','rms_v','mean_abs_v','positive_auc_vs','negative_auc_abs_vs','absolute_auc_vs',
     'crest_factor','half_peak_width_s','duration_s','skewness','kurtosis','zero_crossings','template_corr','snr_proxy_db']
GROUPS=['amplitude','area','time_width','wave_statistics','reference_quality']
P6_GROUP=['amplitude']*4+['area','time_width']
P16_GROUP=['amplitude']*5+['area']*3+['amplitude']+['time_width']*2+['wave_statistics']*3+['reference_quality']*2
SEEDS=[42,123,2026]

def width_fraction(wave):
    values=np.asarray(wave,dtype=np.float64);peak=int(np.argmax(abs(values)));threshold=.5*abs(float(values[peak]))
    if threshold<=1e-12:return 0.
    left=right=peak
    while left>0 and abs(values[left-1])>=threshold:left-=1
    while right+1<len(values) and abs(values[right+1])>=threshold:right+=1
    return (right-left+1)/len(values)

def prepare_base(aligned_waves,physical_rows):
    w=np.asarray(aligned_waves,dtype=np.float32)
    if w.ndim!=2 or w.shape[1]!=256 or len(w)!=len(physical_rows):raise ValueError('Match the rows. Use 256 points for each waveform.')
    result=physical_rows.copy();positive=np.maximum(w,0).sum(1);negative=np.maximum(-w,0).sum(1)
    result['signed_area_balance']=(positive-negative)/np.maximum(positive+negative,1e-9)
    result['half_peak_width_fraction']=np.array([width_fraction(v) for v in w],dtype=np.float32)
    return result[BASE].astype(np.float64)

def mapped_terms(base,family):
    """Compute the fixed functions once in preprocessing."""
    x={k:base[k].to_numpy(np.float64) for k in BASE};v,p,r=x['vpp_v'],x['positive_peak_v'],x['rms_v'];minimum=p-v;safe=np.maximum(v,1e-9)
    dominant=np.where(abs(p)>=abs(minimum),p,minimum)
    if family=='PGD-PGS':
        return np.column_stack([np.log1p(safe),dominant/safe,(np.maximum(p,0)-np.maximum(-minimum,0))/safe,r/safe,x['signed_area_balance'],x['half_peak_width_fraction']]),P6,P6_GROUP
    if family!='PGT-PGD':raise ValueError('Use PGD-PGS or PGT-PGD.')
    x['negative_peak_abs_v']=abs(minimum);x['absolute_auc_vs']=x['positive_auc_vs']+x['negative_auc_abs_vs']
    x['crest_factor']=np.maximum(abs(p),abs(minimum))/np.maximum(r,1e-12)
    return np.column_stack([x[k] for k in P16]),P16,P16_GROUP

def prepare_inference(waves,base,payloads,cycle_ids=None):
    """Prepare data for all three seeds. Do not read composition labels."""
    if len(payloads)!=3 or [p['seed'] for p in payloads]!=SEEDS:raise ValueError('Use seeds 42, 123, and 2026 in that order.')
    if len({p['family'] for p in payloads})!=1:raise ValueError('Use one system at a time.')
    w=np.asarray(waves,dtype=np.float32)
    if len(w)!=len(base) or w.ndim!=2 or w.shape[1]!=256 or not np.isfinite(w).all() or not np.isfinite(base[BASE].to_numpy()).all():
        raise ValueError('Supply finite waveforms and matching physical rows.')
    family=payloads[0]['family'];terms,names,groups=mapped_terms(base,family)
    vpp=base.vpp_v.to_numpy(np.float64);safe=np.maximum(vpp,1e-9).astype(np.float32)
    shape=np.abs(w)/safe[:,None]
    prepared=dict(family=np.array(family),vpp=vpp,physical_terms=terms,term_names=np.array(names),term_groups=np.array(groups),
                  analysis_cycle_id=np.asarray(cycle_ids if cycle_ids is not None else np.arange(len(w)),dtype=str),
                  original_checkpoint_hashes=np.array([p['original_checkpoint_sha256'] for p in payloads]),seeds=np.array(SEEDS),
                  amplitude_scales=np.array([p['amplitude_scale'] for p in payloads]))
    for p in payloads:
        # NumPy keeps this computation in float32, as in the original channel preparation.
        channels=np.stack([w/np.float32(p['amplitude_scale']),shape],axis=1)
        prepared[f'channels_seed_{p["seed"]}']=channels
    return prepared

def main():
    ap=argparse.ArgumentParser(description='Prepare the frozen R01 inference inputs.')
    ap.add_argument('--package',type=Path,required=True);ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--waves',type=Path,required=True);ap.add_argument('--family',choices=['PGD-PGS','PGT-PGD'],required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    f=pd.read_csv(a.features);z=np.load(a.waves);rows=f[f.family.eq(a.family)].reset_index(drop=True)
    if f.analysis_cycle_id.duplicated().any():raise ValueError('Use unique cycle identifiers.')
    ix=pd.Index(z['analysis_cycle_id'].astype(str)).get_indexer(rows.analysis_cycle_id.astype(str))
    if (ix<0).any():raise ValueError('A waveform is missing.')
    payloads=[torch.load(a.package/'models/equivalent'/f'{a.family.lower()}-seed-{s}.pt',map_location='cpu',weights_only=False) for s in SEEDS]
    prepared=prepare_inference(z['waves'][ix],rows[BASE],payloads,rows.analysis_cycle_id)
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**prepared)
    print(json.dumps(dict(status='PASS',family=a.family,rows=len(rows),output=str(a.output),target_labels_read=False),ensure_ascii=False))

if __name__=='__main__':main()
