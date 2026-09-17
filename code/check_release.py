"""Check saved files and predictions without training or bootstrap calculation."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from predict import predict
from preprocessing import BASE
from equivalent import predict_prepared_ensemble

def main():
    parser=argparse.ArgumentParser(description='Check the frozen release without retraining.')
    parser.add_argument('--out',type=Path,default=ROOT/'qa/release-check-current.json')
    args=parser.parse_args();report={'file_checks':0,'controls':[],'r01':[],'retraining':False,'bootstrap_recomputed':False}
    with (ROOT/'manifest-sha256.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            path=(ROOT/row['path']).resolve();assert path.is_relative_to(ROOT.resolve())
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
            report['file_checks']+=1
    expected=pd.read_csv(ROOT/'results/predictions-target-free.csv')
    for family in ['PGD-PGS','PGT-PGD']:
        z=np.load(ROOT/'data'/f'inference-{family.lower()}.npz',allow_pickle=False)
        ids=z['analysis_cycle_id'].astype(str);base=pd.DataFrame(z['base'],columns=z['base_names'].astype(str))[BASE]
        result,_=predict(z['waves'],base,ids,family)
        ref=expected[expected.family.eq(family)&expected.context.eq('final')]
        merged=result.merge(ref,on=['analysis_cycle_id','family','context','model'],suffixes=('_new','_old'),validate='one_to_one')
        assert len(merged)==len(result)==len(ref)
        diff=float(np.max(np.abs(merged.predicted_pct_new-merged.predicted_pct_old)))
        assert diff<=.001
        report['controls'].append({'family':family,'rows':len(result),'max_difference_pp':diff})
        root=ROOT/'reference/r01';prepared=dict(np.load(root/'data'/f'prepared-{family.lower()}.npz',allow_pickle=False))
        payloads=[torch.load(root/'models/equivalent'/f'{family.lower()}-seed-{s}.pt',map_location='cpu',weights_only=False) for s in [42,123,2026]]
        value,_=predict_prepared_ensemble(prepared,payloads);value['analysis_cycle_id']=prepared['analysis_cycle_id']
        old=pd.read_csv(root/'cycle-predictions-and-contributions.csv');ref=old[old.family.eq(family)].set_index('analysis_cycle_id').loc[value.analysis_cycle_id]
        all_diff=float(np.max(np.abs(value.predicted_pct.to_numpy()-ref.predicted_pct.to_numpy())))
        frozen=pd.read_csv(root/'data/frozen-r01-confirmation.csv');frozen=frozen[frozen.family.eq(family)].set_index('analysis_cycle_id')
        fitted=value.set_index('analysis_cycle_id').loc[frozen.index]
        col='predicted_pct' if 'predicted_pct' in frozen else 'prediction_pct'
        frozen_diff=float(np.max(np.abs(fitted.predicted_pct.to_numpy()-frozen[col].to_numpy())))
        assert max(all_diff,frozen_diff)<=.001
        report['r01'].append({'family':family,'cycles':len(value),'all_cycle_max_difference_pp':all_diff,'frozen_confirmation_max_difference_pp':frozen_diff})
    report['status']='PASS';report['tolerance_pp']=.001
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
