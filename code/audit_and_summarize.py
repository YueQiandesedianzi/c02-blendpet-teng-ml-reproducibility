"""Verify fitted inputs. Summarize the locked predictions."""
import gzip,hashlib,json,math,shutil,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from analysis import ROOT,FAMILIES,SEEDS,MODELS,ANCHORS,load_data,make_splits,sha,dump,csv,physical_terms,evidence,WaveOrdinal,matrix,circular_indices,metric_rows,fit_head
from predict import predict
from preprocessing import BASE

def verify():
    meta,base,waves=load_data();splits=make_splits(meta)
    frozen=pd.read_csv(ROOT/'results/predictions-target-free.csv');checks=[];export=[]
    for lockname in ['protocol-lock','prediction-lock']:
        lock=json.loads((ROOT/f'provenance/{lockname}.json').read_text())
        for rel,h in lock['hashes'].items():assert sha(ROOT/rel)==h,(rel,'hash')
    for family in FAMILIES:
        mask=meta.family.eq(family).to_numpy();m=meta[mask].reset_index(drop=True);b=base[mask].reset_index(drop=True);w=waves[mask]
        np.savez_compressed(ROOT/'data'/f'inference-{family.lower()}.npz',waves=w,base=b.to_numpy(),base_names=np.array(BASE),analysis_cycle_id=m.analysis_cycle_id.to_numpy(str))
        for context in ['fold-0','fold-1','fold-2','final']:
            s=splits[splits.family.eq(family)&splits.context.eq(context)].set_index('analysis_cycle_id').loc[m.analysis_cycle_id]
            fit=np.flatnonzero(s.role.eq('fit'));fitids=m.analysis_cycle_id.iloc[fit].tolist()
            assert set(m.target_pct.iloc[fit])==set(ANCHORS)
            terms,names,bank=physical_terms(b,w,m,fit,family)
            saved=np.load(ROOT/'models'/context/f'{family.lower()}-templates.npz')['templates'];assert np.array_equal(bank,saved)
            for seed in SEEDS:
                ck=torch.load(ROOT/'models'/context/f'{family.lower()}-seed-{seed}-cnn.pt',weights_only=False,map_location='cpu')
                assert ck['fit_ids']==fitids and ck['epochs']==120 and ck['physical_training_fields']==[]
                model=WaveOrdinal();model.load_state_dict(ck['state_dict']);ev=evidence(model,w,ck['amplitude_scale'])
                history=pd.read_csv(ROOT/'logs'/f'{family.lower()}-{context}-seed-{seed}-training.csv');assert len(history)==120
                aug=pd.read_csv(ROOT/'logs'/f'{family.lower()}-{context}-seed-{seed}-augmentation.csv.gz',usecols=['analysis_cycle_id'])
                assert set(aug.analysis_cycle_id)<=set(fitids)
                for variant in MODELS:
                    h=json.loads((ROOT/'models'/context/f'{family.lower()}-{variant.lower()}-seed-{seed}-head.json').read_text())
                    assert h['fit_ids']==fitids
                    rebuilt=fit_head(ev,terms,names,b.vpp_v.to_numpy(),m.target_pct.to_numpy(),fit,variant)
                    for key in ['anchor_vpp','coefficient','intercept','center','scale','x','y']:
                        if key in h:assert np.allclose(h[key],rebuilt[key],rtol=1e-12,atol=1e-12),(family,context,variant,key)
                    if variant in ['C+P+V','C+P+V+route']:
                        export.append(dict(family=family,context=context,seed=seed,model=variant,fields=';'.join(h['fields']),n_features=len(h['fields']),fit_rows=len(fitids)))
                checks.append(dict(family=family,context=context,seed=seed,fit_cycles=len(fitids),augmentation_rows=len(aug),template_train_only=True,head_train_only=True,epochs=120))
            out,_=predict(w,b,m.analysis_cycle_id.to_numpy(str),family,context)
            expected=frozen[frozen.family.eq(family)&frozen.context.eq(context)]
            j=expected.merge(out,on=['analysis_cycle_id','family','context','model'],suffixes=('_frozen','_replay'),validate='one_to_one')
            assert len(j)==len(expected)
            delta=float(abs(j.predicted_pct_frozen-j.predicted_pct_replay).max());assert delta<1e-10
            checks.append(dict(family=family,context=context,prediction_rows=len(j),max_replay_difference_pp=delta,target_labels_read_for_inference=False))
    csv(ROOT/'qa/new-control-audit.csv',pd.DataFrame(checks));csv(ROOT/'results/input-matrix.csv',pd.DataFrame(export))
    summary=dict(status='PASS',checkpoints=24,heads=192,locked_predictions=True,outer_labels_used_for_fit=False,confirmation_labels_used_for_fit=False,templates_train_only=True,augmentation_parents_train_only=True,scope='The audit starts from the frozen QC pool. Historical QC used source-file context.')
    dump(ROOT/'qa/new-control-audit.json',summary);print(json.dumps(summary))

def statistics():
    meta,_,_=load_data();pred=pd.read_csv(ROOT/'results/predictions-with-errors.csv')
    pred=pred[pred.evaluation_role.ne('FINAL_TRAINING_REPLAY')]
    csv(ROOT/'results/metrics-by-fold.csv',metric_rows(pred,['evaluation_role','family','context','model','target_pct']))
    seeds=pd.read_csv(ROOT/'results/seed-predictions-target-free.csv').merge(meta[['analysis_cycle_id','source_file','target_pct']],on='analysis_cycle_id',validate='many_to_one')
    seeds['evaluation_role']=np.where(seeds.context.ne('final'),'BLOCKED_OOF',np.where(seeds.target_pct.isin(ANCHORS),'FINAL_TRAINING_REPLAY','REUSED_CONFIRMATION'))
    seeds['error_pp']=seeds.predicted_pct-seeds.target_pct;seeds['absolute_error_pp']=abs(seeds.error_pp)
    seeds.to_csv(ROOT/'results/seed-predictions-with-errors.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
    csv(ROOT/'results/metrics-by-seed-source.csv',metric_rows(seeds[seeds.evaluation_role.ne('FINAL_TRAINING_REPLAY')],['evaluation_role','family','model','seed','target_pct','source_file']))
    comparisons=[('C+V','C'),('C+P+V','C+V'),('C+P+V','P'),('C+P+V+route','C+P+V'),('C+P+V','OLS-V'),('C+P+V','Piecewise-V')]
    rows=[];ci=[]
    for ri,role in enumerate(['REUSED_CONFIRMATION','BLOCKED_OOF']):
        for fi,family in enumerate(FAMILIES):
            bootgroups=[];obs=[]
            subset=pred[pred.family.eq(family)&pred.evaluation_role.eq(role)]
            for ti,target in enumerate(sorted(subset.target_pct.unique())):
                g=subset[subset.target_pct.eq(target)];order=meta[meta.family.eq(family)&meta.target_pct.eq(target)].sort_values(['source_file','valid_block_id','cycle_index','start_s'])
                err=g.pivot(index='analysis_cycle_id',columns='model',values='absolute_error_pp').loc[order.analysis_cycle_id,MODELS].to_numpy()
                rng=np.random.default_rng(20260917+ri*100000+fi*10000+ti*100);boot=np.zeros((20000,8))
                for source in sorted(order.source_file.unique()):
                    e=err[order.source_file.to_numpy()==source];n=len(e);block=math.ceil(n**(1/3))
                    for k in range(0,20000,500):
                        ix=circular_indices(n,block,rng,500);boot[k:k+500]+=e[ix].sum(1)/len(err)
                observed=err.mean(0);bootgroups.append(boot);obs.append(observed)
                for j,name in enumerate(MODELS):
                    lo,hi=np.quantile(boot[:,j],[.025,.975]);ci.append(dict(evaluation_role=role,family=family,target_pct=target,model=name,mae_pp=observed[j],ci95_low_pp=lo,ci95_high_pp=hi))
                for test,reference in comparisons:
                    j,k=MODELS.index(test),MODELS.index(reference);lo,hi=np.quantile(boot[:,j]-boot[:,k],[.025,.975]);rows.append(dict(evaluation_role=role,family=family,target_pct=str(int(target)),model=test,reference=reference,delta_mae_pp=observed[j]-observed[k],ci95_low_pp=lo,ci95_high_pp=hi))
            for aggregation,boot,observed in [('Mean MAE',np.mean(bootgroups,axis=0),np.mean(obs,axis=0)),('Worst MAE',np.max(bootgroups,axis=0),np.max(obs,axis=0))]:
                for test,reference in comparisons:
                    j,k=MODELS.index(test),MODELS.index(reference);lo,hi=np.quantile(boot[:,j]-boot[:,k],[.025,.975]);rows.append(dict(evaluation_role=role,family=family,target_pct=aggregation,model=test,reference=reference,delta_mae_pp=observed[j]-observed[k],ci95_low_pp=lo,ci95_high_pp=hi))
    csv(ROOT/'results/paired-bootstrap-all.csv',pd.DataFrame(rows));csv(ROOT/'results/mae-bootstrap-all.csv',pd.DataFrame(ci))
    q=pd.read_csv(ROOT/'data/source/qc-summary.csv');p=pd.read_csv(ROOT/'data/source/physical-cycles-wide.csv');used=set(meta.physical_cycle_id)
    removed=p[~p.physical_cycle_id.isin(used)].copy();csv(ROOT/'results/historical-omitted-cycles.csv',removed.iloc[:,:min(30,len(removed.columns))])
    counts=meta.groupby('source_file').agg(analysis_rows=('analysis_cycle_id','size'),unique_analysis_cycles=('physical_cycle_id','nunique')).reset_index()
    q=q.merge(counts,on='source_file',how='left');q['post_qc_omitted']=q.qc_cycles-q.unique_analysis_cycles
    csv(ROOT/'results/qc-and-analysis-counts.csv',q)
    sp=pd.read_csv(ROOT/'data/split-ledger.csv');csv(ROOT/'results/split-summary.csv',sp.groupby(['family','context','source_file','target_pct','role'],as_index=False).size())
    print('The summary tables are saved.')

if __name__=='__main__':
    verify();statistics()
