"""Train the fixed controls. Save target-free predictions before evaluation."""
from __future__ import annotations
import argparse,csv as csvlib,gzip,hashlib,importlib.util,json,math,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LinearRegression
from preprocessing import BASE,mapped_terms
from equivalent import interval_router,predict_prepared_ensemble

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'provenance/protocol.json').read_text(encoding='utf-8'))
FAMILIES=P['families'];ANCHORS=P['anchors_pct'];SEEDS=P['seeds'];MODELS=P['models']
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    x.to_csv(p,index=False,encoding='utf-8-sig')
def now():return datetime.now(timezone.utc).isoformat()
def log(s):print(f'{now()} {s}',flush=True)
def wave_source():
    spec=importlib.util.spec_from_file_location('wave_source',ROOT/'code/r01_wave_source.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod
def load_data():
    meta=pd.read_csv(ROOT/'data/cycle-ledger.csv');base=pd.read_csv(ROOT/'data/base-features.csv')
    z=np.load(ROOT/'data/aligned-waves.npz',allow_pickle=False)
    assert np.array_equal(meta.analysis_cycle_id.to_numpy(str),z['analysis_cycle_id'])
    assert np.array_equal(meta.analysis_cycle_id.to_numpy(str),base.analysis_cycle_id.to_numpy(str))
    assert len(meta)==2227 and meta.physical_cycle_id.nunique()==2000
    assert meta.groupby('family').size().to_dict()==dict(zip(FAMILIES,[1075,1152]))
    return meta,base[BASE],z['waves']
def make_splits(meta):
    folds=pd.Series(-1,index=meta.index)
    for (_,source),g in meta.groupby(['family','source_file'],sort=True):
        if not g.target_pct.isin(ANCHORS).all():continue
        ix=g.sort_values(['valid_block_id','cycle_index','start_s']).index.to_numpy()
        for k,part in enumerate(np.array_split(ix,3)):folds.loc[part]=k
    records=[]
    for family in FAMILIES:
        mask=meta.family.eq(family);m=meta[mask].reset_index(drop=True);ff=folds[mask].to_numpy()
        for k in [0,1,2,-1]:
            role=np.where(m.target_pct.isin(ANCHORS),'fit','REUSED_CONFIRMATION').astype(object)
            if k>=0:
                role[ff==k]='validation'
                for source,g in m[m.target_pct.isin(ANCHORS)].groupby('source_file',sort=True):
                    ix=g.sort_values(['valid_block_id','cycle_index','start_s']).index.to_numpy()
                    positions=np.flatnonzero(ff[ix]==k);gap=math.ceil(len(ix)**(1/3))
                    if len(positions):
                        near=np.r_[ix[max(0,positions.min()-gap):positions.min()],ix[positions.max()+1:positions.max()+1+gap]]
                        role[near]='embargo'
            out=m[['analysis_cycle_id','physical_cycle_id','family','source_file','target_pct']].copy()
            out['context']='final' if k<0 else f'fold-{k}';out['role']=role;records.append(out)
    s=pd.concat(records,ignore_index=True)
    for (_,context),g in s.groupby(['family','context']):
        fit=g[g.role.eq('fit')];valid=g[g.role.eq('validation')]
        assert set(fit.target_pct)==set(ANCHORS)
        assert not set(fit.physical_cycle_id)&set(valid.physical_cycle_id)
    assert s[s.source_file.eq('100PGD.csv')].groupby(['physical_cycle_id','context']).role.nunique().max()==1
    return s

class WaveOrdinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv1d(2,16,9,padding=4),nn.GroupNorm(4,16),nn.ReLU(),nn.MaxPool1d(2))
        self.conv2=nn.Sequential(nn.Conv1d(16,32,5,padding=2),nn.GroupNorm(8,32),nn.ReLU(),nn.MaxPool1d(2))
        self.wave_projection=nn.Linear(32,8);self.wave_head=nn.Linear(8,1)
        self.evidence_bias=nn.Parameter(torch.tensor(0.))
        self.threshold_base=nn.Parameter(torch.tensor(-1.5))
        self.threshold_delta_raw=nn.Parameter(torch.full((3,),math.log(math.expm1(1.))))
    def evidence(self,x):
        z=self.conv2(self.conv1(x));z=F.adaptive_avg_pool1d(z,1).flatten(1)
        return self.wave_head(self.wave_projection(z)).reshape(-1)
    def forward(self,x):
        e=self.evidence(x)+self.evidence_bias
        steps=F.softplus(self.threshold_delta_raw)+1e-4
        thresholds=torch.cat([self.threshold_base.reshape(1),self.threshold_base+torch.cumsum(steps,0)])
        logits=e[:,None]-thresholds[None,:]
        return 25*torch.sigmoid(logits).sum(1),logits
def channels(w,scale):
    w=np.asarray(w,np.float32);v=np.maximum(np.ptp(w,axis=1),1e-9)
    return np.stack([w/np.float32(scale),np.abs(w)/v[:,None]],1).astype(np.float32)
def normalize_rows(w):
    x=np.asarray(w,float)-np.mean(w,axis=1,keepdims=True)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def physical_terms(base,waves,meta,fit,family,bank=None):
    terms,names,_=mapped_terms(base,family)
    if family=='PGT-PGD':
        z=waves/np.maximum(np.ptp(waves,axis=1),1e-9)[:,None]
        if bank is None:
            bank=normalize_rows(np.stack([np.median(z[fit[meta.source_file.to_numpy()[fit]==s]],axis=0) for s in sorted(meta.source_file.iloc[fit].unique())]))
        terms[:,names.index('template_corr')]=np.max(normalize_rows(z)@bank.T,axis=1)
        d=np.diff(waves.astype(float),axis=1)
        noise=1.4826*np.median(abs(d-np.median(d,axis=1,keepdims=True)),axis=1)/np.sqrt(2.)
        terms[:,names.index('snr_proxy_db')]=20*np.log10(np.maximum(np.sqrt(np.mean(waves.astype(float)**2,axis=1)),1e-12)/np.maximum(noise,1e-12))
    else:bank=np.empty((0,256))
    return terms,names,bank
def augment(waves,indices,seed,epoch,writer,ids):
    rng=np.random.default_rng(seed+epoch*3571);n=len(indices);w=waves[indices]
    shifts=rng.integers(-4,5,n);ts=rng.uniform(.98,1.02,n);amp=rng.uniform(.98,1.02,n);nf=rng.uniform(0,.005,n)
    axis=np.linspace(-1,1,256);out=np.empty_like(w)
    for j in range(n):
        x=np.roll(w[j],shifts[j]);x=np.interp(np.clip(axis/ts[j],-1,1),axis,x)*amp[j]
        out[j]=x+rng.normal(size=256)*(np.ptp(x)*nf[j])
        writer.writerow([epoch,j,ids[indices[j]],seed+epoch*3571,int(shifts[j]),ts[j],amp[j],nf[j],j*256])
    return out
def train_cnn(waves,meta,fit,family,context,seed):
    path=ROOT/'models'/context/f'{family.lower()}-seed-{seed}-cnn.pt';path.parent.mkdir(parents=True,exist_ok=True)
    id_hash=hashlib.sha256('\n'.join(meta.analysis_cycle_id.iloc[fit]).encode()).hexdigest()
    if path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert payload['fit_id_sha256']==id_hash and payload['protocol_sha256']==sha(ROOT/'provenance/protocol.json')
        model=WaveOrdinal();model.load_state_dict(payload['state_dict']);return model,payload
    torch.manual_seed(seed);np.random.seed(seed);torch.use_deterministic_algorithms(True)
    model=WaveOrdinal();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    src=wave_source();scale=src.robust_scale(waves[fit].reshape(-1))
    labels=meta.target_pct.to_numpy(np.float32);sources=meta.source_file.to_numpy(str)
    augpath=ROOT/'logs'/f'{family.lower()}-{context}-seed-{seed}-augmentation.csv.gz'
    history=[];start=time.monotonic();log(f'Train {family} {context} seed {seed}.')
    with gzip.open(augpath,'wt',encoding='utf-8',newline='') as f:
        writer=csvlib.writer(f);writer.writerow(['epoch','draw_index','analysis_cycle_id','augmentation_seed','shift_points','time_scale','amplitude_scale','noise_fraction','noise_draw_offset'])
        for epoch in range(1,121):
            ix=src.balanced_source_composition_indices(fit,labels,sources,seed+epoch*1009)
            w=augment(waves,ix,seed,epoch,writer,meta.analysis_cycle_id.to_numpy(str))
            x=torch.from_numpy(channels(w,scale));y=torch.from_numpy(labels[ix]);losses=[];model.train()
            for b in range(0,len(x),64):
                pred,logits=model(x[b:b+64]);target=y[b:b+64]
                yt=(target[:,None]>=torch.tensor([25.,50.,75.,100.])[None,:]).float()
                loss=.7*F.huber_loss(pred,target,delta=5.)+.3*F.binary_cross_entropy_with_logits(logits,yt)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),training_draws=len(ix)))
            if epoch%40==0:log(f'{family} {context} seed {seed}: epoch {epoch}.')
    payload=dict(state_dict=model.state_dict(),family=family,context=context,seed=seed,fit_id_sha256=id_hash,
                 fit_ids=meta.analysis_cycle_id.iloc[fit].tolist(),protocol_sha256=sha(ROOT/'provenance/protocol.json'),
                 amplitude_scale=scale,physical_training_fields=[],epochs=120,seconds=time.monotonic()-start,
                 parameter_count=sum(p.numel() for p in model.parameters()))
    torch.save(payload,path);csv(ROOT/'logs'/f'{family.lower()}-{context}-seed-{seed}-training.csv',pd.DataFrame(history))
    return model,payload
@torch.no_grad()
def evidence(model,waves,scale):
    model.eval();x=torch.from_numpy(channels(waves,scale))
    return np.concatenate([model.evidence(c).numpy() for c in x.split(256)]).astype(float)
def matrix(ev,terms,names,vpp,variant):
    fields=[];cols=[]
    if variant in ['C','C+V','C+P+V','C+P+V+route']:fields.append('wave_evidence');cols.append(ev)
    if variant in ['V','C+V','C+P+V','C+P+V+route']:fields.append('vpp_v');cols.append(vpp)
    if variant in ['P','C+P+V','C+P+V+route']:
        for k,n in enumerate(names):
            if n not in fields:fields.append(n);cols.append(terms[:,k])
    return np.column_stack(cols),fields
def fit_head(ev,terms,names,vpp,y,fit,variant):
    av=[float(np.median(vpp[fit[y[fit]==a]])) for a in ANCHORS]
    if variant=='OLS-V':
        model=LinearRegression().fit(np.array(av)[:,None],ANCHORS)
        return dict(variant=variant,anchor_vpp=av,coefficient=model.coef_.tolist(),intercept=float(model.intercept_))
    if variant=='Piecewise-V':
        med=pd.DataFrame({'x':av,'y':ANCHORS}).groupby('x',sort=True).y.mean()
        return dict(variant=variant,anchor_vpp=av,x=med.index.tolist(),y=med.to_list())
    x,fields=matrix(ev,terms,names,vpp,variant)
    scaler=StandardScaler().fit(x[fit]);z=scaler.transform(x)
    subsets=[fit] if variant!='C+P+V+route' else [fit[np.isin(y[fit],[25*k,25*(k+1)])] for k in range(4)]
    models=[Ridge(alpha=100.).fit(z[ix],y[ix]) for ix in subsets]
    return dict(variant=variant,anchor_vpp=av,fields=fields,center=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                coefficient=[m.coef_.tolist() for m in models],intercept=[float(m.intercept_) for m in models])
def apply_head(ev,terms,names,vpp,head):
    variant=head['variant'];route=np.full(len(vpp),-1,dtype=int)
    if variant=='OLS-V':out=vpp*head['coefficient'][0]+head['intercept']
    elif variant=='Piecewise-V':out=np.interp(vpp,head['x'],head['y'])
    else:
        x,fields=matrix(ev,terms,names,vpp,variant);assert fields==head['fields']
        z=(x-np.array(head['center']))/np.array(head['scale'])
        coef=np.array(head['coefficient']);inter=np.array(head['intercept'])
        if variant=='C+P+V+route':
            route=interval_router(vpp,head['anchor_vpp']);out=(z*coef[route]).sum(1)+inter[route]
        else:out=z@coef[0]+inter[0]
    return out,route
def freeze_protocol():
    paths=[ROOT/'provenance/protocol.json',*[ROOT/'code'/n for n in ['analysis.py','preprocessing.py','equivalent.py','r01_wave_source.py']],*[ROOT/'data'/n for n in ['aligned-waves.npz','base-features.csv','cycle-ledger.csv']]]
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    lock=ROOT/'provenance/protocol-lock.json'
    if lock.exists():assert json.loads(lock.read_text())['hashes']==hashes,'The locked inputs changed.'
    else:dump(lock,dict(locked_at_utc=now(),hashes=hashes))
def run_training():
    freeze_protocol();meta,base,waves=load_data();splits=make_splits(meta);csv(ROOT/'data/split-ledger.csv',splits)
    all_ensemble=[];all_seed=[];fitting=[]
    for family in FAMILIES:
        mask=meta.family.eq(family).to_numpy();m=meta[mask].reset_index(drop=True);b=base[mask].reset_index(drop=True);w=waves[mask]
        y=m.target_pct.to_numpy(float);vpp=b.vpp_v.to_numpy(float)
        for context in ['fold-0','fold-1','fold-2','final']:
            s=splits[splits.family.eq(family)&splits.context.eq(context)].set_index('analysis_cycle_id').loc[m.analysis_cycle_id]
            fit=np.flatnonzero(s.role.eq('fit').to_numpy());evaluate=np.arange(len(m)) if context=='final' else np.flatnonzero(s.role.eq('validation').to_numpy())
            terms,names,bank=physical_terms(b,w,m,fit,family)
            d=ROOT/'models'/context;d.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(d/f'{family.lower()}-templates.npz',templates=bank)
            sums={v:[] for v in MODELS}
            for seed in SEEDS:
                model,payload=train_cnn(w,m,fit,family,context,seed);ev=evidence(model,w,payload['amplitude_scale'])
                for variant in MODELS:
                    h=fit_head(ev,terms,names,vpp,y,fit,variant)
                    h.update(family=family,context=context,seed=seed,fit_ids=m.analysis_cycle_id.iloc[fit].tolist(),protocol_sha256=sha(ROOT/'provenance/protocol.json'))
                    dump(d/f'{family.lower()}-{variant.lower()}-seed-{seed}-head.json',h)
                    pred,route=apply_head(ev,terms,names,vpp,h);sums[variant].append(pred)
                    out=pd.DataFrame(dict(analysis_cycle_id=m.analysis_cycle_id.to_numpy()[evaluate],family=family,context=context,model=variant,seed=seed,preclip_pct=pred[evaluate],predicted_pct=np.clip(pred[evaluate],0,100),interval_index=route[evaluate]))
                    all_seed.append(out)
                    fitting.append(dict(family=family,context=context,seed=seed,model=variant,fit_rows=len(fit),cnn_fit_id_sha256=payload['fit_id_sha256'] if 'C' in variant else '',physical_fields=json.dumps(names if variant in ['P','C+P+V','C+P+V+route'] else []),head_fields=json.dumps(h.get('fields',['vpp_v'])),amplitude_scale=payload['amplitude_scale'] if 'C' in variant else None))
            for variant in MODELS:
                pred=np.mean(sums[variant],axis=0)
                all_ensemble.append(pd.DataFrame(dict(analysis_cycle_id=m.analysis_cycle_id.to_numpy()[evaluate],family=family,context=context,model=variant,preclip_pct=pred[evaluate],predicted_pct=np.clip(pred[evaluate],0,100))))
            log(f'Completed {family} {context}.')
    ensemble=pd.concat(all_ensemble,ignore_index=True);seeds=pd.concat(all_seed,ignore_index=True)
    csv(ROOT/'results/predictions-target-free.csv',ensemble);csv(ROOT/'results/seed-predictions-target-free.csv',seeds);csv(ROOT/'results/model-fit-ledger.csv',pd.DataFrame(fitting))
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'models').rglob('*')) if p.is_file()}
    hashes.update({p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'results/predictions-target-free.csv',ROOT/'results/seed-predictions-target-free.csv']})
    dump(ROOT/'provenance/prediction-lock.json',dict(locked_at_utc=now(),protocol_sha256=sha(ROOT/'provenance/protocol.json'),hashes=hashes))
    log('All target-free predictions are locked. Evaluation has not run.')
def metric_rows(frame,keys):
    rows=[]
    for key,g in frame.groupby(keys,sort=True):
        if not isinstance(key,tuple):key=(key,)
        err=g.predicted_pct.to_numpy()-g.target_pct.to_numpy();a=abs(err)
        rows.append(dict(zip(keys,key))|dict(n=len(g),source_files=g.source_file.nunique(),mae_pp=float(a.mean()),rmse_pp=float(np.sqrt(np.mean(err**2))),bias_pp=float(err.mean()),median_ae_pp=float(np.median(a)),p95_ae_pp=float(np.quantile(a,.95)),max_ae_pp=float(a.max()),within_4pp=float(np.mean(a<=4)),clipped_fraction=float(np.mean(g.preclip_pct!=g.predicted_pct))))
    return pd.DataFrame(rows)
def circular_indices(n,block,rng,count):
    starts=rng.integers(0,n,size=(count,math.ceil(n/block)))
    return ((starts[:,:,None]+np.arange(block))%n).reshape(count,-1)[:,:n]
def evaluate():
    lock=json.loads((ROOT/'provenance/prediction-lock.json').read_text());assert lock['protocol_sha256']==sha(ROOT/'provenance/protocol.json')
    for rel,h in lock['hashes'].items():assert sha(ROOT/rel)==h
    meta,_,_=load_data();cols=['analysis_cycle_id','physical_cycle_id','source_file','target_pct','cycle_index','valid_block_id','start_s']
    pred=pd.read_csv(ROOT/'results/predictions-target-free.csv').merge(meta[cols],on='analysis_cycle_id',validate='many_to_one')
    pred['evaluation_role']=np.where(pred.context.ne('final'),'BLOCKED_OOF',np.where(pred.target_pct.isin(ANCHORS),'FINAL_TRAINING_REPLAY','REUSED_CONFIRMATION'))
    pred['error_pp']=pred.predicted_pct-pred.target_pct;pred['absolute_error_pp']=abs(pred.error_pp)
    seeds=pd.read_csv(ROOT/'results/seed-predictions-target-free.csv').merge(meta[cols],on='analysis_cycle_id',validate='many_to_one')
    seeds['evaluation_role']=np.where(seeds.context.ne('final'),'BLOCKED_OOF',np.where(seeds.target_pct.isin(ANCHORS),'FINAL_TRAINING_REPLAY','REUSED_CONFIRMATION'))
    csv(ROOT/'results/predictions-with-errors.csv',pred)
    included=pred[pred.evaluation_role.ne('FINAL_TRAINING_REPLAY')]
    comp=metric_rows(included,['evaluation_role','family','model','target_pct'])
    csv(ROOT/'results/metrics-by-composition.csv',comp)
    csv(ROOT/'results/metrics-by-source.csv',metric_rows(included,['evaluation_role','family','model','target_pct','source_file']))
    sm=metric_rows(seeds[seeds.evaluation_role.ne('FINAL_TRAINING_REPLAY')],['evaluation_role','family','model','seed','target_pct'])
    csv(ROOT/'results/metrics-by-seed-composition.csv',sm)
    ss=sm.groupby(['evaluation_role','family','model','seed'],as_index=False).agg(mean_mae_pp=('mae_pp','mean'),worst_mae_pp=('mae_pp','max'))
    csv(ROOT/'results/metrics-by-seed.csv',ss)
    summary=comp.groupby(['evaluation_role','family','model'],as_index=False).agg(mean_mae_pp=('mae_pp','mean'),worst_mae_pp=('mae_pp','max'))
    stable=ss.groupby(['evaluation_role','family','model'],as_index=False).agg(seed_mae_min_pp=('mean_mae_pp','min'),seed_mae_max_pp=('mean_mae_pp','max'),seed_mae_sd_pp=('mean_mae_pp','std'))
    summary=summary.merge(stable);csv(ROOT/'results/metrics-summary.csv',summary)
    confirmation=pred[pred.evaluation_role.eq('REUSED_CONFIRMATION')];ci=[];paired=[];bootstrap={}
    comparisons=[('C+V','C'),('C+P+V','C+V'),('C+P+V','P'),('C+P+V+route','C+P+V'),('C+P+V','OLS-V'),('C+P+V','Piecewise-V')]
    for fi,family in enumerate(FAMILIES):
        group_boot=[]
        for ti,target in enumerate([10,45,60]):
            g=confirmation[confirmation.family.eq(family)&confirmation.target_pct.eq(target)]
            order=meta[meta.family.eq(family)&meta.target_pct.eq(target)].sort_values(['source_file','valid_block_id','cycle_index','start_s'])
            errors=g.pivot(index='analysis_cycle_id',columns='model',values='absolute_error_pp').loc[order.analysis_cycle_id,MODELS].to_numpy()
            rng=np.random.default_rng(20260917+fi*10000+ti*100);boot=np.zeros((20000,len(MODELS)));lengths={}
            for source in sorted(order.source_file.unique()):
                e=errors[order.source_file.to_numpy()==source];n=len(e);block=math.ceil(n**(1/3));lengths[source]=block
                for start in range(0,20000,500):
                    ix=circular_indices(n,block,rng,min(500,20000-start));boot[start:start+len(ix)]+=e[ix].sum(1)/len(errors)
            bootstrap[f'{family}-{target}']=boot;group_boot.append(boot)
            for j,model in enumerate(MODELS):
                lo,hi=np.quantile(boot[:,j],[.025,.975]);ci.append(dict(family=family,target_pct=target,model=model,n=len(errors),mae_pp=float(errors[:,j].mean()),ci95_low_pp=float(lo),ci95_high_pp=float(hi),block_lengths=json.dumps(lengths),evaluation_role='REUSED_CONFIRMATION'))
            for test,reference in comparisons:
                j,k=MODELS.index(test),MODELS.index(reference);delta=boot[:,j]-boot[:,k];lo,hi=np.quantile(delta,[.025,.975])
                paired.append(dict(family=family,target_pct=target,model=test,reference=reference,delta_mae_pp=float(errors[:,j].mean()-errors[:,k].mean()),ci95_low_pp=float(lo),ci95_high_pp=float(hi),evaluation_role='REUSED_CONFIRMATION'))
        means=np.mean(group_boot,axis=0);worsts=np.max(group_boot,axis=0)
        bootstrap[f'{family}-mean']=means;bootstrap[f'{family}-worst']=worsts
    csv(ROOT/'results/confirmation-mae-ci.csv',pd.DataFrame(ci));csv(ROOT/'results/paired-mae-differences.csv',pd.DataFrame(paired))
    np.savez_compressed(ROOT/'results/bootstrap-replicates.npz',**bootstrap)
    dump(ROOT/'provenance/evaluation-record.json',dict(evaluated_at_utc=now(),prediction_lock_sha256=sha(ROOT/'provenance/prediction-lock.json'),replicates=20000,selection_after_evaluation=False,role='REUSED_CONFIRMATION'))
    log('Evaluation is complete. No model was selected from confirmation errors.')
    print(summary.to_string(index=False))
def replay_r01():
    root=ROOT/'reference/r01';reports=[];rows=[]
    old=pd.read_csv(root/'cycle-predictions-and-contributions.csv')
    frozen=pd.read_csv(root/'data/frozen-r01-confirmation.csv')
    for family in FAMILIES:
        prepared=dict(np.load(root/'data'/f'prepared-{family.lower()}.npz',allow_pickle=False))
        payloads=[torch.load(root/'models/equivalent'/f'{family.lower()}-seed-{s}.pt',map_location='cpu',weights_only=False) for s in SEEDS]
        result,_=predict_prepared_ensemble(prepared,payloads);result['analysis_cycle_id']=prepared['analysis_cycle_id'];result['family']=family
        ref=old[old.family.eq(family)].set_index('analysis_cycle_id').loc[result.analysis_cycle_id]
        diff=float(np.max(abs(result.predicted_pct.to_numpy()-ref.predicted_pct.to_numpy())))
        matching=frozen[frozen.family.eq(family)].set_index('analysis_cycle_id')
        test=result.set_index('analysis_cycle_id').loc[matching.index]
        col='predicted_pct' if 'predicted_pct' in matching else 'prediction_pct'
        fdiff=float(np.max(abs(test.predicted_pct.to_numpy()-matching[col].to_numpy())))
        assert diff<=.001 and fdiff<=.001
        reports.append(dict(family=family,n=len(result),replay_max_difference_pp=diff,frozen_confirmation_max_difference_pp=fdiff,status='PASS'))
        rows.append(result)
    csv(ROOT/'results/r01-replay.csv',pd.concat(rows,ignore_index=True));dump(ROOT/'qa/r01-replay.json',dict(status='PASS',tolerance_pp=.001,systems=reports))
    print(json.dumps(reports,indent=2))
def main():
    parser=argparse.ArgumentParser(description='Run the fixed analysis.');parser.add_argument('action',choices=['train','evaluate','replay-r01'])
    a=parser.parse_args()
    if a.action=='train':run_training()
    elif a.action=='evaluate':evaluate()
    else:replay_r01()
if __name__=='__main__':main()
