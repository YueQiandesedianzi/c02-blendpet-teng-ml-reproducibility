"""Redraw selected panels from frozen cycles and prediction tables."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
MODELS=['OLS-V','Piecewise-V','V','C','P','C+V','C+P+V','C+P+V+route']
LABELS=['Linear V','Piecewise V','V','C','P','C+V','C+P+V','+ route']
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
    for fmt in ['png','svg']:fig.savefig(OUT/f'{name}.{fmt}',dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def box(ax,x,y,w,h,text):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',linewidth=.8,edgecolor='#59717F',facecolor='#EDF3F7'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1,color='#485B68'))
def main():
    OUT.mkdir(exist_ok=True)
    z=np.load(ROOT/'data/aligned-waves.npz');meta=pd.read_csv(ROOT/'data/cycle-ledger.csv');raw=pd.read_csv(ROOT/'data/source/physical-cycles-wide.csv')
    i=meta.index[meta.family.eq('PGD-PGS')&meta.target_pct.eq(25)][0]
    row=raw[raw.physical_cycle_id.eq(meta.physical_cycle_id[i])].iloc[0]
    w=row[[f'wave_voltage_{j:03d}_v' for j in range(256)]].to_numpy(float);xx=np.arange(256)
    left,right=np.median(w[:26]),np.median(w[-26:]);baseline=left+(right-left)/230*(xx-12.5)
    fig,ax=plt.subplots(1,2,figsize=(8,3.0),layout='constrained')
    ax[0].plot(xx,w,color='#9BA6AF',label='Frozen resampled cycle');ax[0].plot(xx,baseline,color='#C77B37',label='Edge baseline');ax[0].plot(xx,w-baseline,color='#245783',label='Corrected cycle')
    ax[0].set(xlabel='Point index',ylabel='Voltage (V)',title='a  Local baseline correction');ax[0].legend(frameon=False,fontsize=7.5)
    ax[1].plot(xx,z['waves'][i],color='#245783');ax[1].axvline(128,color='#C77B37',ls='--')
    ax[1].set(xlabel='Aligned point index',ylabel='Voltage (V)',title='b  Dominant peak alignment');save(fig,'figure-s2')
    fig,ax=plt.subplots(figsize=(8,3.1));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    box(ax,.01,.66,.22,.18,'N × 2 × 256 waveform');box(ax,.28,.66,.25,.18,'CNN → scalar evidence e');arrow(ax,(.23,.75),(.28,.75))
    box(ax,.01,.30,.22,.21,'Normalized P6 branch\nHistorical R01 only')
    box(ax,.59,.48,.38,.35,'Ordered logits: e + β·P6 + b − θk\nqk = sigmoid(logitk)\nTraining output = 25 Σqk')
    arrow(ax,(.53,.75),(.59,.70));arrow(ax,(.23,.40),(.59,.56))
    box(ax,.38,.03,.59,.25,'Loss = 0.7 Huber (δ = 5 pp) + 0.3 ordinal BCE\nNew controls omit β·P6\nThe CNN is shared within each system, fold and seed')
    arrow(ax,(.80,.48),(.80,.28));save(fig,'figure-s3')
    comp=pd.read_csv(ROOT/'results/metrics-by-composition.csv');vmax=np.ceil(comp.mae_pp.max()/5)*5
    fig,axs=plt.subplots(2,2,figsize=(8,6.0),layout='constrained')
    for j,family in enumerate(['PGD-PGS','PGT-PGD']):
        for i,role in enumerate(['BLOCKED_OOF','REUSED_CONFIRMATION']):
            ax=axs[i,j];g=comp[comp.family.eq(family)&comp.evaluation_role.eq(role)].pivot(index='model',columns='target_pct',values='mae_pp').loc[MODELS]
            im=ax.imshow(g.to_numpy(),aspect='auto',cmap='YlOrRd',vmin=0,vmax=vmax)
            for rr in range(8):
                for cc in range(len(g.columns)):ax.text(cc,rr,f'{g.iloc[rr,cc]:.1f}',ha='center',va='center',fontsize=8.5,color='white' if g.iloc[rr,cc]>.55*vmax else 'black')
            ax.set(yticks=range(8),yticklabels=LABELS,xticks=range(len(g.columns)),xticklabels=[str(int(x)) for x in g.columns],xlabel='Composition (%)',title=f'{chr(97+2*i+j)}  {family} | '+('Blocked OOF' if i==0 else 'Reused confirmation'))
            ax.tick_params(labelsize=9);ax.title.set_fontsize(10)
    cb=fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.75,pad=.025);cb.set_label('Composition MAE (pp)');save(fig,'figure-s4')
    pred=pd.read_csv(ROOT/'results/predictions-with-errors.csv');fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for ax,family in zip(axs,['PGD-PGS','PGT-PGD']):
        g=pred[pred.family.eq(family)&pred.evaluation_role.eq('REUSED_CONFIRMATION')]
        ax.boxplot([g[g.model.eq(m)].absolute_error_pp for m in MODELS],tick_labels=LABELS,showfliers=False,whis=(5,95))
        ax.tick_params(axis='x',rotation=50,labelsize=9);ax.set(ylabel='Cycle absolute error (pp)',title=family+' | reused confirmation')
    save(fig,'cycle-error-boxplots')
    print('The selected figures were redrawn from frozen values.')
if __name__=='__main__':main()
