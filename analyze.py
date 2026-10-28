"""Regenerate figures and LaTeX tables from frozen experiment receipts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import METHODS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis'; OUT.mkdir(exist_ok=True)
RUNS=json.loads((ROOT/'results/all_runs.json').read_text())
COLORS=['#6a737d','#4c78a8','#e69f00','#009e73','#9c4fa3','#d55e00']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                     'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
NAMES=['Cancer','Digits','Bank']

def select(**kwargs):
    return [r for r in RUNS if all(r[k]==v for k,v in kwargs.items())]

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)


fig,axes=plt.subplots(2,3,figsize=(7,4.3))
for col,n in enumerate(NAMES):
    for m,color in zip(METHODS,COLORS):
        r=select(dataset=n,seed=0,group='main',method=m)[0];h=r['history']
        for row,xkey in enumerate(['passes','seconds']):
            ax=axes[row,col];xs=[max(t[xkey],1e-5) for t in h];ys=[t['bound'] for t in h]
            ax.plot(xs,ys,color=color,label=m,lw=1.25)
            if r['status']!='certified': ax.plot(xs[-1],ys[-1],'x',color=color,ms=5)
            ax.set_yscale('log');ax.set_xscale('log');ax.axhline(1e-6,color='black',ls=':',lw=.8)
            ax.set_ylim(1e-11,1e3);ax.grid(alpha=.18)
    axes[0,col].set_title(n);axes[0,col].set_xlabel('Data passes (log scale)')
    axes[1,col].set_xlabel('Solve time (seconds, log scale)')
axes[0,0].set_ylabel('Certified error bound');axes[1,0].set_ylabel('Certified error bound')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=6,bbox_to_anchor=(.5,-.015))
fig.tight_layout(rect=(0,.05,1,1));save(fig,'convergence')

fig,axes=plt.subplots(1,3,figsize=(7,2.35))
for ax,n in zip(axes,NAMES):
    for m,color in zip(METHODS,COLORS):
        rr=sorted([r for r in select(dataset=n,seed=0,scaled=True,method=m) if r['group'] in ['main','regularization']],key=lambda r:r['lam'])
        ax.plot([r['lam'] for r in rr],[r['iterations'] for r in rr],'-o',ms=3,color=color,label=m,lw=1)
        for r in rr:
            if r['status']!='certified':ax.plot(r['lam'],r['iterations'],'x',color=color,ms=7,mew=1.3)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Regularization lambda');ax.set_title(n);ax.grid(alpha=.18)
axes[0].set_ylabel('Iterations (cap = 2500)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=6,bbox_to_anchor=(.5,-.05))
fig.tight_layout(rect=(0,.08,1,1));save(fig,'regularization')

fig,axes=plt.subplots(1,2,figsize=(7,2.25))
for ax,n in zip(axes,['Cancer','Bank']):
    for m,color in zip(METHODS,COLORS):
        r=select(dataset=n,group='scaling',method=m)[0];h=r['history']
        ax.plot([t['passes'] for t in h],[t['bound'] for t in h],color=color,label=m,lw=1.2)
        if r['status']!='certified':ax.plot(h[-1]['passes'],h[-1]['bound'],'x',color=color,ms=5)
    ax.set_xscale('log');ax.set_yscale('log');ax.axhline(1e-6,color='black',ls=':',lw=.8)
    ax.set_xlabel('Data passes');ax.set_title(n+' in raw numeric units');ax.grid(alpha=.18)
axes[0].set_ylabel('Certified error bound')
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=6,bbox_to_anchor=(.5,-.06))
fig.tight_layout(rect=(0,.08,1,1));save(fig,'raw_units')

def write(name,lines): (OUT/name).write_text('\n'.join(lines)+'\n'+r'\bottomrule'+'\n')
lines=[]
for m in METHODS:
    cells=[]
    for n in NAMES:
        rr=select(dataset=n,group='main',method=m)
        it=int(np.median([r['iterations'] for r in rr]));secs=np.median([r['seconds'] for r in rr])
        ok=sum(r['status']=='certified' for r in rr)
        cells.append(f'{it:,} / {secs:.4f} / {ok}/3')
    lines.append(m+' & '+' & '.join(cells)+r' \\')
write('main_table.tex',lines)
lines=[]
for m in METHODS:
    cells=[]
    for n in NAMES:
        rr=select(dataset=n,group='main',method=m)
        med=int(np.median([r['passes'] for r in rr]));lo=min(r['seconds'] for r in rr);hi=max(r['seconds'] for r in rr)
        cells.append(f'{med:,} / [{lo:.4f}, {hi:.4f}]')
    lines.append(m+' & '+' & '.join(cells)+r' \\')
write('work_table.tex',lines)
lines=[]
for n in NAMES:
    r=select(dataset=n,group='main',method='GD',seed=0)[0]
    lines.append(f"{n} & {r['ntrain']:,} / {r['ntest']:,} & {r['d']} & {r['train_positive']*100:.1f}\\% & {r['L']:.3f} & {r['condition_upper']:,.0f}"+r' \\')
write('data_table.tex',lines)
lines=[]
for n in NAMES:
    rr=select(dataset=n,group='main',method='Newton-CG')
    cells=[f"{np.mean([r[k] for r in rr]):.4f} $\\pm$ {np.std([r[k] for r in rr],ddof=1):.4f}" for k in ['test_accuracy','test_balanced_accuracy','test_auc','test_logloss']]
    lines.append(n+' & '+' & '.join(cells)+r' \\')
write('predict_table.tex',lines)
lines=[]
for n in ['Cancer','Bank']:
    for m in METHODS:
        r=select(dataset=n,group='scaling',method=m)[0]
        lines.append(f"{n} & {m} & {r['iterations']:,} & {r['seconds']:.4f} & {r['counts']['hvp']} & ${r['bound']:.2e}$ & {'Yes' if r['status']=='certified' else 'No'}".replace('e+','\\mathrm{e}{+}').replace('e-','\\mathrm{e}{-}')+r' \\')
write('raw_table.tex',lines)
summary=dict(runs=len(RUNS),certified=sum(r['status']=='certified' for r in RUNS),
             main_certified=sum(r['status']=='certified' for r in select(group='main')),
             main_runs=len(select(group='main')),
             reference_bound_max=max(r['ref_bound'] for r in RUNS),
             raw_restarts={r['dataset']:r['restarts'] for r in select(group='scaling',method='Restart-AGD')})
(OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
