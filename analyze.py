"""Regenerate figures and LaTeX tables from frozen experiment receipts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
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
    axes[0,col].set_title(n);axes[0,col].set_xlabel('Data passes')
    axes[1,col].set_xlabel('Solve time (s)')
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

# Model architecture: only training features enter the optimization objective.
# The six optimizers update the SAME parameter vector and objective.
fig,ax=plt.subplots(figsize=(7,1.3))
ax.set_xlim(0,7);ax.set_ylim(0,1.3);ax.axis('off')
centers=[.85,2.55,4.25,6.0]
labels=['Train-fitted\nfeatures + bias',r'Linear score'+'\n'+r'$z=x^\top w$',
        'Sigmoid\n'+r'$p=\sigma(z)$','Training objective\n'+r'$f_\lambda(w)$']
for x,label in zip(centers,labels):
    ax.add_patch(FancyBboxPatch((x-.71,.66),1.42,.53,boxstyle='round,pad=.025',
                               facecolor='#eef5f7',edgecolor='#31576b',lw=1))
    ax.text(x,.925,label,ha='center',va='center',fontsize=9)
for left,right in zip(centers[:-1],centers[1:]):
    ax.annotate('',(right-.75,.925),(left+.75,.925),arrowprops=dict(arrowstyle='->',lw=1,color='#31576b'))
ax.plot([6,6,2.55],[.62,.34,.34],color='#009e73',lw=1)
ax.annotate('',(2.55,.63),(2.55,.34),arrowprops=dict(arrowstyle='->',lw=1,color='#009e73'))
ax.text(4.28,.12,'Six solvers update w; same model, split, lambda and certificate',
        ha='center',va='center',fontsize=8,color='#075e49')
fig.tight_layout(pad=.2);save(fig,'model_pipeline')

# Show times ONLY for certified solves. Capped times must not look competitive.
fig,axes=plt.subplots(1,3,figsize=(7,2.1),sharey=True)
for ax,n in zip(axes,NAMES):
    for j,(m,color) in enumerate(zip(METHODS,COLORS)):
        rr=select(dataset=n,group='main',method=m)
        good=[r['seconds'] for r in rr if r['status']=='certified']
        if good:
            med=float(np.median(good));lo=min(good);hi=max(good)
            ax.barh(j,med,height=.6,color=color,alpha=.75)
            ax.errorbar(med,j,xerr=[[med-lo],[hi-med]],fmt='none',ecolor='black',capsize=2,lw=.8)
            ax.text(max(hi*1.18,med*1.4,.001),j,f'{len(good)}/3',va='center',fontsize=7)
        else:
            ax.text(.001,j,'0/3: capped',va='center',fontsize=7,color='#a44800')
    ax.set_title(n);ax.set_xscale('log');ax.set_xlim(.0007,4);ax.set_xlabel('Certified solve time (s)')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
axes[0].set_yticks(range(len(METHODS)),METHODS,fontsize=8);axes[0].invert_yaxis()
fig.tight_layout(pad=.5);save(fig,'certified_timing')

# All 17 configurations x 6 methods. Same rows as the complete run archive.
instances=[]
for r in RUNS:
    if r['key'] not in [x['key'] for x in instances]:instances.append(r)
matrix=np.array([[int(select(key=r['key'],method=m)[0]['status']=='certified') for m in METHODS] for r in instances])
rowlabels=[f"{r['dataset']}{' raw' if not r['scaled'] else ''} s{r['seed']} / {r['lam']:g}" for r in instances]
from matplotlib.colors import ListedColormap
fig,ax=plt.subplots(figsize=(7,3.35))
ax.imshow(matrix,cmap=ListedColormap(['#f8ddbf','#9bcdbb']),vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(6),['GD','AGD','Restart-AGD','L-BFGS','Newton-CG','Newton-PCG'],fontsize=8)
ax.set_yticks(range(len(instances)),rowlabels,fontsize=7)
ax.tick_params(length=0);ax.xaxis.tick_top()
for i in range(len(instances)):
    for j in range(6):ax.text(j,i,'OK' if matrix[i,j] else 'CAP',ha='center',va='center',fontsize=6.5)
ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,len(instances),1),minor=True)
ax.grid(which='minor',color='white',linewidth=.8);ax.tick_params(which='minor',length=0)
for sp in ax.spines.values():sp.set_visible(False)
fig.tight_layout(pad=.3);save(fig,'status_matrix')

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
