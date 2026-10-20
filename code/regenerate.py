from pathlib import Path
import json, shutil, hashlib, collections, statistics, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
def read(n): return json.loads((OUT/'evidence'/n).read_text(encoding='utf-8'))
names=['exp2_2_2_phase_diagram.json']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def finish(fig,name):
 fig.tight_layout();fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight');fig.savefig(OUT/'figures'/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)
def ms(vals,scale=1,digits=4):
 a=np.asarray(vals)*scale
 return f'${a.mean():.{digits}f}\\pm{a.std(ddof=1):.{digits}f}$' if len(a)>1 else f'${a.mean():.{digits}f}$'
def table(name,cols,header,rows):
 text='\\begin{tabular}{'+cols+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(r)+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'
 (OUT/'tables'/f'{name}.tex').write_text(text)

phase=read(names[0]);groups=collections.defaultdict(list)
for r in phase['results']:groups[(float(r['lambda_rep']),float(r['d0']),float(r['eta']))].append(r)
stats=[]
for (lr,d0,eta),rs in groups.items():
 rho=eta*d0*d0/(2*lr);pred=max(0,d0*d0-2*lr/eta);ys=[r['D_rep'] for r in rs]
 stats.append(dict(d0=d0,rho=rho,pred=pred,measured=np.mean(ys),sd=np.std(ys,ddof=1)))
supercrit=[r for r in stats if r['rho']>1];x=np.array([r['pred'] for r in supercrit]);y=np.array([r['measured'] for r in supercrit]);coef=np.polyfit(x,y,1)
summary={'phase':{'n_supercritical':len(x),'pearson_r':float(np.corrcoef(x,y)[0,1]),'regression_R2':float(np.corrcoef(x,y)[0,1]**2),'identity_R2':float(1-np.sum((y-x)**2)/np.sum((y-y.mean())**2)),'MAE':float(np.mean(abs(y-x))),'RMSE':float(np.sqrt(np.mean((y-x)**2))),'slope':float(coef[0]),'intercept':float(coef[1])}}
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
for d0 in sorted(set(r['d0'] for r in stats)):
 rs=[r for r in stats if r['d0']==d0];ax[0].errorbar([r['pred'] for r in rs],[r['measured'] for r in rs],yerr=[r['sd'] for r in rs],fmt='o',ms=3,capsize=1,label=f'$d_0={d0:g}$');ax[1].scatter([r['rho'] for r in rs],[r['measured']-r['pred'] for r in rs],s=13)
ax[0].plot([0,2.3],[0,2.3],'k--',lw=1);ax[0].set(xlabel='Reduced prediction',ylabel='Measured squared distance');ax[0].legend(fontsize=7,ncol=2);ax[1].axhline(0,c='k',ls='--',lw=1);ax[1].axvline(1,c='gray',ls=':');ax[1].set(xscale='log',xlabel=r'$\rho$',ylabel='Measured minus predicted');finish(fig,'phase_validation')

fig,ax=plt.subplots(figsize=(6.8,3.2));xx=np.linspace(.25,4,400);ax.plot(xx,np.maximum(0,1-1/xx),'k--',label='Reduced prediction')
for n,label in [('exp3_transition.json','Synthetic'),('exp3_transition_cifar10.json','CIFAR-10')]:
 d=read(n);g=d['summary_by_eta'];a=[g[str(e)] for e in d['etas']];ax.errorbar([r['rho'] for r in a],[r['D_rep_mean'] for r in a],yerr=[r['D_rep_std'] for r in a],fmt='o-',ms=4,capsize=3,label=label)
ax.axvline(1,c='gray',ls=':');ax.set(xlabel=r'$\rho$',ylabel=r'$D_{\mathrm{rep}}$');ax.legend();finish(fig,'transition')

method_names={'independent':'Independent','DML':'DML','rep_without_diversity':'Cooperation only','diversity_without_rep':'Target only','SML':'SML','DiffDML':'Diff-DML-inspired'}
for dataset,name in [('cifar10','exp3_baseline_comparison_cifar10.json'),('cifar100','exp3_cifar100_fixed.json')]:
 d=read(name);rows=[]
 for key,label in method_names.items():
  rs=d['results'][key]['seeds'];rows.append([label,ms([r['ensemble_acc'] for r in rs],100,2),ms([r['best_single_acc'] for r in rs],100,2),ms([r['D_rep'] for r in rs]),ms([r['prediction_disagreement_rate'] for r in rs],100,2)])
 table(dataset,'lrrrr','Method & Ensemble (\\%) & Best peer (\\%) & $D_{\\rm rep}$ & Disagreement (\\%)',rows)

audit=read('audit_results.json');ag=collections.defaultdict(list)
for r in audit['evaluations']:
 if r['status']=='ok':ag[r['file'].split('__seed_')[0]].append(r)
summary['audit']={};rows=[]
for method in ['independent','dml','sml']:
 rs=ag[f'matched__{method}__d0_1.0000'];metrics={k:[] for k in ['D','cka','cka_raw','pre','post','removed','acc','legacy']}
 for r in rs:
  ps=r['pairs'];metrics['D'].append(r['D_rep']);metrics['acc'].append(r['ensemble_acc']);metrics['legacy'].append(r['legacy_equal_batch_D_rep'])
  for k,key in [('cka','linear_cka_normalized'),('cka_raw','linear_cka_raw')]:metrics[k].append(np.mean([p[key] for p in ps]))
  for k,key in [('pre','before_mse'),('post','after_mse'),('removed','fraction_removed')]:metrics[k].append(np.mean([p['procrustes_normalized'][key] for p in ps]))
 summary['audit'][method]={k:{'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1))} for k,v in metrics.items()}
 rows.append([method.upper() if method!='independent' else 'Independent',ms(metrics['acc'],100,3),ms(metrics['D']),ms(metrics['cka']),ms(metrics['post'])])
table('audit','lrrrr','Method & Accuracy (\\%) & $D_{\\rm rep}$ & Normalized CKA & Aligned distance',rows)
fig,ax=plt.subplots(1,2,figsize=(8,3.1));labels=['Independent','DML','SML'];pos=np.arange(3)
for offset,key,lab in [(-.18,'pre','Before alignment'),(.18,'post','After alignment')]:
 ax[0].bar(pos+offset,[summary['audit'][m][key]['mean'] for m in ['independent','dml','sml']],width=.36,label=lab)
ax[0].set_yscale('log');ax[0].set_xticks(pos,labels);ax[0].set_ylabel('Held-out squared distance (log scale)');ax[0].legend(fontsize=8)
ax[1].errorbar(pos,[summary['audit'][m]['cka']['mean'] for m in ['independent','dml','sml']],yerr=[summary['audit'][m]['cka']['sd'] for m in ['independent','dml','sml']],fmt='o',capsize=4);ax[1].set_xticks(pos,labels);ax[1].set(ylabel='Normalized linear CKA',ylim=(.92,1));finish(fig,'alignment')

rows=[]
for d0 in [1.2,1.7607,1.9367]:
 rs=ag[f'geometry__sml__d0_{d0:.4f}'];rows.append([str(d0),f'{len(rs)}/3',f'{3-len(rs)}/3',ms([r['ensemble_acc'] for r in rs],100,2),ms([r['D_rep'] for r in rs]),ms([np.mean([p['linear_cka_normalized'] for p in r['pairs']]) for r in rs]),ms([np.mean([p['procrustes_normalized']['after_mse'] for p in r['pairs']]) for r in rs])])
table('geometry','rrrrrrr','$d_0$ & Valid & Failed & Accuracy (\\%) & $D_{\\rm rep}$ & CKA & Aligned',rows)

d=read('exp4_operating_point_check.json');rows=[]
for par,vals in d['summary'].items():
 for v in vals:
  rs=[r for r in d['runs'] if r['swept_param']==par and r['swept_value']==v[par]]
  label={'lambda_out':r'$\lambda_{\rm out}$','lambda_rep':r'$\lambda_{\rm rep}$','lambda_div':r'$\lambda_{\rm div}$','d0':r'$d_0$'}[par]
  rows.append([label,str(v[par]),ms([r['ensemble_acc'] for r in rs],100,2),ms([r['D_rep'] for r in rs]),ms([r['prediction_disagreement_rate'] for r in rs],100,2)])
table('ablation','lrrrr','Parameter & Value & Accuracy (\\%) & $D_{\\rm rep}$ & Disagreement (\\%)',rows)

d=read('exp5_k_feasibility_cifar10.json');fig,ax=plt.subplots(figsize=(6.6,3.3));rows=[]
for k in [2,3,4,5]:
 rs=sorted([r for r in d['aggregated'] if r['K']==k],key=lambda r:r['d0']);color=next(ax._get_lines._cycler_items)['color'] if False else None
 line=ax.errorbar([r['d_star_squared_unconstrained'] for r in rs],[r['D_rep_mean'] for r in rs],yerr=[r['D_rep_std'] for r in rs],fmt='o-',ms=3,capsize=2,label=f'$K={k}$');xx=np.linspace(0,5,100);ax.plot(xx,np.minimum(xx,2*k/(k-1)),ls='--',lw=.7,alpha=.5,color=line.lines[0].get_color())
 for r in rs[-3:]:rows.append([str(k),f"{r['d0']:.4f}",f"{r['D_pred_squared']:.4f}",f"${r['D_rep_mean']:.4f}\\pm{r['D_rep_std']:.4f}$",f"{r['E_d0_mean']:.4f}",f"${100*r['ensemble_acc_mean']:.2f}\\pm{100*r['ensemble_acc_std']:.2f}$"])
ax.set(xlabel='Unconstrained effective target',ylabel=r'$D_{\mathrm{rep}}$');ax.legend(ncol=2);finish(fig,'finite_k')
table('finite_k','rrrrrr','$K$ & $d_0$ & Prediction & $D_{\\rm rep}$ & $E_{d_0}$ & Accuracy (\\%)',rows)
(OUT/'evidence/recomputed_summary.json').write_text(json.dumps(summary,indent=2))
p=summary['phase']
(OUT/'tables/phase_statistics.tex').write_text('For these 85 configuration means, the fitted slope is %.4f and the intercept is %.4f; Pearson correlation is %.5f, regression $R^2$ is %.5f, and identity-line $R^2$ is %.5f. The mean absolute error is %.4f and root mean squared error is %.4f.\n' % (p['slope'],p['intercept'],p['pearson_r'],p['regression_R2'],p['identity_R2'],p['MAE'],p['RMSE']))
print('Regenerated tables, figures, and summary statistics successfully.')
