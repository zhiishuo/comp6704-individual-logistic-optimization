"""Independently recompute saved objectives, certificates and test metrics."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits
from experiment import ROOT, TOL, dataset, check_derivatives

def verify():
    rs=json.loads((ROOT/'results/all_runs.json').read_text())
    assert len(rs)==102 and len({(r['key'],r['method']) for r in rs})==102
    errors=[]; groups={}
    for r in rs: groups.setdefault(r['key'],[]).append(r)
    for key,rr in groups.items():
        r=rr[0];X,y,Xt,yt,tr,te=dataset(r['dataset'],r['seed'],r['scaled'])
        split=np.load(ROOT/f'results/{key}_split.npz')
        assert np.array_equal(tr,split['train_indices']) and np.array_equal(te,split['test_indices'])
        assert not np.intersect1d(tr,te).size
        lam=r['lam']
        # Recompute directly rather than calling Logistic.fg.
        def recompute(w):
            z=X@w
            f=np.mean(np.logaddexp(0,z)-y*z)+lam*np.dot(w,w)/2
            g=np.asarray(X.T@(expit(z)-y)).ravel()/len(y)+lam*w
            return f,np.dot(g,g)/(2*lam)
        rf,rb=recompute(split['reference_w']);assert rb<1e-8
        for r in rr:
            w=np.load(ROOT/f"results/{key}_{r['method']}_solution.npz")['w']
            f,b=recompute(w);p=expit(Xt@w)
            assert abs(f-r['objective'])<1e-12
            assert abs(b-r['bound'])<1e-10*max(1,b)
            assert f-rf>=-rb-1e-10 and f-rf<=b+1e-10
            assert (r['status']=='certified') == (b<=TOL)
            assert abs(log_loss(yt,p)-r['test_logloss'])<1e-12
            assert abs(roc_auc_score(yt,p)-r['test_auc'])<1e-12
            assert abs(np.mean((p>=.5)==yt)-r['test_accuracy'])<1e-12
            assert abs(balanced_accuracy_score(yt,p>=.5)-r['test_balanced_accuracy'])<1e-12
            c=r['counts'];assert r['passes']==2*c['fg']+c['f']+2*c['hvp']+c['diag']
            assert len(r['history'])==r['iterations']+1
            if r['method'] in ['GD','Restart-AGD','L-BFGS','Newton-CG','Newton-PCG']:
                assert max(np.diff([h['f'] for h in r['history']]),default=0)<1e-11
            errors.append(abs(f-r['objective']))
    receipt=dict(runs_verified=len(rs),instances=len(groups),certified=sum(r['status']=='certified' for r in rs),
                 max_objective_error=max(errors),derivative_checks=check_derivatives(),
                 checks=['disjoint split indices','saved solution objective','gradient certificate',
                         'independent reference bound','test metrics','work counts','monotone accepted objectives'])
    (ROOT/'results/verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):verify()
