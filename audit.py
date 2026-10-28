"""Additional derivative, curvature, CG, BFGS and receipt consistency checks."""
import csv
import json
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from experiment import ROOT, Logistic, dataset, pcg, lbfgs_direction


def audit():
    rng=np.random.default_rng(20261028)
    checks=[]
    for name,scaled in [('Cancer',True),('Digits',True),('Bank',True),('Cancer',False),('Bank',False)]:
        X,y,_,_,_,_=dataset(name,0,scaled)
        o=Logistic(X,y,.001)
        v=rng.normal(size=X.shape[1]);v/=max(1.,np.max(np.abs(X@v)))
        w=.2*v;h=1e-4
        f,g=o.fg(w)
        fd=(o.value(w+h*v)-o.value(w-h*v))/(2*h)
        derr=abs(fd-g@v)
        D=o.curvature(w)
        H=(X.T@X.multiply(D[:,None])).toarray()/len(y)+.001*np.eye(X.shape[1])
        hv=o.hvp(v,D);fdhv=(o.fg(w+h*v)[1]-o.fg(w-h*v)[1])/(2*h)
        hrel=np.linalg.norm(hv-fdhv)/max(1.,np.linalg.norm(hv))
        explicitrel=np.linalg.norm(H@v-hv)/max(1.,np.linalg.norm(hv))
        eig=np.linalg.eigvalsh(H)
        L=.001+.25*np.linalg.eigvalsh((X.T@X).toarray()/len(y))[-1]
        assert derr<1e-8 and hrel<1e-7 and explicitrel<1e-10
        assert eig[0]>=.001-1e-8 and eig[-1]<=L*(1+1e-10)
        checks.append(dict(dataset=name,scaled=scaled,directional_gradient_error=float(derr),
                           directional_hvp_relative_error=float(hrel),
                           explicit_hessian_relative_error=float(explicitrel),min_eigenvalue=float(eig[0])))
    # Test the two-loop recursion against explicit inverse-BFGS updates.
    d=9;A=rng.normal(size=(d,d));A=A.T@A+np.eye(d);pairs=[]
    for _ in range(6):
        s=rng.normal(size=d);pairs.append((s,A@s))
    s,q=pairs[-1];B=(s@q)/(q@q)*np.eye(d)
    for s,q in pairs:
        rho=1/(s@q);V=np.eye(d)-rho*np.outer(s,q)
        B=V@B@V.T+rho*np.outer(s,s)
    g=rng.normal(size=d)
    bfgs_error=float(np.max(np.abs(lbfgs_direction(g,pairs)+B@g)))
    assert bfgs_error<1e-12
    # Compare matrix-free (P)CG with a dense direct solution on an SPD fixture.
    X=sparse.csr_matrix(rng.normal(size=(80,d)));y=rng.integers(2,size=80)
    o=Logistic(X,y,.1);D=o.curvature(np.zeros(d));rhs=rng.normal(size=d)
    H=(X.T@X.multiply(D[:,None])).toarray()/80+.1*np.eye(d)
    errors=[]
    for diag in [np.ones(d),o.diagonal(D)]:
        z=pcg(o,D,rhs,diag,1e-11)
        errors.append(float(np.linalg.norm(H@z-rhs)/np.linalg.norm(rhs)))
    assert max(errors)<1e-10
    runs=json.loads((ROOT/'results/all_runs.json').read_text())
    with (ROOT/'results/summary.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==len(runs)==102
    bykey={(r['key'],r['method']):r for r in runs}
    for row in rows:
        r=bykey[(row['key'],row['method'])]
        assert json.loads((ROOT/f"results/{r['key']}_{r['method']}.json").read_text())==r
        for k in ['objective','bound','seconds','iterations','passes','test_accuracy','test_balanced_accuracy','test_auc','test_logloss']:
            assert float(row[k])==float(r[k])
    result=dict(real_data_derivatives=checks,lbfgs_dense_recursion_max_error=bfgs_error,
                cg_relative_residuals=errors,consistent_individual_json_csv_aggregate_runs=len(runs))
    (ROOT/'results/additional_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):audit()
