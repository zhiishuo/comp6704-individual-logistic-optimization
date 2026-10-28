"""Reproduce all experiments: python experiment.py [--quick].

Educational implementations of six deterministic full-batch optimizers.
Only scipy's independent reference solve delegates optimization to a library.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parent
METHODS = ['GD', 'AGD', 'Restart-AGD', 'L-BFGS', 'Newton-CG', 'Newton-PCG']
TOL = 1e-6  # certified absolute training-objective error upper bound


def raw_data(name):
    if name == 'Cancer':
        ds = load_breast_cancer()
        return ds.data, ds.target, None
    if name == 'Digits':
        ds = load_digits()
        return ds.data, ds.target % 2, None  # odd digit is positive
    with zipfile.ZipFile(ROOT / 'bank_marketing.zip') as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('bank.zip'))) as inner:
            rows = list(csv.DictReader(io.StringIO(inner.read('bank-full.csv').decode()), delimiter=';'))
    # duration is only known AFTER the contact. Exclude it before splitting.
    numeric = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
    categorical = [k for k in rows[0] if k not in numeric + ['duration', 'y']]
    names = numeric + categorical
    X = np.array([[r[k] for k in names] for r in rows], dtype=object)
    y = np.array([r['y'] == 'yes' for r in rows], dtype=int)
    return X, y, len(numeric)


def dataset(name, seed, scaled=True):
    X, y, nnum = raw_data(name)
    tr, te = train_test_split(np.arange(len(y)), test_size=.2, random_state=seed, stratify=y)
    if nnum is None:
        scaler = StandardScaler() if scaled else None
        a = scaler.fit_transform(X[tr]) if scaled else X[tr].copy()
        b = scaler.transform(X[te]) if scaled else X[te].copy()
        a = sparse.csr_matrix(a); b = sparse.csr_matrix(b)
    else:
        aa, bb = X[tr, :nnum].astype(float), X[te, :nnum].astype(float)
        if scaled:
            scaler = StandardScaler()
            aa, bb = scaler.fit_transform(aa), scaler.transform(bb)
        enc = OneHotEncoder(handle_unknown='ignore', dtype=np.float64)
        ca = enc.fit_transform(X[tr, nnum:]); cb = enc.transform(X[te, nnum:])
        a = sparse.hstack([aa, ca], format='csr'); b = sparse.hstack([bb, cb], format='csr')
    # Penalizing the intercept ensures the stated global strong-convexity bound.
    a = sparse.hstack([a, np.ones((len(tr), 1))], format='csr')
    b = sparse.hstack([b, np.ones((len(te), 1))], format='csr')
    return a, y[tr].astype(float), b, y[te], tr, te


class Logistic:
    def __init__(self, X, y, lam):
        self.X, self.y, self.lam = X, y, lam
        self.n, self.d = X.shape
        self.counts = dict(fg=0, f=0, hvp=0, diag=0)

    def fg(self, w):
        self.counts['fg'] += 1
        z = self.X @ w
        f = np.mean(np.logaddexp(0., z) - self.y * z) + self.lam * (w @ w) / 2
        g = np.asarray(self.X.T @ (expit(z) - self.y)).ravel() / self.n + self.lam * w
        return float(f), g

    def value(self, w):
        self.counts['f'] += 1
        z = self.X @ w
        return float(np.mean(np.logaddexp(0., z) - self.y * z) + self.lam * (w @ w) / 2)

    def curvature(self, w):
        p = expit(self.X @ w)
        # One additional forward data pass to construct D, counted in f.
        self.counts['f'] += 1
        return p * (1 - p)

    def hvp(self, v, D):
        self.counts['hvp'] += 1
        return np.asarray(self.X.T @ (D * (self.X @ v))).ravel() / self.n + self.lam * v

    def diagonal(self, D):
        self.counts['diag'] += 1
        return np.asarray(self.X.power(2).T @ D).ravel() / self.n + self.lam

    def passes(self):
        c = self.counts
        return 2*c['fg'] + c['f'] + 2*c['hvp'] + c['diag']


def pcg(obj, D, rhs, diagonal, rtol):
    """Zero-start conjugate gradients; SPD guaranteed by lambda > 0."""
    x = np.zeros_like(rhs); r = rhs.copy(); z = r / diagonal; p = z.copy()
    rz = r @ z; target = rtol * np.linalg.norm(rhs)
    for j in range(min(200, 2*len(rhs))):
        q = obj.hvp(p, D); denom = p @ q
        if denom <= 0 or not np.isfinite(denom):
            break
        alpha = rz / denom; x += alpha*p; r -= alpha*q
        if np.linalg.norm(r) <= target:
            break
        z = r / diagonal; newrz = r @ z
        p = z + (newrz/rz)*p; rz = newrz
    return x


def lbfgs_direction(g, pairs):
    q = g.copy(); alphas = []
    for s, y in reversed(pairs):
        a = (s @ q) / (s @ y); alphas.append(a); q -= a*y
    gamma = (pairs[-1][0] @ pairs[-1][1]) / (pairs[-1][1] @ pairs[-1][1]) if pairs else 1.
    r = gamma*q
    for (s, y), a in zip(pairs, reversed(alphas)):
        r += s * (a - (y @ r)/(s @ y))
    return -r


def solve(obj, method, L, maxiter=2500, maxtime=30.):
    w = np.zeros(obj.d); v = w.copy(); t = 1.; pairs = []; restarts = 0; backtracks = 0
    history = []; start = time.perf_counter(); f, g = obj.fg(w)
    def record(k):
        history.append(dict(k=k, seconds=time.perf_counter()-start, f=f,
                            grad=float(np.linalg.norm(g)), bound=float(g@g/(2*obj.lam)), passes=obj.passes()))
    record(0); status = 'iteration_limit'
    for k in range(1, maxiter+1):
        if history[-1]['bound'] <= TOL:
            status = 'certified'; break
        if time.perf_counter()-start >= maxtime:
            status = 'time_limit'; break
        oldw, oldg, oldf = w.copy(), g.copy(), f
        if method in ['GD', 'AGD', 'Restart-AGD']:
            if method == 'GD':
                w = w - g/L
            else:
                _, gv = obj.fg(v); candidate = v - gv/L
                if method == 'Restart-AGD' and obj.value(candidate) > f + 1e-14:
                    # Function restart with a GD fallback guarantees descent.
                    restarts += 1; t = 1.; v = w.copy(); candidate = w - g/L
                tn = (1 + np.sqrt(1+4*t*t))/2
                v = candidate + ((t-1)/tn)*(candidate-w); t = tn; w = candidate
            f, g = obj.fg(w)
        else:
            if method == 'L-BFGS':
                direction = lbfgs_direction(g, pairs)
            else:
                D = obj.curvature(w)
                diagonal = obj.diagonal(D) if method == 'Newton-PCG' else np.ones(obj.d)
                direction = pcg(obj, D, -g, diagonal, min(.5, np.sqrt(np.linalg.norm(g))))
            if direction @ g >= 0:
                direction = -g
            alpha = 1.; slope = g @ direction
            for bt in range(60):
                if obj.value(w+alpha*direction) <= f + 1e-4*alpha*slope:
                    break
                alpha *= .5; backtracks += 1
            else:
                status = 'line_search_failed'; break
            w = w + alpha*direction; f, g = obj.fg(w)
            if method == 'L-BFGS':
                s, y = w-oldw, g-oldg
                if s@y > 1e-12*np.linalg.norm(s)*np.linalg.norm(y):
                    pairs.append((s, y)); pairs = pairs[-10:]
        record(k)
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            status = 'nonfinite'; break
    if history[-1]['bound'] <= TOL:
        status = 'certified'
    return w, dict(method=method, status=status, iterations=history[-1]['k'],
                   seconds=history[-1]['seconds'], objective=f, bound=history[-1]['bound'],
                   passes=obj.passes(), counts=obj.counts, restarts=restarts,
                   backtracks=backtracks, history=history)


def check_derivatives():
    rng = np.random.default_rng(90)
    X = sparse.csr_matrix(rng.normal(size=(21, 7))); y = rng.integers(2, size=21)
    o = Logistic(X, y, .01); w = rng.normal(size=7); v = rng.normal(size=7); h = 1e-5
    _, g = o.fg(w)
    fd = np.array([(o.value(w+h*np.eye(7)[i])-o.value(w-h*np.eye(7)[i]))/(2*h) for i in range(7)])
    hv = o.hvp(v, o.curvature(w)); fdv = (o.fg(w+h*v)[1]-o.fg(w-h*v)[1])/(2*h)
    grad_error = float(np.max(np.abs(g-fd))); hvp_error = float(np.max(np.abs(hv-fdv)))
    assert grad_error < 1e-8 and hvp_error < 1e-8
    return dict(gradient_max_error=grad_error, hvp_max_error=hvp_error)


def run(quick=False):
    out = ROOT/('results_quick' if quick else 'results'); out.mkdir(exist_ok=True)
    env = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
               platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
               threads=threadpool_info(), tolerance=TOL, derivative_check=check_derivatives(),
               bank_zip_sha256=hashlib.sha256((ROOT/'bank_marketing.zip').read_bytes()).hexdigest())
    (out/'environment.json').write_text(json.dumps(env, indent=2))
    configs = [(n,s,.001,True,'main') for n in ['Cancer','Digits','Bank'] for s in ([0] if quick else [0,1,2])]
    if not quick:
        configs += [(n,0,l,True,'regularization') for n in ['Cancer','Digits','Bank'] for l in [.0001,.01]]
        configs += [(n,0,.001,False,'scaling') for n in ['Cancer','Bank']]
    results = []; all_info = []
    for name, seed, lam, scaled, group in configs:
        X, y, Xt, yt, tr, te = dataset(name, seed, scaled)
        key = f'{name}_s{seed}_l{lam:g}_{"scaled" if scaled else "raw"}'
        tic = time.perf_counter()
        eig = np.linalg.eigvalsh((X.T@X).toarray()/len(y))
        L = .25*float(eig[-1])+lam
        setup_time = time.perf_counter()-tic
        ref = Logistic(X,y,lam)
        oracle = minimize(ref.fg, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B',
                          options=dict(gtol=1e-12, ftol=1e-15, maxiter=10000, maxls=60))
        rf, rg = ref.fg(oracle.x); refbound = float(rg@rg/(2*lam))
        if refbound >= 1e-8:
            # Independent dense-Hessian trust-region polish for difficult raw units.
            def hessian(w):
                p = expit(X@w); D = p*(1-p)
                return (X.T@X.multiply(D[:,None])).toarray()/len(y) + lam*np.eye(X.shape[1])
            polished = minimize(ref.fg, oracle.x, jac=True, hess=hessian, method='trust-exact',
                                 options=dict(gtol=1e-10,maxiter=200))
            pf, pg = ref.fg(polished.x)
            if np.linalg.norm(pg) < np.linalg.norm(rg):
                oracle = polished; rf, rg = pf, pg; refbound = float(rg@rg/(2*lam))
        assert refbound < 1e-8, (key, refbound)
        info = dict(key=key, dataset=name, seed=seed, lam=lam, scaled=scaled, group=group,
                    ntrain=len(y), ntest=len(yt), d=X.shape[1], nnz=X.nnz,
                    train_positive=float(np.mean(y)), test_positive=float(np.mean(yt)),
                    L=L, condition_upper=L/lam, spectral_setup_seconds=setup_time,
                    ref_objective=rf, ref_bound=refbound,
                    train_ids_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),
                    test_ids_sha256=hashlib.sha256(te.tobytes()).hexdigest())
        all_info.append(info)
        np.savez_compressed(out/f'{key}_split.npz', train_indices=tr, test_indices=te, reference_w=oracle.x)
        # Rotate deterministic execution order across seeds to reduce fixed order bias.
        order = np.roll(np.array(METHODS), seed)
        for method in order:
            cached = out/f'{key}_{method}.json'
            if cached.exists():
                results.append(json.loads(cached.read_text()))
                print(key, method, 'saved run retained',flush=True)
                continue
            o = Logistic(X,y,lam)
            w, res = solve(o, str(method), L, maxiter=2500, maxtime=30.)
            probs = expit(Xt@w)
            res.update(info)
            res.update(test_accuracy=float(accuracy_score(yt, probs>=.5)),
                       test_balanced_accuracy=float(balanced_accuracy_score(yt,probs>=.5)),
                       test_auc=float(roc_auc_score(yt,probs)), test_logloss=float(log_loss(yt,probs)),
                       observed_gap=float(res['objective']-rf))
            assert res['observed_gap'] >= -refbound-1e-10
            assert res['observed_gap'] <= res['bound']+1e-10
            np.savez_compressed(out/f'{key}_{method}_solution.npz',w=w)
            (out/f'{key}_{method}.json').write_text(json.dumps(res,indent=2))
            print(key, method, res['status'],res['iterations'],f"{res['seconds']:.3f}s",f"bound={res['bound']:.2e}",flush=True)
            results.append(res)
        (out/'instances.json').write_text(json.dumps(all_info,indent=2))
    flat = [{k:v for k,v in r.items() if k not in ['history','counts']} for r in results]
    with (out/'summary.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (out/'all_runs.json').write_text(json.dumps(results,indent=2))
    print('Completed',len(results),'runs',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--quick',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=1):
        run(args.quick)
