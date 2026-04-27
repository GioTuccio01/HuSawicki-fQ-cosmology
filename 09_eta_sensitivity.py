"""
09_eta_sensitivity.py — Sensitivity su eta.

Ripete l'analisi principale (Pantheon+, 5 param) per diversi valori fissi
di eta, per verificare quanto il risultato su alpha dipende dalla scelta eta=√6.

Valori testati: eta ∈ {√3, 2, √6, 3, 2√3}
cioè x_0 ∈ {2, 1.5, 1, 0.667, 0.5}
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import minimize
from time import time
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import (load_CC, load_BAO_DESI, load_CMB_Planck,
                          load_PantheonPlus, load_fsigma8)

print("SENSITIVITY ANALYSIS SU eta")
print("="*60)

CC = load_CC()
BAO_BLOCKS = load_BAO_DESI()
CMB_MEAN, CMB_ICOV = load_CMB_Planck()
SN_Z, SN_MB, C_SN_inv, Ainv_MB, Binv_MB = load_PantheonPlus()
FS8 = load_fsigma8(); FS8_Z, FS8_V, FS8_S = FS8[:,0], FS8[:,1], FS8[:,2]

C_KMS=299792.458; Z_STAR=1089.80; GAMMA=0.55
X_SAMP = np.unique(np.concatenate([
    np.linspace(1e-5, 0.3, 100), np.linspace(0.3, 50, 300), np.linspace(50, 15000, 80)]))
Z_LOW = np.linspace(1e-5, 15.0, 300); Z_HIGH = np.linspace(15.0, Z_STAR, 400)

def build_chi2(eta_fix):
    X0 = 6.0/eta_fix**2
    def Lam_of(a, Om0):
        return 0.5*((1-Om0)*X0 + a*(1+6*X0-3*X0*X0)/(1+X0)**3)
    def solve_E(z_arr, alpha, Om0):
        Lam = Lam_of(alpha, Om0)
        R_s = 2*Lam - alpha*(1+6*X_SAMP-3*X_SAMP**2)/(1+X_SAMP)**3
        g_s = X_SAMP - R_s
        if not np.all(np.diff(g_s)>0): return None, None, None
        tgt = Om0*X0*(1+np.asarray(z_arr,float))**3
        xs = np.interp(tgt, g_s, X_SAMP)
        return np.sqrt(xs/X0), xs, Lam
    def rd_Aub(Om,H0,ob):
        h=H0/100; return 55.154/((Om*h*h)**0.25351 * ob**0.12807)
    def rs_ratio(Om,H0,ob):
        h=H0/100; return 1.0 - 0.0206*ob**0.165 * (Om*h*h)**0.05
    def chi2_fQ(theta):
        a, Om, H0, ob, s8 = theta
        rd = rd_Aub(Om,H0,ob); rstar = rd*rs_ratio(Om,H0,ob)
        E_lo, xs, Lam = solve_E(Z_LOW, a, Om)
        if E_lo is None: return np.inf
        cum = np.concatenate(([0.0], cumulative_trapezoid(1/E_lo, Z_LOW)))
        Ecc = np.interp(CC[:,0], Z_LOW, E_lo)
        c2cc = np.sum(((CC[:,1]-H0*Ecc)/CC[:,2])**2)
        fac=C_KMS/H0/rd; c2bao=0.
        for bl in BAO_BLOCKS:
            z=bl['z']; Eh=np.interp(z,Z_LOW,E_lo)
            DM=fac*np.interp(z,Z_LOW,cum); DH=fac/Eh
            DV=(z*DM*DM*DH)**(1/3)
            mods=np.array([{'DM':DM,'DH':DH,'DV':DV}[k] for k in bl['kinds']])
            dv=bl['vals']-mods; c2bao+=dv@bl['icov']@dv
        E_hi = np.sqrt(Om*(1+Z_HIGH)**3 + 2*Lam/X0)
        Istar = trapezoid(1/E_lo, Z_LOW) + trapezoid(1/E_hi, Z_HIGH)
        R_th = np.sqrt(Om)*Istar; lA_th = np.pi*(C_KMS/H0)*Istar/rstar
        dv = np.array([R_th, lA_th, ob]) - CMB_MEAN
        c2cmb = dv @ CMB_ICOV @ dv
        DMsn = (C_KMS/H0)*np.interp(SN_Z, Z_LOW, cum)
        mu_th = 5*np.log10((1+SN_Z)*DMsn)+25
        r = SN_MB - mu_th
        c2sn = r @ C_SN_inv @ r - (r @ Binv_MB)**2/Ainv_MB
        Om_z = Om*(1+Z_LOW)**3/E_lo**2
        fQ = +1 + a*(3-xs)/(1+xs)**3; Ge = +1/fQ
        f_z = (Om_z*Ge)**GAMMA
        cf = np.concatenate(([0.0], cumulative_trapezoid(f_z/(1+Z_LOW), Z_LOW)))
        fs8_full = s8 * f_z * np.exp(-cf)
        fs8_mod = np.interp(FS8_Z, Z_LOW, fs8_full)
        c2fs8 = np.sum(((FS8_V-fs8_mod)/FS8_S)**2)
        return c2cc + c2bao + c2cmb + c2sn + c2fs8
    return chi2_fQ

def in_prior(th):
    a, Om, H0, ob, s8 = th
    return (-0.80<a<0.33 and 0.10<Om<0.50 and 55<H0<85 and
            0.018<ob<0.026 and 0.5<s8<1.1)

def mh_chain(log_p, start, init_scale, n_burn, n_prod, rng):
    ndim = len(start)
    chol = np.linalg.cholesky(np.diag(init_scale**2))
    scale = 1.0; cur = start.copy(); cur_lp = log_p(cur)
    burn = np.zeros((n_burn, ndim)); acc=0; win=200
    for i in range(n_burn):
        prop = cur + scale*(chol @ rng.standard_normal(ndim))
        plp = log_p(prop)
        if plp - cur_lp > np.log(rng.random()+1e-300):
            cur, cur_lp = prop, plp; acc += 1
        burn[i] = cur
        if (i+1)%win==0 and i<n_burn-win:
            r = acc/win
            if r<0.15: scale*=0.55
            elif r<0.22: scale*=0.85
            elif r>0.55: scale*=1.45
            elif r>0.40: scale*=1.15
            acc = 0
    cov_emp = np.cov(burn[n_burn//2:].T) + 1e-12*np.eye(ndim)
    chol = np.linalg.cholesky(cov_emp)
    prod = np.zeros((n_prod, ndim)); acc=0
    for i in range(n_prod):
        prop = cur + chol @ rng.standard_normal(ndim)
        plp = log_p(prop)
        if plp - cur_lp > np.log(rng.random()+1e-300):
            cur, cur_lp = prop, plp; acc += 1
        prod[i] = cur
    return prod, acc/n_prod

def gelman_rubin(chains):
    M, N, D = chains.shape
    means = chains.mean(axis=1)
    W = chains.var(axis=1, ddof=1).mean(axis=0)
    B = N * means.var(axis=0, ddof=1)
    V = (1-1/N)*W + (1/N)*B
    return np.sqrt(V/W)

# Valori di eta da testare
eta_vals = [1.1, 1.5, 2.0, np.sqrt(6), 3.0]
eta_labels = [r"√3", "2", r"√6", "3", "2√3"]

results = []
for eta_fix, lbl in zip(eta_vals, eta_labels):
    print(f"\n--- eta = {lbl} = {eta_fix:.3f}  (x_0 = {6/eta_fix**2:.3f}) ---")
    chi2_fn = build_chi2(eta_fix)
    def log_p(th):
        if not in_prior(th): return -np.inf
        c = chi2_fn(th)
        return -0.5*c if np.isfinite(c) else -np.inf
    # MAP
    res = minimize(lambda x: -log_p(x), [0.0, 0.3, 68, 0.02236, 0.80],
                   method='Nelder-Mead', options={'xatol':1e-6,'fatol':1e-4,
                   'maxiter':10000,'adaptive':True})
    map_th = res.x; chi2_map = chi2_fn(map_th)
    # MCMC ridotto (2 catene, 1500 burn + 5000 prod)
    init_scale = np.array([0.05, 0.006, 0.6, 0.00015, 0.02])
    chains = []
    for ch in range(2):
        rng = np.random.default_rng(5000+ch)
        st = map_th + 0.3*init_scale*rng.standard_normal(5)
        if not in_prior(st): st = map_th.copy()
        s, acc = mh_chain(log_p, st, init_scale, 1500, 5000, rng)
        chains.append(s)
    chains = np.array(chains)
    flat = chains.reshape(-1, 5)[::2]
    Rhat = gelman_rubin(chains)
    q = np.quantile(flat, [0.16,0.5,0.84], axis=0)
    names = ['alpha','Om0','H0','ombh2','sigma8']
    print(f"MAP: a={map_th[0]:+.4f} Om={map_th[1]:.4f} H0={map_th[2]:.2f} "
          f"s8={map_th[4]:.4f}  chi2={chi2_map:.2f}")
    print(f"Marg: a={q[1,0]:+.4f}+{q[2,0]-q[1,0]:.4f}/-{q[1,0]-q[0,0]:.4f}  "
          f"Rhat(a)={Rhat[0]:.3f}")
    sig = abs(q[1,0])/((q[2,0]-q[0,0])/2)
    results.append({
        'eta_label': lbl, 'eta': eta_fix, 'x0': 6/eta_fix**2,
        'map': map_th, 'chi2': chi2_map,
        'alpha_med': q[1,0], 'alpha_lo': q[0,0], 'alpha_hi': q[2,0],
        'sigma_alpha': sig, 'Rhat': Rhat[0],
    })

# Tabella finale
print("\n\n" + "="*75)
print("SENSITIVITY TABLE: risultato in funzione di eta fissato")
print("="*75)
print(f"{'eta':>8} {'x_0':>7} {'chi2_min':>10} {'alpha_MAP':>11} "
      f"{'alpha_50%':>11} {'sigma':>7} {'Rhat':>6}")
for r in results:
    print(f"{r['eta_label']:>8} {r['x0']:>7.3f} {r['chi2']:>10.2f} "
          f"{r['map'][0]:>+11.4f} {r['alpha_med']:>+11.4f} "
          f"{r['sigma_alpha']:>6.2f}σ {r['Rhat']:>6.3f}")
print()
print("Interpretazione: se la significatività di alpha!=0 resta ~2σ attraverso")
print("la scelta di eta, il risultato principale è robusto. Se varia da <1σ a >3σ,")
print("c'è forte dipendenza dal prior implicito su eta = scelta canonica giustificata.")
