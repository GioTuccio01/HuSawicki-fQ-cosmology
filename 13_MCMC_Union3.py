"""
13_MCMC_Union3.py — Analisi principale con Union3 SNe al posto di Pantheon+.

Cross-check del risultato α<0 sostituendo SN con Union3 binned
(Rubin+2023, 22 bin). La covarianza qui include STAT+SYS, quindi
l'analisi è metodologicamente più robusta di DES-Y5 stat-only.

ATTESO (dalla letteratura DESI+Union3): preferenza per dinamicità DE
intermedia, tra Pantheon+ e DES-Y5.

NOTA: Union3 usa il framework Unity (Bayesian hierarchical) che e'
metodologicamente diverso da SALT3+BBC usato da Pantheon+ e DES-Y5.
I distance moduli di output sono comparabili ma la calibrazione
interna differisce.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import minimize
from time import time
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import (load_CC, load_BAO_DESI_official, load_CMB_Planck,
                          load_Union3, load_fsigma8)

print("="*65)
print("ANALISI f(Q) con Union3 (22 bin)")
print("="*65)

CC = load_CC()
BAO_BLOCKS = load_BAO_DESI_official()
CMB_MEAN, CMB_ICOV = load_CMB_Planck()
SN_Z, SN_MU, C_SN_inv, Ainv_MB, Binv_MB = load_Union3()
N_SN = len(SN_Z)
FS8 = load_fsigma8(); FS8_Z, FS8_V, FS8_S = FS8[:,0], FS8[:,1], FS8[:,2]

N_TOT = len(CC) + sum(len(b['kinds']) for b in BAO_BLOCKS) + 3 + N_SN + len(FS8)
print(f"  CC:       {len(CC)} pt")
print(f"  BAO:      12 pt")
print(f"  CMB:      3 pt")
print(f"  Union3:   {N_SN} bin (include STAT+SYS)")
print(f"  fsigma8:  {len(FS8)} pt")
print(f"  TOTALE:   {N_TOT}\n")

C_KMS = 299792.458; OM_R = 9.2e-5; Z_STAR = 1089.80; GAMMA = 0.55
ETA_FIX = np.sqrt(6.0); X0_FIX = 1.0

X_SAMP = np.unique(np.concatenate([
    np.linspace(1e-5, 0.3, 100), np.linspace(0.3, 50, 300),
    np.linspace(50, 15000, 80)]))
Z_LOW = np.linspace(1e-5, 15.0, 300); Z_HIGH = np.linspace(15.0, Z_STAR, 400)

def Lam_of(a, Om0):
    return 0.5*((1.0-Om0)*X0_FIX + a*(1+6*X0_FIX-3*X0_FIX*X0_FIX)/(1+X0_FIX)**3)

def solve_E(z_arr, alpha, Om0):
    Lam = Lam_of(alpha, Om0)
    R_s = 2*Lam - alpha*(1+6*X_SAMP-3*X_SAMP**2)/(1+X_SAMP)**3
    g_s = X_SAMP - R_s
    if not np.all(np.diff(g_s) > 0): return None, None, None
    tgt = Om0*X0_FIX*(1+np.asarray(z_arr, float))**3
    xs = np.interp(tgt, g_s, X_SAMP)
    return np.sqrt(xs/X0_FIX), xs, Lam

def rd_Aub(Om, H0, ob):
    h = H0/100; return 55.154/((Om*h*h)**0.25351 * ob**0.12807)
def rs_ratio(Om, H0, ob):
    h = H0/100; return 1.0 - 0.0206*ob**0.165 * (Om*h*h)**0.05

def chi2_fQ(theta, return_parts=False):
    a, Om, H0, ob, s8 = theta
    rd = rd_Aub(Om, H0, ob); rstar = rd*rs_ratio(Om, H0, ob)
    E_lo, xs, Lam = solve_E(Z_LOW, a, Om)
    if E_lo is None:
        return (np.inf, None) if return_parts else np.inf
    cum = np.concatenate(([0.0], cumulative_trapezoid(1/E_lo, Z_LOW)))
    Ecc = np.interp(CC[:,0], Z_LOW, E_lo)
    c2cc = np.sum(((CC[:,1] - H0*Ecc)/CC[:,2])**2)
    fac = C_KMS/H0/rd; c2bao = 0.0
    for bl in BAO_BLOCKS:
        z = bl['z']; Eh = np.interp(z, Z_LOW, E_lo)
        DM = fac*np.interp(z, Z_LOW, cum); DH = fac/Eh
        DV = (z*DM*DM*DH)**(1/3)
        mods = np.array([{'DM':DM,'DH':DH,'DV':DV}[k] for k in bl['kinds']])
        dv = bl['vals'] - mods
        c2bao += dv @ bl['icov'] @ dv
    E_hi = np.sqrt(Om*(1+Z_HIGH)**3 + OM_R*(1+Z_HIGH)**4 + 2*Lam/X0_FIX)
    Istar = trapezoid(1/E_lo, Z_LOW) + trapezoid(1/E_hi, Z_HIGH)
    R_th = np.sqrt(Om)*Istar; lA_th = np.pi*(C_KMS/H0)*Istar/rstar
    dv = np.array([R_th, lA_th, ob]) - CMB_MEAN
    c2cmb = dv @ CMB_ICOV @ dv
    DMsn = (C_KMS/H0)*np.interp(SN_Z, Z_LOW, cum)
    mu_th = 5*np.log10((1+SN_Z)*DMsn) + 25
    r = SN_MU - mu_th
    c2sn = r @ C_SN_inv @ r - (r @ Binv_MB)**2/Ainv_MB
    Om_z = Om*(1+Z_LOW)**3/E_lo**2
    fQ = +1 + a*(3-xs)/(1+xs)**3; Ge = +1/fQ
    f_z = (Om_z*Ge)**GAMMA
    cf = np.concatenate(([0.0], cumulative_trapezoid(f_z/(1+Z_LOW), Z_LOW)))
    fs8_full = s8 * f_z * np.exp(-cf)
    fs8_mod = np.interp(FS8_Z, Z_LOW, fs8_full)
    c2fs8 = np.sum(((FS8_V - fs8_mod)/FS8_S)**2)
    tot = c2cc + c2bao + c2cmb + c2sn + c2fs8
    if return_parts: return tot, (c2cc, c2bao, c2cmb, c2sn, c2fs8)
    return tot

def chi2_LCDM(th):
    Om, H0, ob, s8 = th
    if not (0.1<Om<0.5 and 55<H0<85 and 0.018<ob<0.026 and 0.5<s8<1.1): return 1e10
    rd = rd_Aub(Om, H0, ob); rstar = rd*rs_ratio(Om, H0, ob)
    z_all = np.concatenate([Z_LOW, Z_HIGH]); E_all = np.sqrt(Om*(1+z_all)**3+OM_R*(1+z_all)**4+(1-Om-OM_R))
    Ecc = np.interp(CC[:,0], z_all, E_all)
    c2cc = np.sum(((CC[:,1]-H0*Ecc)/CC[:,2])**2)
    cum = np.concatenate(([0.0], cumulative_trapezoid(1/E_all, z_all)))
    fac = C_KMS/H0/rd; c2bao = 0.
    for bl in BAO_BLOCKS:
        z = bl['z']; Eh = np.interp(z, z_all, E_all)
        DM = fac*np.interp(z, z_all, cum); DH = fac/Eh
        DV = (z*DM*DM*DH)**(1/3)
        mods = np.array([{'DM':DM,'DH':DH,'DV':DV}[k] for k in bl['kinds']])
        dv = bl['vals'] - mods
        c2bao += dv @ bl['icov'] @ dv
    Istar = trapezoid(1/E_all, z_all)
    R_th = np.sqrt(Om)*Istar; lA_th = np.pi*(C_KMS/H0)*Istar/rstar
    dv = np.array([R_th, lA_th, ob]) - CMB_MEAN
    c2cmb = dv @ CMB_ICOV @ dv
    cum_lo = np.interp(Z_LOW, z_all, cum); E_lo = np.interp(Z_LOW, z_all, E_all)
    DMsn = (C_KMS/H0)*np.interp(SN_Z, Z_LOW, cum_lo)
    mu_th = 5*np.log10((1+SN_Z)*DMsn) + 25
    r = SN_MU - mu_th
    c2sn = r @ C_SN_inv @ r - (r @ Binv_MB)**2/Ainv_MB
    Om_z = Om*(1+Z_LOW)**3/E_lo**2
    f_z = Om_z**GAMMA
    cf = np.concatenate(([0.0], cumulative_trapezoid(f_z/(1+Z_LOW), Z_LOW)))
    fs8 = s8*f_z*np.exp(-cf)
    c2fs8 = np.sum(((FS8_V - np.interp(FS8_Z, Z_LOW, fs8))/FS8_S)**2)
    return c2cc+c2bao+c2cmb+c2sn+c2fs8

def in_prior(th):
    a, Om, H0, ob, s8 = th
    return (-0.33<a<0.80 and 0.10<Om<0.50 and 55<H0<85 and
            0.018<ob<0.026 and 0.5<s8<1.1)
def log_prob(th):
    if not in_prior(th): return -np.inf
    c = chi2_fQ(th)
    return -0.5*c if np.isfinite(c) else -np.inf

print("MAP f(Q)...")
res = minimize(lambda x: -log_prob(x), [0.0, 0.3, 68.0, 0.02236, 0.80],
               method='Nelder-Mead',
               options={'xatol':1e-6, 'fatol':1e-4, 'maxiter':10000, 'adaptive':True})
map_fQ = res.x; chi2_map, parts = chi2_fQ(map_fQ, return_parts=True)
print(f"  a={map_fQ[0]:+.4f} Om={map_fQ[1]:.4f} H0={map_fQ[2]:.2f} "
      f"omb={map_fQ[3]:.5f} s8={map_fQ[4]:.4f}")
print(f"  chi2={chi2_map:.2f} [CC={parts[0]:.1f} BAO={parts[1]:.1f} "
      f"CMB={parts[2]:.2f} SN={parts[3]:.1f} fs8={parts[4]:.1f}]")

resL = minimize(chi2_LCDM, [0.3,68,0.02236,0.80], method='Nelder-Mead',
                options={'xatol':1e-6,'fatol':1e-4,'maxiter':5000,'adaptive':True})
chi2_L = resL.fun
print(f"\nMAP LCDM: chi2={chi2_L:.2f}  Delta(LCDM-fQ)={chi2_L-chi2_map:+.2f}")

def mh_chain(log_p, start, init_scale, n_burn, n_prod, rng):
    ndim = len(start)
    chol = np.linalg.cholesky(np.diag(init_scale**2))
    scale = 1.0; cur = start.copy(); cur_lp = log_p(cur)
    burn = np.zeros((n_burn, ndim)); acc = 0; win = 200
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
            acc=0
    cov_emp = np.cov(burn[n_burn//2:].T) + 1e-12*np.eye(ndim)
    chol = np.linalg.cholesky(cov_emp)
    prod = np.zeros((n_prod, ndim)); acc = 0
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

print("\nMCMC (4 catene)...")
init_scale = np.array([0.05, 0.006, 0.6, 0.00015, 0.02])
chains = []
for ch in range(4):
    rng = np.random.default_rng(7000+ch)
    st = map_fQ + 0.3*init_scale*rng.standard_normal(5)
    if not in_prior(st): st = map_fQ.copy()
    print(f"  Chain {ch+1}/4...", end=" ", flush=True)
    tc = time()
    s, acc = mh_chain(log_prob, st, init_scale, 3000, 10000, rng)
    chains.append(s); print(f"acc={acc:.2f}, {time()-tc:.0f}s")
chains = np.array(chains)

Rhat = gelman_rubin(chains)
names = ['alpha','Om0','H0','ombh2','sigma8']
print("\nR-hat:")
for nm, r_ in zip(names, Rhat):
    flag = "OK" if r_<1.05 else ("~" if r_<1.1 else "!!!")
    print(f"  {nm:>7}: {r_:.4f} {flag}")

flat = chains.reshape(-1, 5)[::3]
q = np.quantile(flat, [0.16,0.5,0.84], axis=0)
print("\nMarginali:")
for nm, qq in zip(names, q.T):
    print(f"  {nm:>7} = {qq[1]:+11.5f} +{qq[2]-qq[1]:.5f}/-{qq[1]-qq[0]:.5f}")

am, asd = np.median(flat[:,0]), np.std(flat[:,0])
print(f"\nalpha = {am:+.4f} +/- {asd:.4f} => {abs(am)/asd:.2f} sigma")
print(f"frac(alpha<0) = {np.mean(flat[:,0]<0):.1%}")

AIC_fQ = chi2_map + 2*5; AIC_L = chi2_L + 2*4
BIC_fQ = chi2_map + 5*np.log(N_TOT); BIC_L = chi2_L + 4*np.log(N_TOT)
print(f"\nDelta chi2 (LCDM-fQ) = {chi2_L-chi2_map:+.2f}")
print(f"Delta AIC  = {AIC_L-AIC_fQ:+.2f}")
print(f"Delta BIC  = {BIC_L-BIC_fQ:+.2f}")

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "plots", "13_Union3_results.npz"),
         flat=flat, map_fQ=map_fQ, chi2_map=chi2_map, chi2_L=chi2_L, Rhat=Rhat)
print(f"\nRisultati salvati")
