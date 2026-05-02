"""
04_MCMC_main_eta_fixed.py — Analisi principale.

Modello f(Q) con phantom crossing, eta = sqrt(6) fissato (x_0 = 1).
5 parametri liberi: (alpha, Om0, H0, ombh2, sigma8_0)

Dataset: CC + DESI BAO + CMB Planck + Pantheon+ (1580 SNe) + f*sigma_8.

Output: corner plot, posteriors, tabella parametri con R-hat, confronto LCDM.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import minimize
from time import time
import os

# Aggiunge 'scripts/' al path per importare data_loaders
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import (load_CC, load_BAO_DESI, load_CMB_Planck,
                          load_PantheonPlus, load_fsigma8)

# =====================================================================
# Caricamento dati
# =====================================================================
print("="*65)
print("CARICAMENTO DATI")
print("="*65)

CC = load_CC()
BAO_BLOCKS = load_BAO_DESI()
CMB_MEAN, CMB_ICOV = load_CMB_Planck()
SN_Z, SN_MB, C_SN_inv, Ainv_MB, Binv_MB = load_PantheonPlus()
N_SN = len(SN_Z)
FS8 = load_fsigma8()
FS8_Z, FS8_V, FS8_S = FS8[:,0], FS8[:,1], FS8[:,2]

N_TOT = len(CC) + sum(len(b['kinds']) for b in BAO_BLOCKS) + 3 + N_SN + len(FS8)
print(f"  CC:       {len(CC)} pt")
print(f"  BAO:      {sum(len(b['kinds']) for b in BAO_BLOCKS)} pt (DESI DR1)")
print(f"  CMB:      3 pt (Planck 2018 compressed)")
print(f"  SN:       {N_SN} pt (Pantheon+)")
print(f"  fsigma8:  {len(FS8)} pt (Gold-18)")
print(f"  TOTALE:   {N_TOT} punti di dato")

# =====================================================================
# Modello (eta = sqrt(6) fissato)
# =====================================================================
C_KMS  = 299792.458
OM_R   = 9.2e-5  # densita' radiazione (Tcmb=2.7255 K, N_eff=3.046)
Z_STAR = 1089.80
GAMMA  = 0.55
ETA_FIX = np.sqrt(6.0)
X0_FIX = 1.0

X_SAMP = np.unique(np.concatenate([
    np.linspace(1e-5, 0.3, 100),
    np.linspace(0.3, 50, 300),
    np.linspace(50, 15000, 80),
]))
Z_LOW = np.linspace(1e-5, 15.0, 300)
Z_HIGH = np.linspace(15.0, Z_STAR, 400)

def Lam_of(a, Om0):
    return 0.5*((1.0-Om0)*X0_FIX + a*(1 + 6*X0_FIX - 3*X0_FIX*X0_FIX)/(1+X0_FIX)**3)

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

# =====================================================================
# Chi^2
# =====================================================================
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
    R_th = np.sqrt(Om)*Istar
    lA_th = np.pi*(C_KMS/H0)*Istar/rstar
    dv = np.array([R_th, lA_th, ob]) - CMB_MEAN
    c2cmb = dv @ CMB_ICOV @ dv

    DMsn = (C_KMS/H0)*np.interp(SN_Z, Z_LOW, cum)
    mu_th = 5*np.log10((1+SN_Z)*DMsn) + 25
    r = SN_MB - mu_th
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

def chi2_LCDM(th, return_parts=False):
    Om, H0, ob, s8 = th
    if not (0.1<Om<0.5 and 55<H0<85 and 0.018<ob<0.026 and 0.5<s8<1.1):
        return (1e10, None) if return_parts else 1e10
    rd = rd_Aub(Om, H0, ob); rstar = rd*rs_ratio(Om, H0, ob)
    z_all = np.concatenate([Z_LOW, Z_HIGH])
    E_all = np.sqrt(Om*(1+z_all)**3 + OM_R*(1+z_all)**4 + (1-Om-OM_R))
    Ecc = np.interp(CC[:,0], z_all, E_all)
    c2cc = np.sum(((CC[:,1] - H0*Ecc)/CC[:,2])**2)
    cum = np.concatenate(([0.0], cumulative_trapezoid(1/E_all, z_all)))
    fac = C_KMS/H0/rd; c2bao = 0.0
    for bl in BAO_BLOCKS:
        z = bl['z']; Eh = np.interp(z, z_all, E_all)
        DM = fac*np.interp(z, z_all, cum); DH = fac/Eh
        DV = (z*DM*DM*DH)**(1/3)
        mods = np.array([{'DM':DM,'DH':DH,'DV':DV}[k] for k in bl['kinds']])
        dv = bl['vals'] - mods
        c2bao += dv @ bl['icov'] @ dv
    Istar = trapezoid(1/E_all, z_all)
    R_th = np.sqrt(Om)*Istar
    lA_th = np.pi*(C_KMS/H0)*Istar/rstar
    dv = np.array([R_th, lA_th, ob]) - CMB_MEAN
    c2cmb = dv @ CMB_ICOV @ dv
    cum_lo = np.interp(Z_LOW, z_all, cum); E_lo = np.interp(Z_LOW, z_all, E_all)
    DMsn = (C_KMS/H0)*np.interp(SN_Z, Z_LOW, cum_lo)
    mu_th = 5*np.log10((1+SN_Z)*DMsn) + 25
    r = SN_MB - mu_th
    c2sn = r @ C_SN_inv @ r - (r @ Binv_MB)**2/Ainv_MB
    Om_z = Om*(1+Z_LOW)**3/E_lo**2
    f_z = Om_z**GAMMA
    cf = np.concatenate(([0.0], cumulative_trapezoid(f_z/(1+Z_LOW), Z_LOW)))
    fs8 = s8*f_z*np.exp(-cf)
    c2fs8 = np.sum(((FS8_V - np.interp(FS8_Z, Z_LOW, fs8))/FS8_S)**2)
    tot = c2cc + c2bao + c2cmb + c2sn + c2fs8
    if return_parts: return tot, (c2cc, c2bao, c2cmb, c2sn, c2fs8)
    return tot

def in_prior(th):
    a, Om, H0, ob, s8 = th
    return (-0.33<a<0.80 and 0.10<Om<0.50 and 55<H0<85 and
            0.018<ob<0.026 and 0.5<s8<1.1)

def log_prob(th):
    if not in_prior(th): return -np.inf
    c = chi2_fQ(th)
    return -0.5*c if np.isfinite(c) else -np.inf

# =====================================================================
# MAP
# =====================================================================
print("\n" + "="*65)
print(f"MAP f(Q)  [eta = {ETA_FIX:.4f} fissato]")
print("="*65)
t0 = time()
start = [0.0, 0.3, 68.0, 0.02236, 0.80]
res = minimize(lambda x: -log_prob(x), start, method='Nelder-Mead',
               options={'xatol':1e-6, 'fatol':1e-4, 'maxiter':10000, 'adaptive':True})
map_fQ = res.x
chi2_map, parts = chi2_fQ(map_fQ, return_parts=True)
print(f"  a={map_fQ[0]:+.4f}  Om={map_fQ[1]:.4f}  H0={map_fQ[2]:.2f}"
      f"  omb={map_fQ[3]:.5f}  s8={map_fQ[4]:.4f}")
print(f"  chi2={chi2_map:.2f} [CC={parts[0]:.1f} BAO={parts[1]:.1f} "
      f"CMB={parts[2]:.2f} SN={parts[3]:.1f} fs8={parts[4]:.1f}]  ({time()-t0:.1f}s)")

print("\nMAP LCDM")
resL = minimize(chi2_LCDM, [0.3, 68, 0.02236, 0.80], method='Nelder-Mead',
                options={'xatol':1e-6, 'fatol':1e-4, 'maxiter':5000, 'adaptive':True})
chi2_L, partsL = chi2_LCDM(resL.x, return_parts=True)
print(f"  Om={resL.x[0]:.4f}  H0={resL.x[1]:.2f}  omb={resL.x[2]:.5f}  s8={resL.x[3]:.4f}")
print(f"  chi2={chi2_L:.2f}  Delta(LCDM-fQ) = {chi2_L-chi2_map:+.2f}")

# =====================================================================
# MCMC multi-catena
# =====================================================================
def mh_chain(log_p, start, init_scale, n_burn, n_prod, rng):
    ndim = len(start)
    chol = np.linalg.cholesky(np.diag(init_scale**2))
    scale = 1.0; cur = start.copy(); cur_lp = log_p(cur)
    burn = np.zeros((n_burn, ndim)); acc = 0; win = 200
    for i in range(n_burn):
        prop = cur + scale*(chol @ rng.standard_normal(ndim))
        plp = log_p(prop)
        if plp - cur_lp > np.log(rng.random() + 1e-300):
            cur, cur_lp = prop, plp; acc += 1
        burn[i] = cur
        if (i+1) % win == 0 and i < n_burn-win:
            r = acc/win
            if r < 0.15:   scale *= 0.55
            elif r < 0.22: scale *= 0.85
            elif r > 0.55: scale *= 1.45
            elif r > 0.40: scale *= 1.15
            acc = 0
    cov_emp = np.cov(burn[n_burn//2:].T) + 1e-12*np.eye(ndim)
    chol = np.linalg.cholesky(cov_emp)
    prod = np.zeros((n_prod, ndim)); acc = 0
    for i in range(n_prod):
        prop = cur + chol @ rng.standard_normal(ndim)
        plp = log_p(prop)
        if plp - cur_lp > np.log(rng.random() + 1e-300):
            cur, cur_lp = prop, plp; acc += 1
        prod[i] = cur
    return prod, acc/n_prod

def gelman_rubin(chains):
    M, N, D = chains.shape
    means = chains.mean(axis=1)
    W = chains.var(axis=1, ddof=1).mean(axis=0)
    B = N * means.var(axis=0, ddof=1)
    V = (1 - 1/N)*W + (1/N)*B
    return np.sqrt(V/W)

print("\n" + "="*65)
print("MULTI-CHAIN MCMC (4 catene, 5 parametri)")
print("="*65)
t0 = time()
n_burn, n_prod = 3000, 10000
init_scale = np.array([0.05, 0.006, 0.6, 0.00015, 0.02])
chains = []
for ch in range(4):
    rng = np.random.default_rng(3000+ch)
    st = map_fQ + 0.3*init_scale*rng.standard_normal(5)
    if not in_prior(st): st = map_fQ.copy()
    print(f"  Chain {ch+1}/4...", end=" ", flush=True)
    tc = time()
    s, acc = mh_chain(log_prob, st, init_scale, n_burn, n_prod, rng)
    chains.append(s)
    print(f"acc={acc:.2f}, {time()-tc:.1f}s")
chains = np.array(chains)
print(f"Tempo totale MCMC: {time()-t0:.1f}s")

Rhat = gelman_rubin(chains)
names = ['alpha', 'Om0', 'H0', 'ombh2', 'sigma8']
print("\n=== Gelman-Rubin R-hat ===")
for nm, r_ in zip(names, Rhat):
    flag = "OK" if r_ < 1.05 else ("~" if r_ < 1.1 else "!!!")
    print(f"  {nm:>7}: R-hat = {r_:.4f}  {flag}")

flat = chains.reshape(-1, 5)[::3]
q = np.quantile(flat, [0.16, 0.5, 0.84], axis=0)
print("\n=== Marginali (50% +/- 68% CI) ===")
for nm, qq in zip(names, q.T):
    print(f"  {nm:>7} = {qq[1]:+11.5f}  +{qq[2]-qq[1]:.5f}/-{qq[1]-qq[0]:.5f}")
rds = np.array([rd_Aub(c[1], c[2], c[3]) for c in flat])
print(f"  r_d     = {np.median(rds):.3f}  +{np.quantile(rds,0.84)-np.median(rds):.3f}/"
      f"-{np.median(rds)-np.quantile(rds,0.16):.3f}")

am, asd = np.median(flat[:,0]), np.std(flat[:,0])
print(f"\nalpha = {am:+.4f} +/- {asd:.4f}  =>  {abs(am)/asd:.2f} sigma da 0")
print(f"frac(alpha<0) = {np.mean(flat[:,0]<0):.1%}")

AIC_fQ = chi2_map + 2*5; AIC_L = chi2_L + 2*4
BIC_fQ = chi2_map + 5*np.log(N_TOT); BIC_L = chi2_L + 4*np.log(N_TOT)
print(f"\n=== Model comparison (LCDM vs f(Q)) ===")
print(f"  Delta chi2 = {chi2_L-chi2_map:+.2f}")
print(f"  Delta AIC  = {AIC_L-AIC_fQ:+.2f}  "
      f"{'favorisce fQ' if AIC_L>AIC_fQ else 'favorisce LCDM'}")
print(f"  Delta BIC  = {BIC_L-BIC_fQ:+.2f}  "
      f"{'favorisce fQ' if BIC_L>BIC_fQ else 'favorisce LCDM'}")

# =====================================================================
# Corner plot
# =====================================================================
def corner_plot(samples, labels, truths=None, bins=32, color='steelblue', cmap='Blues'):
    n = samples.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(2.2*n, 2.2*n))
    mn, mx = samples.min(0), samples.max(0); pd = 0.05*(mx-mn)
    for i in range(n):
        for j in range(n):
            ax = axes[i,j]
            if j > i: ax.set_visible(False); continue
            if i == j:
                ax.hist(samples[:,i], bins=bins, color=color, alpha=0.85,
                        edgecolor='navy', lw=0.3, density=True)
                qq = np.quantile(samples[:,i], [0.16, 0.5, 0.84])
                for q_, s_ in zip(qq, [':', '-', ':']):
                    ax.axvline(q_, color='k', lw=1, ls=s_)
                ax.set_title(f'{labels[i]}\n'
                    fr'${qq[1]:.4g}^{{+{qq[2]-qq[1]:.2g}}}_{{-{qq[1]-qq[0]:.2g}}}$',
                    fontsize=9)
                if truths is not None: ax.axvline(truths[i], color='crimson', lw=1.3)
                ax.set_xlim(mn[i]-pd[i], mx[i]+pd[i]); ax.set_yticks([])
            else:
                H, xe, ye = np.histogram2d(samples[:,j], samples[:,i], bins=bins)
                ax.pcolormesh(xe, ye, H.T, cmap=cmap, shading='auto')
                flat_h = np.sort(H.ravel())[::-1]; cum = np.cumsum(flat_h); cum /= cum[-1]
                try:
                    l68 = flat_h[np.searchsorted(cum, 0.68)]
                    l95 = flat_h[np.searchsorted(cum, 0.95)]
                    levs = sorted(set([l95, l68]))
                    if len(levs) >= 1:
                        xc = 0.5*(xe[:-1]+xe[1:]); yc = 0.5*(ye[:-1]+ye[1:])
                        ax.contour(xc, yc, H.T, levels=levs, colors='navy', linewidths=0.9)
                except: pass
                if truths is not None:
                    ax.plot(truths[j], truths[i], 'x', color='crimson', ms=7, mew=1.6)
                ax.set_xlim(mn[j]-pd[j], mx[j]+pd[j])
                ax.set_ylim(mn[i]-pd[i], mx[i]+pd[i])
            if i == n-1: ax.set_xlabel(labels[j], fontsize=10)
            else: ax.tick_params(labelbottom=False)
            if j == 0 and i > 0: ax.set_ylabel(labels[i], fontsize=10)
            elif j != 0: ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=8)
    plt.tight_layout(); return fig

labels_tex = [r'$\alpha$', r'$\Omega_{m,0}$', r'$H_0$', r'$\omega_b$', r'$\sigma_{8,0}$']
fig = corner_plot(flat, labels_tex, truths=map_fQ)
fig.suptitle(r'Main MCMC, $\eta = \sqrt{6}$ fixed', fontsize=11, y=1.005)

# Output in ../plots/
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "04_main_corner.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nCorner plot salvato: {out_path}")
