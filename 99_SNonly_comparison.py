"""
99_SNonly_comparison.py
=======================
Run di confronto SN-only per i 3 datasets:
  - Pantheon+ (load_PantheonPlus dal data_loaders.py originale)
  - DES-Y5 STAT+SYS  (load_DESY5_full dal patch corretto v2)
  - Union3 binned    (load_Union3 dal patch corretto)

Il chi^2 include SOLO il termine SN, con M marginalizzato analiticamente.
H0, omega_b, sigma_8 sono FISSI (non vincolano niente nel chi^2 SN).
Parametri liberi: alpha, Omega_m  (2 dim invece di 5).

Output:
  - 99_SNonly_summary.png  (grafico riassuntivo con 3 corner sovrapposti)
  - terminale: tabella di confronto sigma_alpha tra i 3 datasets

Rationale:
  Nel run completo (CC + BAO + CMB + SN + fsigma8) il vincolo su alpha è
  dominato dai dati CMB+BAO+fsigma8 che sono CONDIVISI tra tutti i 3 run.
  Per vedere il VERO potere discriminante di ciascun dataset SN, bisogna
  isolarlo. In letteratura (modello w0wa, simile a f(Q)):
      Pantheon+ SN-only -> ~2.5 sigma
      DES-Y5 SN-only    -> ~3.9 sigma  <-- il piu' potente
      Union3 SN-only    -> ~3.5 sigma
  Se i risultati riproducono questa gerarchia il pipeline e' OK.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from time import time
import os, sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import loader (riusa quello che hai gia')
from data_loaders import load_PantheonPlus
from data_loaders_DESY5_patch import load_DESY5_full
from data_loaders_Union3_patch import load_Union3

# Costanti fisiche
C_KMS = 299792.458
OM_R  = 9.2e-5
ETA_FIX = np.sqrt(6.0)
X0_FIX  = 1.0
H0_FID  = 67.4   # arbitrario, M-marg lo assorbe

# Griglia z
Z_LOW = np.linspace(1e-5, 5.0, 400)
X_SAMP = np.unique(np.concatenate([
    np.linspace(1e-5, 0.3, 100), np.linspace(0.3, 50, 300),
    np.linspace(50, 15000, 80)]))


def Lam_of(a, Om0):
    return 0.5*((1.0-Om0)*X0_FIX + a*(1+6*X0_FIX-3*X0_FIX*X0_FIX)/(1+X0_FIX)**3)


def solve_E(z_arr, alpha, Om0):
    Lam = Lam_of(alpha, Om0)
    R_s = 2*Lam - alpha*(1+6*X_SAMP-3*X_SAMP**2)/(1+X_SAMP)**3
    g_s = X_SAMP - R_s
    if not np.all(np.diff(g_s) > 0): return None
    tgt = Om0*X0_FIX*(1+np.asarray(z_arr, float))**3
    xs = np.interp(tgt, g_s, X_SAMP)
    return np.sqrt(xs/X0_FIX)


# ============================================================
# CHI^2 SN-only generico
# ============================================================
def make_chi2_SNonly(SN_Z, SN_MU, C_SN_inv, Ainv, Binv):
    """Closure che cattura i dati SN e restituisce chi2 fQ e LCDM."""

    def chi2_fQ(theta):
        a, Om = theta
        if not (-2.0 < a < 2.0): return np.inf
        if not (0.05 < Om < 0.6): return np.inf
        E_lo = solve_E(Z_LOW, a, Om)
        if E_lo is None: return np.inf
        cum = np.concatenate(([0.0], cumulative_trapezoid(1/E_lo, Z_LOW)))
        DM = (C_KMS/H0_FID)*np.interp(SN_Z, Z_LOW, cum)
        mu_th = 5*np.log10((1+SN_Z)*DM) + 25
        r = SN_MU - mu_th
        return r @ C_SN_inv @ r - (r @ Binv)**2/Ainv

    def chi2_LCDM(theta):
        (Om,) = theta
        if not (0.05 < Om < 0.6): return np.inf
        E = np.sqrt(Om*(1+Z_LOW)**3 + OM_R*(1+Z_LOW)**4 + (1-Om-OM_R))
        cum = np.concatenate(([0.0], cumulative_trapezoid(1/E, Z_LOW)))
        DM = (C_KMS/H0_FID)*np.interp(SN_Z, Z_LOW, cum)
        mu_th = 5*np.log10((1+SN_Z)*DM) + 25
        r = SN_MU - mu_th
        return r @ C_SN_inv @ r - (r @ Binv)**2/Ainv

    return chi2_fQ, chi2_LCDM


# ============================================================
# Mini-MCMC Metropolis-Hastings 2D (veloce)
# ============================================================
def mh_chain_2d(log_p, start, scale, n_burn, n_prod, rng):
    cur = start.copy(); cur_lp = log_p(cur); ndim = len(cur)
    burn = np.zeros((n_burn, ndim)); acc = 0; win = 200
    for i in range(n_burn):
        prop = cur + scale*rng.standard_normal(ndim)
        plp = log_p(prop)
        if plp - cur_lp > np.log(rng.uniform()):
            cur, cur_lp = prop, plp; acc += 1
        burn[i] = cur
        if (i+1)%win == 0 and i < n_burn-win:
            r = acc/(i+1)
            if r < 0.15: scale *= 0.7
            elif r > 0.45: scale *= 1.3
    cov_emp = np.cov(burn[n_burn//2:].T) + 1e-12*np.eye(ndim)
    L = np.linalg.cholesky(cov_emp)
    prod = np.zeros((n_prod, ndim)); acc = 0
    for i in range(n_prod):
        prop = cur + 2.4/np.sqrt(ndim) * (L @ rng.standard_normal(ndim))
        plp = log_p(prop)
        if plp - cur_lp > np.log(rng.uniform()):
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


# ============================================================
# Routine completa per un dataset
# ============================================================
def run_SNonly(name, loader_fn):
    print(f"\n{'='*65}")
    print(f"  {name}  --  SN-only")
    print(f"{'='*65}")

    z, mu, Cinv, A, B = loader_fn()
    N_SN = len(z)
    print(f"  N_SN = {N_SN}")

    chi2_fQ, chi2_LCDM = make_chi2_SNonly(z, mu, Cinv, A, B)
    log_prob = lambda th: -0.5 * chi2_fQ(th)

    # MAP
    print("  MAP fQ...", end=" ", flush=True)
    res = minimize(chi2_fQ, [0.05, 0.30], method='Nelder-Mead',
                   options={'xatol':1e-5,'fatol':1e-3,'maxiter':3000})
    map_fQ = res.x; chi2_map = res.fun
    print(f"alpha={map_fQ[0]:+.4f}  Om={map_fQ[1]:.4f}  chi2={chi2_map:.2f}")

    # MAP LCDM
    res_L = minimize(chi2_LCDM, [0.30], method='Nelder-Mead',
                     options={'xatol':1e-5,'fatol':1e-3,'maxiter':3000})
    chi2_L = res_L.fun
    print(f"  MAP LCDM... Om={res_L.x[0]:.4f}  chi2={chi2_L:.2f}")
    print(f"  Delta chi^2 (LCDM-fQ) = {chi2_L-chi2_map:+.2f}")

    # 4 catene
    print("  MCMC (4 catene)...", end=" ", flush=True)
    init_scale = np.array([0.05, 0.02])
    chains = []
    for ch in range(4):
        rng = np.random.default_rng(8000+ch)
        st = map_fQ + 0.3*init_scale*rng.standard_normal(2)
        s, acc = mh_chain_2d(log_prob, st, init_scale, 3000, 10000, rng)
        chains.append(s)
    chains = np.array(chains)
    Rhat = gelman_rubin(chains)
    print(f"R-hat: alpha={Rhat[0]:.4f}, Om={Rhat[1]:.4f}")

    flat = chains.reshape(-1, 2)[::3]
    am = np.median(flat[:,0]); asd = np.std(flat[:,0])
    Omm = np.median(flat[:,1]); Omsd = np.std(flat[:,1])
    print(f"  alpha = {am:+.4f} +/- {asd:.4f}  =>  {abs(am)/asd:.2f} sigma")
    print(f"  Om    = {Omm:+.4f} +/- {Omsd:.4f}")
    print(f"  frac(alpha>0) = {np.mean(flat[:,0]>0):.1%}")

    return {
        'name': name,
        'N_SN': N_SN,
        'flat': flat,
        'alpha': (am, asd),
        'Om': (Omm, Omsd),
        'sigma': abs(am)/asd,
        'dchi2': chi2_L - chi2_map,
        'frac_pos': np.mean(flat[:,0]>0),
    }


# ============================================================
# Main: 3 run + plot di confronto
# ============================================================
print("="*65)
print("SN-ONLY COMPARISON")
print("Parametri liberi: alpha, Omega_m  (H0, omb, sigma8 fissati)")
print("="*65)

results = []

t0 = time()
results.append(run_SNonly("Pantheon+",      load_PantheonPlus))
results.append(run_SNonly("DES-Y5 STAT+SYS", load_DESY5_full))
results.append(run_SNonly("Union3 binned",   load_Union3))
print(f"\nTempo totale: {time()-t0:.0f}s")

# Tabella riassuntiva
print("\n" + "="*65)
print("RISULTATI SN-ONLY (atteso: DES-Y5 piu' stretto di Pantheon+)")
print("="*65)
print(f"{'Dataset':<22s} | {'N_SN':>6s} | {'alpha (median+/-1sigma)':<22s} | {'Sign.':>6s} | {'D.chi2':>7s}")
print("-"*80)
for r in results:
    a, da = r['alpha']
    print(f"{r['name']:<22s} | {r['N_SN']:>6d} | {a:+.4f} +/- {da:.4f}    | {r['sigma']:>5.2f}σ | {r['dchi2']:+7.2f}")

# Plot di confronto - 2 figure separate
print("\nGenerazione plot di confronto...")

colors = {'Pantheon+': 'royalblue',
          'DES-Y5 STAT+SYS': 'crimson',
          'Union3 binned': 'darkorange'}

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(out_dir, exist_ok=True)

# === Figure A: marginal posterior on alpha ===
fig_a, ax = plt.subplots(figsize=(7.5, 5.5))
for r in results:
    a, da = r['alpha']
    n = r['name']
    ax.hist(r['flat'][:,0], bins=60, density=True, alpha=0.5,
            color=colors[n], label=f"{n}: α={a:+.3f}±{da:.3f} ({r['sigma']:.2f}σ)")
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.7, label='ΛCDM (α=0)')
ax.set_xlabel(r'$\alpha$', fontsize=12)
ax.set_ylabel('Posterior density', fontsize=12)
ax.set_title('SN-only: marginal posterior on α', fontsize=12)
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
fig_a.savefig(os.path.join(out_dir, "99_SNonly_summary_a_alpha.png"), dpi=150, bbox_inches='tight')
plt.close(fig_a)

# === Figure B: 2D contours (alpha, Om) ===
fig_b, ax = plt.subplots(figsize=(7.5, 5.5))
for r in results:
    n = r['name']
    H, xe, ye = np.histogram2d(r['flat'][:,0], r['flat'][:,1], bins=40)
    flat_h = np.sort(H.ravel())[::-1]
    cum = np.cumsum(flat_h); cum /= cum[-1]
    l68 = flat_h[np.searchsorted(cum, 0.68)]
    l95 = flat_h[np.searchsorted(cum, 0.95)]
    xc = 0.5*(xe[:-1]+xe[1:]); yc = 0.5*(ye[:-1]+ye[1:])
    ax.contour(xc, yc, H.T, levels=[l95, l68], colors=colors[n],
               linewidths=1.6, alpha=0.85)
    ax.plot(np.median(r['flat'][:,0]), np.median(r['flat'][:,1]),
            'x', color=colors[n], ms=10, mew=2, label=n)
ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel(r'$\alpha$', fontsize=12)
ax.set_ylabel(r'$\Omega_{m,0}$', fontsize=12)
ax.set_title(r'SN-only: 68/95% contours in $(\alpha, \Omega_{m,0})$', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig_b.savefig(os.path.join(out_dir, "99_SNonly_summary_b_contours.png"), dpi=150, bbox_inches='tight')
plt.close(fig_b)
print(f"  Saved: 99_SNonly_summary_a_alpha.png and 99_SNonly_summary_b_contours.png in {out_dir}")

# Salva i flat samples per uso futuro
np.savez(os.path.join(out_dir, "99_SNonly_results.npz"),
         pan_flat=results[0]['flat'],
         des_flat=results[1]['flat'],
         u3_flat=results[2]['flat'])
print(f"  Salvato: {out_dir}/99_SNonly_results.npz")
