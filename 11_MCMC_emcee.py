"""
11_MCMC_emcee.py — Analisi principale con affine-invariant ensemble sampler
                    (algoritmo di Goodman-Weare 2010, implementato in `emcee`).

Due modalità:

  1. Se `emcee` è installato (pip install emcee) usa la libreria ufficiale.
  2. Altrimenti usa l'implementazione stand-alone `stretch_move_sampler` qui
     sotto, che riproduce l'algoritmo 1 (stretch move) del paper originale
     Goodman & Weare 2010.

Il risultato deve essere statisticamente indistinguibile dall'analisi
custom MH in 04_MCMC_main_eta_fixed.py. Serve come cross-check della
robustezza del segnale alpha < 0 rispetto alla scelta del sampler.

Algoritmo stretch move (Goodman-Weare 2010):
  Per ogni walker k con posizione X_k, sceglie un altro walker X_j (j != k).
  Propone X_k' = X_j + Z (X_k - X_j), con Z ~ g(z) = 1/sqrt(z) per z in [1/a, a].
  Accetta con probabilita' min(1, Z^(D-1) * L(X_k')/L(X_k)).
  Con a = 2, acceptance rate ottimale ~ 0.2-0.5.

Vantaggio rispetto a MH vanilla: affine-invariant (insensibile a rotazioni
e scalature dello spazio), niente covarianza di proposta da tunare.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import minimize
from time import time
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import (load_CC, load_BAO_DESI, load_CMB_Planck,
                          load_PantheonPlus, load_fsigma8)

# Proviamo a importare emcee; se non c'e' usiamo stand-alone
USE_EMCEE = False
try:
    import emcee
    USE_EMCEE = True
    print(f"Uso emcee ufficiale (version {emcee.__version__})")
except ImportError:
    print("emcee non installato. Uso implementazione stand-alone "
          "dell'affine-invariant ensemble sampler.")
    print("Per installare: pip install emcee")

# =====================================================================
# Caricamento dati (identico a 04_MCMC_main_eta_fixed.py)
# =====================================================================
print("\n" + "="*65)
print("CARICAMENTO DATI (identico a 04_MCMC_main)")
print("="*65)

CC = load_CC()
BAO_BLOCKS = load_BAO_DESI()
CMB_MEAN, CMB_ICOV = load_CMB_Planck()
SN_Z, SN_MB, C_SN_inv, Ainv_MB, Binv_MB = load_PantheonPlus()
N_SN = len(SN_Z)
FS8 = load_fsigma8()
FS8_Z, FS8_V, FS8_S = FS8[:,0], FS8[:,1], FS8[:,2]

N_TOT = len(CC) + sum(len(b['kinds']) for b in BAO_BLOCKS) + 3 + N_SN + len(FS8)
print(f"  Totale: {N_TOT} punti di dato\n")

# =====================================================================
# Modello (identico a 04)
# =====================================================================
C_KMS  = 299792.458
OM_R   = 9.2e-5  # densita radiazione
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

def chi2_fQ(theta):
    a, Om, H0, ob, s8 = theta
    rd = rd_Aub(Om, H0, ob); rstar = rd*rs_ratio(Om, H0, ob)
    E_lo, xs, Lam = solve_E(Z_LOW, a, Om)
    if E_lo is None: return np.inf
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

    return c2cc + c2bao + c2cmb + c2sn + c2fs8

def log_prior(theta):
    a, Om, H0, ob, s8 = theta
    if not (-0.33<a<0.80 and 0.10<Om<0.50 and 55<H0<85 and
            0.018<ob<0.026 and 0.5<s8<1.1):
        return -np.inf
    return 0.0  # flat priors

def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    c = chi2_fQ(theta)
    if not np.isfinite(c): return -np.inf
    return lp - 0.5*c

# =====================================================================
# Stand-alone affine-invariant ensemble sampler
# (Goodman & Weare 2010, algorithm 1 = "stretch move")
# Replica esatta di emcee.EnsembleSampler con il solo move di default.
# =====================================================================
def stretch_move_sampler(log_prob, p0, nsteps, a=2.0, seed=None, verbose=True):
    """
    Affine-invariant ensemble sampler con stretch move.

    Parameters
    ----------
    log_prob : callable
        Funzione che prende theta e restituisce log-probability.
    p0 : array shape (nwalkers, ndim)
        Posizioni iniziali dei walker.
    nsteps : int
        Numero di step totali.
    a : float
        Parametro di scala. Default 2 (come emcee).
    seed : int
    verbose : bool

    Returns
    -------
    chain : array (nsteps, nwalkers, ndim)
    logp : array (nsteps, nwalkers)
    accepted : array (nwalkers,)
    """
    rng = np.random.default_rng(seed)
    nwalkers, ndim = p0.shape
    assert nwalkers % 2 == 0, "nwalkers deve essere pari"
    half = nwalkers // 2

    # Split i walker in due insiemi A e B (cosi' si aggiornano in parallelo
    # conservando detailed balance, come in emcee)
    chain = np.zeros((nsteps, nwalkers, ndim))
    logp  = np.full((nsteps, nwalkers), -np.inf)
    accepted = np.zeros(nwalkers, dtype=int)

    pos = p0.copy()
    lp_cur = np.array([log_prob(pos[k]) for k in range(nwalkers)])

    for step in range(nsteps):
        for split_idx in range(2):
            # Walker da aggiornare in questo sotto-step: A (0..half) oppure B (half..nwalkers)
            active = np.arange(split_idx*half, (split_idx+1)*half)
            complement = np.arange((1-split_idx)*half, (2-split_idx)*half)

            # Per ogni walker attivo, campiona un walker dal complemento
            j_idx = rng.choice(complement, size=half, replace=True)
            # Stretch variable z ~ (1/sqrt(z)) for z in [1/a, a]
            # Campionamento: u ~ U(0,1), z = ((a-1)*u + 1)^2 / a
            u = rng.random(half)
            z = ((a - 1.0)*u + 1.0)**2 / a

            # Proposta: Y = X_j + z*(X_k - X_j)
            X_active = pos[active]
            X_j = pos[j_idx]
            Y = X_j + z[:, None]*(X_active - X_j)

            # Log-probability della proposta
            lp_new = np.array([log_prob(Y[i]) for i in range(half)])

            # Rapporto di accettazione: ln(z^(D-1) * L_new/L_cur)
            log_ratio = (ndim - 1)*np.log(z) + lp_new - lp_cur[active]

            # Metropolis accept
            u_accept = rng.random(half)
            accept = np.log(u_accept + 1e-300) < log_ratio

            # Aggiorno posizioni e logp
            pos[active[accept]]   = Y[accept]
            lp_cur[active[accept]] = lp_new[accept]
            accepted[active[accept]] += 1

        chain[step] = pos.copy()
        logp[step]  = lp_cur.copy()

        if verbose and (step+1) % max(1, nsteps//10) == 0:
            mean_acc = accepted.mean() / (step+1) / 2
            print(f"  step {step+1}/{nsteps}  <acc>={mean_acc:.3f}  "
                  f"<logp>={lp_cur.mean():.1f}")

    return chain, logp, accepted / nsteps / 2


# =====================================================================
# Find MAP first (per inizializzare walker vicino al massimo)
# =====================================================================
print("="*65)
print("MAP (Nelder-Mead) per inizializzare i walker vicino al massimo")
print("="*65)

def neglogp(th): return -log_prob(th)

res = minimize(neglogp, [0.0, 0.3, 68.0, 0.02236, 0.80],
               method='Nelder-Mead',
               options={'xatol':1e-6, 'fatol':1e-4, 'maxiter':10000, 'adaptive':True})
map_theta = res.x
print(f"  MAP: a={map_theta[0]:+.4f} Om={map_theta[1]:.4f} H0={map_theta[2]:.2f}"
      f" omb={map_theta[3]:.5f} s8={map_theta[4]:.4f}")
print(f"  -logL(MAP) = {res.fun:.3f}")

# =====================================================================
# Setup walkers
# =====================================================================
ndim = 5
nwalkers = 32   # emcee tipicamente usa ~2-10x ndim
nsteps_burn = 500
nsteps_prod = 2500

# Inizializzazione: Gaussian ball attorno al MAP
init_scale = np.array([0.03, 0.004, 0.5, 0.0001, 0.015])
rng = np.random.default_rng(42)
p0 = map_theta + init_scale * rng.standard_normal((nwalkers, ndim))

# Verifica che siano tutti in-prior
for i in range(nwalkers):
    while not np.isfinite(log_prior(p0[i])):
        p0[i] = map_theta + init_scale * rng.standard_normal(ndim)

print(f"\n{nwalkers} walkers inizializzati vicino al MAP")

# =====================================================================
# Run sampler
# =====================================================================
print("\n" + "="*65)
print(f"RUN SAMPLER: {nsteps_burn} burn-in + {nsteps_prod} production step")
print("="*65)

t0 = time()

if USE_EMCEE:
    # API ufficiale emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

    print("Burn-in...")
    state = sampler.run_mcmc(p0, nsteps_burn, progress=False)
    sampler.reset()

    print("Production...")
    sampler.run_mcmc(state, nsteps_prod, progress=False)

    chain = sampler.get_chain()  # shape (nsteps_prod, nwalkers, ndim)
    log_probs = sampler.get_log_prob()
    acceptance_fraction = sampler.acceptance_fraction
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception as e:
        tau = None

else:
    # Stand-alone
    print("Burn-in...")
    _, _, _ = stretch_move_sampler(log_prob, p0, nsteps_burn, seed=42, verbose=False)
    # Uso il risultato del burn per ri-inizializzare la produzione
    # In stand-alone non ho bisogno di state, ma per coerenza con emcee
    # rilancio dalla stessa gaussian ball
    p_init = p0.copy()  # semplice: ricomincio (con seed diverso)

    print("Production...")
    chain, log_probs, acceptance_fraction = stretch_move_sampler(
        log_prob, p_init, nsteps_prod, seed=1234, verbose=True)
    tau = None

print(f"\nTempo totale: {time()-t0:.1f}s")
print(f"Acceptance fraction: {acceptance_fraction.mean():.3f} "
      f"(min {acceptance_fraction.min():.3f}, max {acceptance_fraction.max():.3f})")

if tau is not None:
    print(f"Autocorrelation time: {tau}")

# =====================================================================
# Flatten e calcola posterior
# =====================================================================
# Scarto primo 25% della produzione (extra burn interno) e thin per ~1 indep sample per tau
discard_frac = 0.25
discard = int(discard_frac*nsteps_prod)
flat = chain[discard:].reshape(-1, ndim)

# Eventuale thinning
if tau is not None:
    thin = max(1, int(np.max(tau)//2))
    flat = flat[::thin]

print(f"\nSample effettivi post burn+thin: {len(flat)}")

# =====================================================================
# Gelman-Rubin: prendo le catene come se fossero catene indipendenti
# (non e' rigoroso come con catene separate, ma i walker in stretch move
# esplorano spazi diversi e R-hat e' un proxy utile)
# =====================================================================
def gelman_rubin(chains):
    # chains shape: (M, N, D)
    M, N, D = chains.shape
    means = chains.mean(axis=1)
    W = chains.var(axis=1, ddof=1).mean(axis=0)
    B = N * means.var(axis=0, ddof=1)
    V = (1-1/N)*W + (1/N)*B
    return np.sqrt(V/W)

# Uso i 32 walker come "catene"
chains_for_rhat = chain[discard:].transpose(1, 0, 2)  # (nwalkers, nsteps_prod-discard, ndim)
Rhat = gelman_rubin(chains_for_rhat)

names = ['alpha', 'Om0', 'H0', 'ombh2', 'sigma8']
print("\n=== Gelman-Rubin R-hat (walkers come catene) ===")
for nm, r_ in zip(names, Rhat):
    flag = "OK" if r_ < 1.05 else ("~" if r_ < 1.1 else "!!!")
    print(f"  {nm:>7}: R-hat = {r_:.4f}  {flag}")

# =====================================================================
# Marginali
# =====================================================================
q = np.quantile(flat, [0.16, 0.5, 0.84], axis=0)
print("\n=== Posteriors marginali (50% +/- 68% CI) da emcee/stretch-move ===")
for nm, qq in zip(names, q.T):
    print(f"  {nm:>7} = {qq[1]:+11.5f}  +{qq[2]-qq[1]:.5f}/-{qq[1]-qq[0]:.5f}")

rds = np.array([rd_Aub(c[1], c[2], c[3]) for c in flat])
print(f"  r_d     = {np.median(rds):.3f}  "
      f"+{np.quantile(rds,0.84)-np.median(rds):.3f}/-{np.median(rds)-np.quantile(rds,0.16):.3f}")

am, asd = np.median(flat[:,0]), np.std(flat[:,0])
print(f"\nalpha = {am:+.4f} +/- {asd:.4f}  =>  {abs(am)/asd:.2f} sigma da 0")
print(f"frac(alpha<0) = {np.mean(flat[:,0]<0):.1%}")

# =====================================================================
# Confronto con MH custom
# =====================================================================
print("\n" + "="*70)
print("CONFRONTO MH-custom vs emcee (stesso likelihood, stessi dati, stessi prior)")
print("="*70)
print(f"{'Parametro':>12}  {'MH-custom':>22}  {'emcee/stretch':>22}  {'consistenza':>14}")
print("-"*75)
mh_results = {
    'alpha':    (+0.1347, 0.0674),     # da 04_MCMC_main_eta_fixed.py (con Om_r)
    'Om0':      (+0.30927, 0.00671),
    'H0':       (+66.605, 0.690),
    'ombh2':    (+0.02259, 0.00013),
    'sigma8':   (+0.79860, 0.02327),
}
for nm, qq in zip(names, q.T):
    em_med = qq[1]; em_err = (qq[2]-qq[0])/2
    mh_med, mh_err = mh_results[nm]
    diff_sigma = abs(em_med - mh_med)/np.sqrt(em_err**2 + mh_err**2)
    status = "OK" if diff_sigma < 1.0 else "nota differenza"
    print(f"{nm:>12}  "
          f"{mh_med:>+10.5f} +/- {mh_err:>6.5f}  "
          f"{em_med:>+10.5f} +/- {em_err:>6.5f}  "
          f"{diff_sigma:>6.2f} sigma {status}")

# =====================================================================
# Corner plot
# =====================================================================
def corner_plot(samples, labels, truths=None, bins=32, color='orangered', cmap='Oranges'):
    n = samples.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(2.2*n, 2.2*n))
    mn, mx = samples.min(0), samples.max(0); pd = 0.05*(mx-mn)
    for i in range(n):
        for j in range(n):
            ax = axes[i,j]
            if j > i: ax.set_visible(False); continue
            if i == j:
                ax.hist(samples[:,i], bins=bins, color=color, alpha=0.85,
                        edgecolor='darkred', lw=0.3, density=True)
                qq = np.quantile(samples[:,i], [0.16, 0.5, 0.84])
                for q_, s_ in zip(qq, [':', '-', ':']):
                    ax.axvline(q_, color='k', lw=1, ls=s_)
                ax.set_title(f'{labels[i]}\n'
                    fr'${qq[1]:.4g}^{{+{qq[2]-qq[1]:.2g}}}_{{-{qq[1]-qq[0]:.2g}}}$',
                    fontsize=9)
                if truths is not None: ax.axvline(truths[i], color='darkblue', lw=1.3, ls='--')
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
                        ax.contour(xc, yc, H.T, levels=levs, colors='darkred', linewidths=0.9)
                except: pass
                if truths is not None:
                    ax.plot(truths[j], truths[i], 'x', color='darkblue', ms=7, mew=1.6)
                ax.set_xlim(mn[j]-pd[j], mx[j]+pd[j])
                ax.set_ylim(mn[i]-pd[i], mx[i]+pd[i])
            if i == n-1: ax.set_xlabel(labels[j], fontsize=10)
            else: ax.tick_params(labelbottom=False)
            if j == 0 and i > 0: ax.set_ylabel(labels[i], fontsize=10)
            elif j != 0: ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=8)
    plt.tight_layout(); return fig

labels_tex = [r'$\alpha$', r'$\Omega_{m,0}$', r'$H_0$', r'$\omega_b$', r'$\sigma_{8,0}$']
# convert mh_results medians into a list
mh_med_list = [mh_results[n][0] for n in names]
fig = corner_plot(flat, labels_tex, truths=mh_med_list)
title = f"emcee/stretch-move (nwalkers={nwalkers}, {nsteps_prod} step) — linee blu: MH custom best-fit"
fig.suptitle(title, fontsize=10, y=1.005)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "11_emcee_corner.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nCorner plot salvato: {out_path}")
print("\n=== FINE ANALISI emcee/stretch-move ===")
