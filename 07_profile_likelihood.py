"""
07_profile_likelihood.py — Profile likelihood 1-parametro su alpha.

Per ogni valore fissato di alpha, minimizza chi2 sugli altri 5 parametri (inclusa eta
libera). Rivela la forma vera della chi2(alpha) oltre l'approssimazione Gaussiana
e diagnostica la direzione piatta (alpha, eta).
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

print("Caricamento dati...")
CC = load_CC()
BAO_BLOCKS = load_BAO_DESI()
CMB_MEAN, CMB_ICOV = load_CMB_Planck()
SN_Z, SN_MB, C_SN_inv, Ainv_MB, Binv_MB = load_PantheonPlus()
FS8 = load_fsigma8(); FS8_Z, FS8_V, FS8_S = FS8[:,0], FS8[:,1], FS8[:,2]

C_KMS=299792.458; OM_R = 9.2e-5; Z_STAR=1089.80; GAMMA=0.55
X_SAMP = np.unique(np.concatenate([
    np.linspace(1e-5, 0.3, 100), np.linspace(0.3, 50, 300), np.linspace(50, 15000, 80)]))
Z_LOW = np.linspace(1e-5, 15.0, 300); Z_HIGH = np.linspace(15.0, Z_STAR, 400)

def Lam_of(a, eta, Om0):
    x0 = 6.0/eta**2
    return 0.5*((1.0-Om0)*x0 + a*(1 + 6*x0 - 3*x0*x0)/(1+x0)**3)

def solve_E(z_arr, alpha, eta, Om0):
    x0 = 6.0/eta**2; Lam = Lam_of(alpha, eta, Om0)
    R_s = 2*Lam - alpha*(1+6*X_SAMP-3*X_SAMP**2)/(1+X_SAMP)**3
    g_s = X_SAMP - R_s
    if not np.all(np.diff(g_s) > 0): return None, None, None
    tgt = Om0*x0*(1+np.asarray(z_arr, float))**3
    xs = np.interp(tgt, g_s, X_SAMP)
    return np.sqrt(xs/x0), xs, Lam

def rd_Aub(Om, H0, ob):
    h=H0/100; return 55.154/((Om*h*h)**0.25351 * ob**0.12807)
def rs_ratio(Om, H0, ob):
    h=H0/100; return 1.0 - 0.0206*ob**0.165 * (Om*h*h)**0.05

def chi2_fQ(theta):
    a, eta, Om, H0, ob, s8 = theta
    rd = rd_Aub(Om,H0,ob); rstar = rd*rs_ratio(Om,H0,ob)
    E_lo, xs, Lam = solve_E(Z_LOW, a, eta, Om)
    if E_lo is None: return np.inf
    cum = np.concatenate(([0.0], cumulative_trapezoid(1/E_lo, Z_LOW)))
    Ecc = np.interp(CC[:,0], Z_LOW, E_lo)
    c2cc = np.sum(((CC[:,1]-H0*Ecc)/CC[:,2])**2)
    fac = C_KMS/H0/rd; c2bao=0.
    for bl in BAO_BLOCKS:
        z=bl['z']; Eh=np.interp(z,Z_LOW,E_lo)
        DM = fac*np.interp(z,Z_LOW,cum); DH=fac/Eh
        DV = (z*DM*DM*DH)**(1/3)
        mods = np.array([{'DM':DM,'DH':DH,'DV':DV}[k] for k in bl['kinds']])
        dv = bl['vals']-mods; c2bao += dv @ bl['icov'] @ dv
    x0 = 6.0/eta**2
    E_hi = np.sqrt(Om*(1+Z_HIGH)**3 + OM_R*(1+Z_HIGH)**4 + 2*Lam/x0)
    Istar = trapezoid(1/E_lo, Z_LOW) + trapezoid(1/E_hi, Z_HIGH)
    R_th = np.sqrt(Om)*Istar; lA_th = np.pi*(C_KMS/H0)*Istar/rstar
    dv = np.array([R_th,lA_th,ob])-CMB_MEAN
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

# Profile: per ogni alpha, ottimizza (eta, Om, H0, omb, s8)
alphas = np.linspace(-0.05, 0.40, 30)
chi2_prof = np.zeros_like(alphas)
params_prof = np.zeros((len(alphas), 6))

print("\nProfiling chi2(alpha) a eta = sqrt(6) fisso ...")
t0 = time()
ETA_PROF = np.sqrt(6.0)
start = np.array([0.3, 67.5, 0.02244, 0.80])  # Om, H0, ob, s8
for i, a in enumerate(alphas):
    def f_of(x):
        Om, H0, ob, s8 = x
        if not (0.1<Om<0.5 and 55<H0<85 and 0.018<ob<0.026 and 0.5<s8<1.1):
            return 1e8
        return chi2_fQ([a, ETA_PROF, Om, H0, ob, s8])
    r = minimize(f_of, start, method='Nelder-Mead',
                 options={'xatol':1e-5, 'fatol':1e-3, 'maxiter':3000, 'adaptive':True})
    chi2_prof[i] = r.fun
    start = r.x
    params_prof[i] = np.concatenate(([a], [ETA_PROF], r.x))
    if i%5==0:
        print(f"  a={a:+.3f}  chi2={r.fun:.2f}  Om={r.x[0]:.4f}"
              f"  H0={r.x[1]:.2f}")
print(f"Done in {time()-t0:.1f}s\n")

chi2_min = chi2_prof.min()
a_best = alphas[np.argmin(chi2_prof)]
print(f"chi2_min = {chi2_min:.2f} at alpha = {a_best:+.3f}")

# LCDM reference
def chi2_LCDM(th):
    Om,H0,ob,s8=th
    if not (0.1<Om<0.5 and 55<H0<85 and 0.018<ob<0.026 and 0.5<s8<1.1): return 1e10
    rd=rd_Aub(Om,H0,ob); rstar=rd*rs_ratio(Om,H0,ob)
    z_all=np.concatenate([Z_LOW,Z_HIGH]); E_all=np.sqrt(Om*(1+z_all)**3+OM_R*(1+z_all)**4+(1-Om-OM_R))
    Ecc=np.interp(CC[:,0],z_all,E_all)
    c2cc=np.sum(((CC[:,1]-H0*Ecc)/CC[:,2])**2)
    cum=np.concatenate(([0.], cumulative_trapezoid(1/E_all,z_all)))
    fac=C_KMS/H0/rd; c2bao=0.
    for bl in BAO_BLOCKS:
        z=bl['z']; Eh=np.interp(z,z_all,E_all)
        DM=fac*np.interp(z,z_all,cum); DH=fac/Eh
        DV=(z*DM*DM*DH)**(1/3)
        mods=np.array([{'DM':DM,'DH':DH,'DV':DV}[k] for k in bl['kinds']])
        dv=bl['vals']-mods; c2bao+=dv@bl['icov']@dv
    Istar=trapezoid(1/E_all,z_all)
    R_th,lA_th=np.sqrt(Om)*Istar, np.pi*(C_KMS/H0)*Istar/rstar
    dv=np.array([R_th,lA_th,ob])-CMB_MEAN
    c2cmb=dv@CMB_ICOV@dv
    cum_lo=np.interp(Z_LOW,z_all,cum); E_lo=np.interp(Z_LOW,z_all,E_all)
    DMsn=(C_KMS/H0)*np.interp(SN_Z,Z_LOW,cum_lo)
    mu_th=5*np.log10((1+SN_Z)*DMsn)+25
    r=SN_MB-mu_th
    c2sn=r@C_SN_inv@r-(r@Binv_MB)**2/Ainv_MB
    Om_z=Om*(1+Z_LOW)**3/E_lo**2
    f_z=Om_z**GAMMA
    cf=np.concatenate(([0.],cumulative_trapezoid(f_z/(1+Z_LOW),Z_LOW)))
    fs8=s8*f_z*np.exp(-cf)
    c2fs8=np.sum(((FS8_V-np.interp(FS8_Z,Z_LOW,fs8))/FS8_S)**2)
    return c2cc+c2bao+c2cmb+c2sn+c2fs8

resL = minimize(chi2_LCDM, [0.3,68,0.02236,0.80], method='Nelder-Mead',
                options={'xatol':1e-5,'fatol':1e-3,'maxiter':5000,'adaptive':True})
chi2_L = resL.fun
print(f"chi2(LCDM) = {chi2_L:.2f}")
print(f"Delta chi2 (alpha=0 best -- LCDM best) = consistency check")

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axs[0]

dchi2 = chi2_prof - chi2_L
# Punto di minimo
i_best = np.argmin(dchi2)
a_best_val = alphas[i_best]
dchi2_best = dchi2[i_best]

ax.plot(alphas, dchi2, 'o-', color='steelblue', ms=5, lw=1.5, label=r'$\chi^2_{\mathrm{prof}}(\alpha)-\chi^2_{\Lambda\mathrm{CDM}}$')
ax.axhline(0, color='k', lw=1, ls=':', label=r'$\Lambda$CDM ($\alpha=0$)')

# bande di significatività (misurate rispetto al MINIMO, non rispetto a zero)
for n_sig, color, label in [(1, 'orange', r'$1\sigma$: $\Delta\chi^2=+1$ dal minimo'),
                              (4, 'red',    r'$2\sigma$: $\Delta\chi^2=+4$ dal minimo'),
                              (9, 'purple', r'$3\sigma$: $\Delta\chi^2=+9$ dal minimo')]:
    ax.axhline(dchi2_best + n_sig, color=color, lw=1, ls='--', label=label)

# marca punto di minimo
ax.plot(a_best_val, dchi2_best, 'D', ms=10, color='crimson', zorder=5,
        markeredgecolor='white', markeredgewidth=1.5,
        label=fr'best-fit: $\alpha={a_best_val:+.3f}$')

ax.axvline(0, color='k', lw=0.5, alpha=0.5)
ax.set_xlabel(r'$\alpha$', fontsize=12)
ax.set_ylabel(r'$\chi^2_{\mathrm{prof}}(\alpha) - \chi^2_{\Lambda\mathrm{CDM}}$', fontsize=12)
ax.set_title(r'Profile likelihood a 1 parametro: $\chi^2(\alpha)$ minimizzato sugli altri 4', fontsize=11)
ax.legend(fontsize=9, loc='upper left', framealpha=0.95)
ax.grid(alpha=0.3)
# Y-range leggibile: mostriamo 3-4 sigma attorno al minimo
ax.set_ylim(dchi2_best - 2, dchi2_best + 15)
ax.set_xlim(-0.07, 0.42)   # non serve andare fino a 0.8

# Pannello destro: solo i 3 parametri rilevanti (Om, H0, sigma8)
# eta e' FISSO a sqrt(6) quindi NON lo mostriamo
ax = axs[1]
# Tre righe separate in tre subplot condivisi (twinx) per vederle tutte
ax_Om = ax
ax_H0 = ax.twinx()
ax_s8 = ax.twinx()

# Sposto il terzo asse
ax_s8.spines['right'].set_position(('outward', 50))

# Om0 in rosso
l1, = ax_Om.plot(alphas, params_prof[:,2], 'o-', color='#c1272d', ms=4, lw=1.5, label=r'$\Omega_{m,0}$')
ax_Om.set_ylabel(r'$\Omega_{m,0}$', color='#c1272d', fontsize=11)
ax_Om.tick_params(axis='y', labelcolor='#c1272d')
ax_Om.set_ylim(0.26, 0.36)

# H0 in verde
l2, = ax_H0.plot(alphas, params_prof[:,3], 's-', color='#2ca02c', ms=4, lw=1.5, label=r'$H_0$')
ax_H0.set_ylabel(r'$H_0$ [km/s/Mpc]', color='#2ca02c', fontsize=11)
ax_H0.tick_params(axis='y', labelcolor='#2ca02c')
ax_H0.set_ylim(62, 72)

# sigma8 in blu
l3, = ax_s8.plot(alphas, params_prof[:,5], '^-', color='#1f77b4', ms=4, lw=1.5, label=r'$\sigma_8$')
ax_s8.set_ylabel(r'$\sigma_8$', color='#1f77b4', fontsize=11)
ax_s8.tick_params(axis='y', labelcolor='#1f77b4')
ax_s8.set_ylim(0.77, 0.82)

ax_Om.axvline(0, color='k', lw=0.5, alpha=0.5)
ax_Om.axvline(a_best_val, color='crimson', lw=1, ls=':', alpha=0.6)
ax_Om.set_xlabel(r'$\alpha$', fontsize=12)
ax_Om.set_xlim(-0.07, 0.42)
ax_Om.set_title(r'Valori ottimali di $\Omega_m, H_0, \sigma_8$ a ciascun $\alpha$ (a $\eta=\sqrt{6}$)', fontsize=11)

# Legenda unica
lines = [l1, l2, l3]
labels = [l.get_label() for l in lines]
ax_Om.legend(lines, labels, loc='upper center', fontsize=9, framealpha=0.95)
ax_Om.grid(alpha=0.3)

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "07_profile.png"), dpi=150, bbox_inches='tight')
print(f"\nPlot salvato in {out_dir}")

print(f"\nDelta chi2 significativita' di alpha!=0: sqrt({chi2_L-chi2_min:.2f}) ~ "
      f"{np.sqrt(max(chi2_L-chi2_min,0)):.2f} sigma (1-par profile)")
