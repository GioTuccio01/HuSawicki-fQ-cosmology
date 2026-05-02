"""
07b_profile_2D.py — Profile likelihood 2D nel piano (alpha, Omega_m).

Per ogni punto (alpha_i, Om_j) della griglia, minimizza il chi^2 sui 3 parametri
rimanenti (H_0, omega_b, sigma_8) con eta = sqrt(6) fissato.

Restituisce la mappa chi^2_prof(alpha, Om) e la visualizza in DUE pannelli:
  - Sinistra: colormap + isolivelli 1sigma, 2sigma, 3sigma (Wilks 2D: Δχ²= 2.30, 6.18, 11.83)
  - Destra:   rendering 3D della superficie chi^2_prof

Tempo stimato: ~10-20 minuti per griglia 18x18.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from time import time
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import (load_CC, load_BAO_DESI, load_CMB_Planck,
                          load_PantheonPlus, load_fsigma8)

print("="*70)
print("PROFILE LIKELIHOOD 2D  chi^2_prof(alpha, Omega_m)")
print("Per ogni (alpha, Om): minimizza su (H0, omega_b, sigma8)")
print("eta = sqrt(6) fissato")
print("="*70)

# --------- Dati (stessi di 04) ---------
CC = load_CC()
BAO_BLOCKS = load_BAO_DESI()
CMB_MEAN, CMB_ICOV = load_CMB_Planck()
SN_Z, SN_MB, C_SN_inv, Ainv_MB, Binv_MB = load_PantheonPlus()
FS8 = load_fsigma8()
FS8_Z, FS8_V, FS8_S = FS8[:, 0], FS8[:, 1], FS8[:, 2]

# --------- Setup modello (conv A) ---------
C_KMS = 299792.458; OM_R = 9.2e-5; Z_STAR = 1089.80; GAMMA = 0.55
ETA_FIX = np.sqrt(6.0); X0_FIX = 1.0
X_SAMP = np.unique(np.concatenate([
    np.linspace(1e-5, 0.3, 100),
    np.linspace(0.3, 50, 300),
    np.linspace(50, 15000, 80)]))
Z_LOW = np.linspace(1e-5, 15.0, 300)
Z_HIGH = np.linspace(15.0, Z_STAR, 400)


def Lam_of(a, Om0):
    return 0.5*((1.0 - Om0)*X0_FIX + a*(1 + 6*X0_FIX - 3*X0_FIX*X0_FIX)/(1 + X0_FIX)**3)


def solve_E(z_arr, alpha, Om0):
    Lam = Lam_of(alpha, Om0)
    R_s = 2*Lam - alpha*(1 + 6*X_SAMP - 3*X_SAMP**2)/(1 + X_SAMP)**3
    g_s = X_SAMP - R_s
    if not np.all(np.diff(g_s) > 0):
        return None, None, None
    tgt = Om0*X0_FIX*(1 + np.asarray(z_arr, float))**3
    xs = np.interp(tgt, g_s, X_SAMP)
    return np.sqrt(xs/X0_FIX), xs, Lam


def rd_Aub(Om, H0, ob):
    h = H0/100
    return 55.154/((Om*h*h)**0.25351 * ob**0.12807)


def rs_ratio(Om, H0, ob):
    h = H0/100
    return 1.0 - 0.0206*ob**0.165 * (Om*h*h)**0.05


def chi2_full(alpha, Om, H0, ob, s8):
    """chi2 completo CC+BAO+CMB+SN+fs8 (identico a 04)."""
    rd = rd_Aub(Om, H0, ob); rstar = rd*rs_ratio(Om, H0, ob)
    E_lo, xs, Lam = solve_E(Z_LOW, alpha, Om)
    if E_lo is None: return np.inf
    cum = np.concatenate(([0.0], cumulative_trapezoid(1/E_lo, Z_LOW)))

    # CC
    Ecc = np.interp(CC[:, 0], Z_LOW, E_lo)
    c2cc = np.sum(((CC[:, 1] - H0*Ecc)/CC[:, 2])**2)

    # BAO
    fac = C_KMS/H0/rd; c2bao = 0.
    for bl in BAO_BLOCKS:
        z = bl['z']; Eh = np.interp(z, Z_LOW, E_lo)
        DM = fac*np.interp(z, Z_LOW, cum); DH = fac/Eh
        DV = (z*DM*DM*DH)**(1/3)
        mods = np.array([{'DM': DM, 'DH': DH, 'DV': DV}[k] for k in bl['kinds']])
        dv = bl['vals'] - mods
        c2bao += dv @ bl['icov'] @ dv

    # CMB (high-z extension per Istar)
    E_hi = np.sqrt(Om*(1 + Z_HIGH)**3 + OM_R*(1+Z_HIGH)**4 + 2*Lam/X0_FIX)
    Istar = trapezoid(1/E_lo, Z_LOW) + trapezoid(1/E_hi, Z_HIGH)
    R_th = np.sqrt(Om)*Istar
    lA_th = np.pi*(C_KMS/H0)*Istar/rstar
    dv = np.array([R_th, lA_th, ob]) - CMB_MEAN
    c2cmb = dv @ CMB_ICOV @ dv

    # SN (M marginalizzato)
    DMsn = (C_KMS/H0)*np.interp(SN_Z, Z_LOW, cum)
    mu_th = 5*np.log10((1 + SN_Z)*DMsn) + 25
    r = SN_MB - mu_th
    c2sn = r @ C_SN_inv @ r - (r @ Binv_MB)**2/Ainv_MB

    # fsigma8
    Om_z = Om*(1 + Z_LOW)**3/E_lo**2
    f_z = Om_z**GAMMA
    cf = np.concatenate(([0.0], cumulative_trapezoid(f_z/(1 + Z_LOW), Z_LOW)))
    fs8_full = s8*f_z*np.exp(-cf)
    c2fs8 = np.sum(((FS8_V - np.interp(FS8_Z, Z_LOW, fs8_full))/FS8_S)**2)

    return c2cc + c2bao + c2cmb + c2sn + c2fs8


# --------- Griglia 2D ---------
# Uso range ristretto centrato sul best-fit di Plot 04 (+-3sigma)
#   alpha = +0.159 +/- 0.063 --> [0.0, 0.35]
#   Om    = +0.308 +/- 0.007 --> [0.28, 0.33]
NA, NO = 18, 14    # 18*14 = 252 punti
alpha_grid = np.linspace(0.00, 0.35, NA)
Om_grid    = np.linspace(0.285, 0.330, NO)

print(f"Griglia {NA}x{NO} = {NA*NO} punti")
print(f"alpha in [{alpha_grid[0]:.3f}, {alpha_grid[-1]:.3f}]")
print(f"Om_m in [{Om_grid[0]:.3f}, {Om_grid[-1]:.3f}]")

chi2_map = np.full((NA, NO), np.nan)

# Starting point (best-fit di 04)
start = np.array([67.08, 0.02245, 0.8015])  # H0, omb, s8

print("\nScan in corso...")
t0 = time()
N_done = 0

# Scan serpentino per sfruttare il warm-start
for i, a in enumerate(alpha_grid):
    for j, Om in enumerate(Om_grid):
        def f_of(x):
            H0, ob, s8 = x
            if not (55 < H0 < 85 and 0.018 < ob < 0.026 and 0.5 < s8 < 1.1):
                return 1e10
            return chi2_full(a, Om, H0, ob, s8)
        r = minimize(f_of, start, method='Nelder-Mead',
                     options={'xatol': 1e-4, 'fatol': 1e-2, 'maxiter': 1500})
        chi2_map[i, j] = r.fun
        start = r.x  # warm start
        N_done += 1
        if N_done % 20 == 0:
            frac = N_done/(NA*NO)
            eta_s = (time() - t0)/frac - (time() - t0)
            print(f"  {N_done}/{NA*NO}  ({100*frac:.0f}%)  eta {eta_s/60:.1f} min")

print(f"\nCompletato in {(time()-t0)/60:.1f} min")

# --------- Salvataggio ---------
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(OUT_DIR, exist_ok=True)
np.savez(os.path.join(OUT_DIR, "07b_profile2D.npz"),
         alpha_grid=alpha_grid, Om_grid=Om_grid, chi2_map=chi2_map)

# --------- Post-processing ---------
# Smussamento leggero per isolivelli puliti
chi2_smooth = gaussian_filter(chi2_map, sigma=0.6)
chi2_min = np.nanmin(chi2_smooth)
idx_min = np.unravel_index(np.argmin(chi2_smooth), chi2_smooth.shape)
a_best = alpha_grid[idx_min[0]]
Om_best = Om_grid[idx_min[1]]
dchi2 = chi2_smooth - chi2_min

print(f"\nchi2_min  = {chi2_min:.2f}")
print(f"  a_best  = {a_best:+.3f}")
print(f"  Om_best = {Om_best:.4f}")

# Wilks 2D: Delta chi^2 = 2.30 (1σ 68%), 6.18 (2σ 95%), 11.83 (3σ 99.7%)
levels_2D = [2.30, 6.18, 11.83]
level_labels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$']

# --------- PLOT: 2 figure separate (2D + 3D) ---------
# === Figure A: 2D colormap + isocontours ===
fig_a = plt.figure(figsize=(8, 6))
ax1 = fig_a.add_subplot(1, 1, 1)
AG, OG = np.meshgrid(alpha_grid, Om_grid, indexing='ij')

im = ax1.pcolormesh(AG, OG, np.clip(dchi2, 0, 15),
                    cmap='viridis_r', shading='auto', alpha=0.9)
cb = fig_a.colorbar(im, ax=ax1, label=r'$\Delta\chi^2 = \chi^2_{\rm prof} - \chi^2_{\min}$')

cs = ax1.contour(AG, OG, dchi2, levels=levels_2D,
                 colors=['white', 'orange', 'red'],
                 linewidths=[2.0, 2.0, 2.0],
                 linestyles=['-', '--', ':'])
fmt = {lev: lab for lev, lab in zip(levels_2D, level_labels)}
ax1.clabel(cs, inline=True, fontsize=10, fmt=fmt)

ax1.plot(a_best, Om_best, '*', ms=20, color='red',
         markeredgecolor='white', markeredgewidth=1.5,
         zorder=10, label=fr'best-fit $({a_best:+.3f}, {Om_best:.3f})$')
ax1.axvline(0, color='cyan', lw=1.5, alpha=0.7, ls='--',
            label=r'$\Lambda$CDM ($\alpha=0$)')

ax1.set_xlabel(r'$\alpha$', fontsize=13)
ax1.set_ylabel(r'$\Omega_{m,0}$', fontsize=13)
ax1.set_title(r'2D profile map of $\chi^2_{\rm prof}(\alpha, \Omega_m)$'
              '\n'
              r'profiled over $(H_0, \omega_b, \sigma_8)$',
              fontsize=11)
ax1.legend(fontsize=9, loc='lower right', framealpha=0.92)

plt.tight_layout()
fig_a.savefig(os.path.join(OUT_DIR, "07b_profile_2D_a_map.png"), dpi=150, bbox_inches='tight')
plt.close(fig_a)

# === Figure B: 3D surface ===
fig_b = plt.figure(figsize=(8, 6))
ax2 = fig_b.add_subplot(1, 1, 1, projection='3d')

Z_surf = np.clip(dchi2, 0, 20)
surf = ax2.plot_surface(AG, OG, Z_surf, cmap='viridis_r',
                        linewidth=0.3, alpha=0.92, edgecolor='k',
                        rstride=1, cstride=1, antialiased=True)
ax2.scatter([a_best], [Om_best], [0], color='red', s=200,
            marker='*', edgecolor='white', linewidth=1.5, zorder=10)
ax2.contour(AG, OG, dchi2, levels=levels_2D,
            zdir='z', offset=0, colors=['white', 'orange', 'red'],
            linewidths=1.5, linestyles=['-', '--', ':'])

ax2.set_xlabel(r'$\alpha$', fontsize=11, labelpad=8)
ax2.set_ylabel(r'$\Omega_{m,0}$', fontsize=11, labelpad=8)
ax2.set_zlabel(r'$\Delta\chi^2$', fontsize=11, labelpad=6)
ax2.set_title(r'3D surface of $\chi^2_{\rm prof}(\alpha, \Omega_m)$', fontsize=11)
ax2.view_init(elev=25, azim=-65)

plt.tight_layout()
fig_b.savefig(os.path.join(OUT_DIR, "07b_profile_2D_b_surface.png"), dpi=150, bbox_inches='tight')
plt.close(fig_b)
print(f"\n2 plots saved in {OUT_DIR}: 07b_profile_2D_a_map.png, 07b_profile_2D_b_surface.png")

# --------- Print sigma-intervals per alpha marginalizzato (minimo su Om) ---------
# Per confronto con Plot 07 1D
chi2_prof1d_alpha = np.min(chi2_smooth, axis=1)
d1 = chi2_prof1d_alpha - chi2_prof1d_alpha.min()
# 1sigma 1D: Delta chi2 <= 1
mask_1s = d1 <= 1
if mask_1s.any():
    a_lo = alpha_grid[mask_1s].min()
    a_hi = alpha_grid[mask_1s].max()
    print(f"\n1-param 1sigma interval (profilato anche su Om):")
    print(f"  alpha in [{a_lo:+.3f}, {a_hi:+.3f}]")
    print(f"  confronto con Plot 07 1D: alpha = +0.152 +0.06/-0.06")

print("\n=== FINE 07b ===")
