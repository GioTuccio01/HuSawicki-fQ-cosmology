"""
01_background_CC_BAO.py — Background phenomenology of f(Q) Hu-Sawicki vs CC + BAO DESI DR1
[VERSIONE A FIGURE SINGOLE: 4 PNG separati invece di un 2x2]
- f(Q) = Q + 2 Lambda + alpha*mu^2 (x-1)/(1+x)^2
- Background numerico (no perturbazioni); Lambda fissato per Omega_DE,0 = 0.7 a x_0
- Output: 01_background_a_wDE.png, 01_background_b_Hz.png,
          01_background_c1_BAO.png, 01_background_c2_chi2.png

Convenzione A: alpha > 0 e' il regime DESI-favorito.
"""
import os, numpy as np, matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_HERE, "..", "plots")
os.makedirs(_OUT_DIR, exist_ok=True)

# =========================================================================
# 1.  COSTANTI E DATI
# =========================================================================
C_KMS = 299792.458

# Cosmic chronometers
CC = np.array([
    [0.0700,  69.0, 19.6], [0.0900,  69.0, 12.0], [0.1200,  68.6, 26.2],
    [0.1700,  83.0,  8.0], [0.1791,  75.0,  4.0], [0.1993,  75.0,  5.0],
    [0.2000,  72.9, 29.6], [0.2700,  77.0, 14.0], [0.2800,  88.8, 36.6],
    [0.3519,  83.0, 14.0], [0.3802,  83.0, 13.5], [0.4000,  95.0, 17.0],
    [0.4004,  77.0, 10.2], [0.4247,  87.1, 11.2], [0.4497,  92.8, 12.9],
    [0.4700,  89.0, 49.6], [0.4783,  80.9,  9.0], [0.4800,  97.0, 62.0],
    [0.5930, 104.0, 13.0], [0.6800,  92.0,  8.0], [0.7500,  98.8, 33.6],
    [0.7810, 105.0, 12.0], [0.8750, 125.0, 17.0], [0.8800,  90.0, 40.0],
    [0.9000, 117.0, 23.0], [1.0370, 154.0, 20.0], [1.3000, 168.0, 17.0],
    [1.3630, 160.0, 33.6], [1.4300, 177.0, 18.0], [1.5300, 140.0, 14.0],
    [1.7500, 202.0, 40.0], [1.9650, 186.5, 50.4],
])

# BAO DESI DR1
BAO = [
    (0.295,  7.93, 0.15, "DV"), (0.510, 13.62, 0.25, "DM"),
    (0.510, 20.98, 0.61, "DH"), (0.706, 16.85, 0.32, "DM"),
    (0.706, 20.08, 0.60, "DH"), (0.930, 21.71, 0.28, "DM"),
    (0.930, 17.88, 0.35, "DH"), (1.317, 27.79, 0.69, "DM"),
    (1.317, 13.82, 0.42, "DH"), (1.491, 26.07, 0.67, "DV"),
    (2.330, 39.71, 0.94, "DM"), (2.330,  8.52, 0.17, "DH"),
]

H0_FID = 67.4
RD_FID = 147.05

# =========================================================================
# 2.  BACKGROUND SOLVER (numerico)
# =========================================================================
def x0_from_eta(eta): return 6.0/eta**2
def Lam_canon(alpha, eta):
    x0 = x0_from_eta(eta)
    return 0.5*(0.7*x0 + alpha*(1+6*x0-3*x0*x0)/(1+x0)**3)

def Ez(z_arr, alpha, eta):
    x0 = x0_from_eta(eta)
    Lam = Lam_canon(alpha, eta)
    x_grid = np.unique(np.concatenate([
        np.linspace(1e-5, 0.4, 80), np.linspace(0.4, 80, 300),
        np.linspace(80, 15000, 80)]))
    R_x = 2*Lam - alpha*(1+6*x_grid-3*x_grid**2)/(1+x_grid)**3
    g_x = x_grid - R_x
    if not np.all(np.diff(g_x) > 0):
        return None, None, Lam
    target = 0.3*x0*(1+np.asarray(z_arr, float))**3
    xs = np.interp(target, g_x, x_grid)
    return np.sqrt(xs/x0), xs, Lam

def D_of_x(x, alpha):
    """D(x) = f_Q + 2 Q f_QQ = 1 + 3 alpha (x^2-6x+1)/(1+x)^4"""
    return 1.0 + 3*alpha*(x**2 - 6*x + 1)/(1+x)**4

def w_DE(x, alpha, Lam):
    """w_DE(x) = -(x + 4 Hdot/mu^2)/R, con 4 Hdot/mu^2 = -(x-R)/D(x).
    Crossing strutturale a R'(x)=0, cioè x = 3 +- 2 sqrt(2)."""
    R = 2*Lam - alpha*(1+6*x-3*x**2)/(1+x)**3
    fourHdot = -(x - R)/D_of_x(x, alpha)
    return -(x + fourHdot)/R

def z_of_x(x, alpha, eta):
    x0 = x0_from_eta(eta); Lam = Lam_canon(alpha, eta)
    R0  = 2*Lam - alpha*(1+6*x0-3*x0*x0)/(1+x0)**3
    Rx  = 2*Lam - alpha*(1+6*x-3*x*x)/(1+x)**3
    arg = (x - Rx)/(0.3*x0)
    if arg <= 0: return np.nan
    return arg**(1/3) - 1

# =========================================================================
# 3.  OBSERVABLES
# =========================================================================
def _grids(alpha, eta, zmax=3.5, n=300):
    z = np.linspace(1e-5, zmax, n)
    E, _, _ = Ez(z, alpha, eta)
    return z, E

def DM_over_rd(z_target, alpha, eta, H0, rd, zg=None):
    if zg is None: zg = _grids(alpha, eta)
    z, E = zg
    cum = np.concatenate(([0.0], cumulative_trapezoid(1/E, z)))
    DM = (C_KMS/H0)*np.interp(z_target, z, cum)
    return DM/rd

def DH_over_rd(z_target, alpha, eta, H0, rd, zg=None):
    if zg is None: zg = _grids(alpha, eta)
    z, E = zg
    Ez_at = np.interp(z_target, z, E)
    return (C_KMS/(H0*Ez_at))/rd

def DV_over_rd(z_target, alpha, eta, H0, rd, zg=None):
    Dm = DM_over_rd(z_target, alpha, eta, H0, rd, zg)*rd
    Dh = DH_over_rd(z_target, alpha, eta, H0, rd, zg)*rd
    return ((z_target * Dm**2 * Dh)**(1/3))/rd

# =========================================================================
# 4.  CHI^2
# =========================================================================
def chi2_CC(alpha, eta, H0):
    E, _, _ = Ez(CC[:,0], alpha, eta)
    if E is None: return np.inf
    return float(np.sum(((CC[:,1] - H0*E)/CC[:,2])**2))

def chi2_BAO(alpha, eta, H0, rd):
    zg = _grids(alpha, eta, zmax=2.6)
    if zg[1] is None: return np.inf
    c2 = 0.0
    for (z, obs, sig, kind) in BAO:
        if kind=="DM":   pred = DM_over_rd(z, alpha, eta, H0, rd, zg)
        elif kind=="DH": pred = DH_over_rd(z, alpha, eta, H0, rd, zg)
        else:            pred = DV_over_rd(z, alpha, eta, H0, rd, zg)
        c2 += ((obs-pred)/sig)**2
    return c2

# =========================================================================
# 5.  FIGURA (a):  w_DE(z) varying eta, alpha fissato
# =========================================================================
fig_a, ax_wz = plt.subplots(figsize=(7.5, 5.2))
alpha_a = 0.15
z_a     = np.linspace(0.0, 2.5, 500)
etas    = [1.5, 2.0, np.sqrt(6.0)]
cols_a  = ['#2e4a8f', '#d1495b', '#2ca02c']
for et, cc in zip(etas, cols_a):
    E, xs, Lam = Ez(z_a, alpha_a, et)
    w = w_DE(xs, alpha_a, Lam)
    ax_wz.plot(z_a, w, lw=2.2, color=cc,
               label=fr'$\eta={et:.2f}$  ($x_0={x0_from_eta(et):.2f}$)')
xp, xm = 3.0+2.0*np.sqrt(2), 3.0-2.0*np.sqrt(2)
for et, cc in zip(etas, cols_a):
    for xc in (xp, xm):
        zc = z_of_x(xc, alpha_a, et)
        if np.isfinite(zc) and 0 < zc < z_a[-1]:
            ax_wz.axvline(zc, ls=':', color=cc, alpha=0.7, lw=1.3)
ax_wz.axhspan(-1, 0, color='orange', alpha=0.07)
ax_wz.axhspan(-2, -1, color='steelblue', alpha=0.07)
ax_wz.text(2.4, -0.92, 'quintessence', fontsize=8.5, color='#b35900',
           ha='right', alpha=0.9, fontweight='bold')
ax_wz.text(2.4, -1.22, 'phantom', fontsize=8.5, color='#1f4e79',
           ha='right', alpha=0.9, fontweight='bold')
ax_wz.axhline(-1, ls='--', color='k', alpha=0.55, lw=1, label=r'$w=-1$')
ax_wz.set(xlabel=r'$z$', ylabel=r'$w_{\rm DE}$',
          title=fr'$w_{{\rm DE}}(z)$ at $\alpha={alpha_a}$, varying $\eta=\mu/H_0$')
ax_wz.set_ylim(-1.25, -0.80)
ax_wz.set_xlim(0, 2.5)
ax_wz.grid(alpha=0.3); ax_wz.legend(fontsize=9, loc='lower right')
plt.tight_layout()
fig_a.savefig(os.path.join(_OUT_DIR, '01_background_a_wDE.png'), dpi=150, bbox_inches='tight')
plt.close(fig_a)

# =========================================================================
# 6.  FIGURA (b):  H(z) + cosmic chronometers
# =========================================================================
fig_b, ax_Hz = plt.subplots(figsize=(7.5, 5.2))
eta_b  = np.sqrt(6.0)
alpha_list = [-0.2, 0.0, 0.2]
cols_b = ['tab:blue', 'k', 'tab:red']
z_b    = np.linspace(1e-3, 2.1, 400)
for al, cc in zip(alpha_list, cols_b):
    E, _, _ = Ez(z_b, al, eta_b)
    ax_Hz.plot(z_b, H0_FID*E, lw=2, color=cc, label=fr'$\alpha={al:+.1f}$')
ax_Hz.errorbar(CC[:,0], CC[:,1], yerr=CC[:,2], fmt='o', ms=4, color='0.35',
               alpha=0.7, label='CC (32 pt)')
ax_Hz.set(xlabel=r'$z$', ylabel=r'$H(z)$  [km s$^{-1}$ Mpc$^{-1}$]',
          title=fr'$H(z)$ + CC   ($\eta=\sqrt{{6}}$, $H_0={H0_FID}$)')
ax_Hz.grid(alpha=0.3); ax_Hz.legend(fontsize=9)
plt.tight_layout()
fig_b.savefig(os.path.join(_OUT_DIR, '01_background_b_Hz.png'), dpi=150, bbox_inches='tight')
plt.close(fig_b)

# =========================================================================
# 7.  FIGURA (c.1):  BAO (D_M/r_d, D_H/r_d) vs DESI DR1
# =========================================================================
fig_c1, ax_dm = plt.subplots(figsize=(7.5, 5.2))
for al, cc in zip(alpha_list, cols_b):
    zg = _grids(al, eta_b, zmax=2.6)
    zc_plot = np.linspace(0.05, 2.45, 300)
    ax_dm.plot(zc_plot, DM_over_rd(zc_plot, al, eta_b, H0_FID, RD_FID, zg),
               lw=2, color=cc, label=fr'$D_M/r_d$, $\alpha={al:+.1f}$')
    ax_dm.plot(zc_plot, DH_over_rd(zc_plot, al, eta_b, H0_FID, RD_FID, zg),
               lw=2, ls='--', color=cc, alpha=0.7,
               label=fr'$D_H/r_d$, $\alpha={al:+.1f}$' if al==0 else None)
for (z, obs, sig, kind) in BAO:
    if kind == "DM":
        ax_dm.errorbar(z, obs, yerr=sig, fmt='o', ms=6, color='crimson',
                       alpha=0.85, zorder=5)
    elif kind == "DH":
        ax_dm.errorbar(z, obs, yerr=sig, fmt='s', ms=6, color='seagreen',
                       alpha=0.85, zorder=5)
    else:
        ax_dm.errorbar(z, obs, yerr=sig, fmt='^', ms=7, color='navy',
                       alpha=0.85, zorder=5)
ax_dm.errorbar([], [], yerr=[], fmt='o', color='crimson',  label=r'DESI $D_M/r_d$')
ax_dm.errorbar([], [], yerr=[], fmt='s', color='seagreen', label=r'DESI $D_H/r_d$')
ax_dm.errorbar([], [], yerr=[], fmt='^', color='navy',     label=r'DESI $D_V/r_d$')
ax_dm.set(xlabel=r'$z_{\rm eff}$', ylabel=r'$D_X/r_d$',
          title=r'BAO: $D_M/r_d$ (—), $D_H/r_d$ (- -) vs DESI DR1')
ax_dm.grid(alpha=0.3); ax_dm.legend(fontsize=8, ncol=2)
plt.tight_layout()
fig_c1.savefig(os.path.join(_OUT_DIR, '01_background_c1_BAO.png'), dpi=150, bbox_inches='tight')
plt.close(fig_c1)

# =========================================================================
# 8.  FIGURA (c.2):  Delta chi^2 vs alpha
# =========================================================================
fig_c2, ax_chi = plt.subplots(figsize=(7.5, 5.2))
alpha_scan = np.linspace(-0.35, 0.35, 15)
cc_arr = np.array([chi2_CC (a, eta_b, H0_FID)         for a in alpha_scan])
bb_arr = np.array([chi2_BAO(a, eta_b, H0_FID, RD_FID) for a in alpha_scan])

ax_chi.plot(alpha_scan, cc_arr - cc_arr.min(), 'o-', lw=1.5, label=r'$\Delta\chi^2_{\rm CC}$')
ax_chi.plot(alpha_scan, bb_arr - bb_arr.min(), 's-', lw=1.5, label=r'$\Delta\chi^2_{\rm BAO}$')
tot = cc_arr + bb_arr
ax_chi.plot(alpha_scan, tot - tot.min(), 'D-', lw=2, color='k', label=r'$\Delta\chi^2_{\rm tot}$')
ax_chi.axvline(0, ls='--', color='gray', alpha=0.6)
ax_chi.axhline(1, ls=':', color='gray', alpha=0.5)
ax_chi.axhline(4, ls=':', color='gray', alpha=0.5)
ax_chi.text(alpha_scan[0], 1.2, r'$1\sigma$', fontsize=8, color='gray')
ax_chi.text(alpha_scan[0], 4.2, r'$2\sigma$', fontsize=8, color='gray')
ax_chi.set(xlabel=r'$\alpha$', ylabel=r'$\Delta\chi^2$',
           title=fr'$\Delta\chi^2(\alpha)$  ($\eta=\sqrt{{6}}$, $H_0,r_d$ fixed)')
ax_chi.grid(alpha=0.3); ax_chi.legend(fontsize=9); ax_chi.set_ylim(-0.3, 14); ax_chi.set_xlim(-0.4, 0.4)
plt.tight_layout()
fig_c2.savefig(os.path.join(_OUT_DIR, '01_background_c2_chi2.png'), dpi=150, bbox_inches='tight')
plt.close(fig_c2)

print(f"4 figures saved in {_OUT_DIR}:")
print(f"  01_background_a_wDE.png")
print(f"  01_background_b_Hz.png")
print(f"  01_background_c1_BAO.png")
print(f"  01_background_c2_chi2.png")

# =========================================================================
# 9.  TABELLE NUMERICHE
# =========================================================================
print("\n=== Phantom crossings in z (roots of x^2-6x+1=0), varying eta, alpha=0.15 ===")
print(f"{'eta':>8} {'x_0':>7} {'z(x_+)':>10} {'z(x_-)':>10}")
for et in etas:
    zp = z_of_x(xp, alpha_a, et)
    zm = z_of_x(xm, alpha_a, et)
    zp_s = f"{zp:+.3f}" if np.isfinite(zp) else "future"
    zm_s = f"{zm:+.3f}" if np.isfinite(zm) else "future"
    print(f"  {et:>6.3f}  {x0_from_eta(et):>5.2f}  {zp_s:>10}  {zm_s:>10}")
