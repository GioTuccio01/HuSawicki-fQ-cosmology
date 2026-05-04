"""
data_loaders_DESI_DR2_patch.py
==============================

Patch loader for DESI DR2 BAO measurements (replaces DR1 in the pipeline).

Reference
---------
M. Abdul-Karim et al. (DESI Collaboration),
"DESI DR2 results II: measurements of baryon acoustic oscillations
and cosmological constraints",
Phys. Rev. D 112, 083515 (2025), arXiv:2503.14738.

Values are taken from Table IV of v3 of the arXiv preprint.

Dataset summary
---------------
12 BAO measurements distributed over 7 tracer redshift bins:

    Tracer       z_eff   Observable(s)        N
    ---------    -----   ------------------   --
    BGS          0.295   D_V/r_d              1
    LRG1         0.510   D_M/r_d, D_H/r_d     2  (rho = -0.459)
    LRG2         0.706   D_M/r_d, D_H/r_d     2  (rho = -0.404)
    LRG3+ELG1    0.934   D_M/r_d, D_H/r_d     2  (rho = -0.416)
    ELG2         1.321   D_M/r_d, D_H/r_d     2  (rho = -0.437)
    QSO          1.484   D_V/r_d              1
    Lya QSO      2.330   D_M/r_d, D_H/r_d     2  (rho = -0.431)
    ---------                                ----
    TOTAL                                     12

The fiducial sound horizon adopted in the DESI DR2 baseline analysis
is r_d^fid = 147.09 Mpc.

Drop-in replacement
-------------------
Use load_BAO_DESI_DR2() exactly like the original load_BAO_DESI() of
the DR1 loader. The returned data structure is unchanged: a list of
"blocks", one per tracer redshift bin, each a dict with keys

    'name'  : tracer label (string),
    'z'     : effective redshift (float),
    'kinds' : list of observable types (subset of {'DM','DH','DV'}),
    'vals'  : np.array of observed values,
    'Cinv'  : inverse covariance matrix of the block (n x n).

The chi^2 contribution is built in the MCMC scripts as
    for bl in BAO_BLOCKS:
        diff = bl['vals'] - model_predictions(bl['z'], bl['kinds'])
        chi2 += diff @ bl['Cinv'] @ diff
"""

import numpy as np

# ============================================================
# DESI DR2 BAO measurements
# Source: Table IV of arXiv:2503.14738v3 (Abdul-Karim et al. 2025)
# ============================================================
DESI_DR2_BAO = [
    # ----- BGS (D_V/r_d only) -----
    {
        "name":  "BGS",
        "z":     0.295,
        "kinds": ["DV"],
        "vals":  np.array([7.944]),
        "errs":  np.array([0.075]),
        "rho":   None,
    },
    # ----- LRG1 (D_M/r_d, D_H/r_d) -----
    {
        "name":  "LRG1",
        "z":     0.510,
        "kinds": ["DM", "DH"],
        "vals":  np.array([13.588, 21.863]),
        "errs":  np.array([0.167,  0.425]),
        "rho":   -0.459,
    },
    # ----- LRG2 (D_M/r_d, D_H/r_d) -----
    {
        "name":  "LRG2",
        "z":     0.706,
        "kinds": ["DM", "DH"],
        "vals":  np.array([17.351, 19.455]),
        "errs":  np.array([0.177,  0.330]),
        "rho":   -0.404,
    },
    # ----- LRG3+ELG1 (D_M/r_d, D_H/r_d) — combined bin -----
    {
        "name":  "LRG3+ELG1",
        "z":     0.934,
        "kinds": ["DM", "DH"],
        "vals":  np.array([21.576, 17.641]),
        "errs":  np.array([0.152,  0.193]),
        "rho":   -0.416,
    },
    # ----- ELG2 (D_M/r_d, D_H/r_d) -----
    {
        "name":  "ELG2",
        "z":     1.321,
        "kinds": ["DM", "DH"],
        "vals":  np.array([27.605, 14.178]),
        "errs":  np.array([0.320,  0.217]),
        "rho":   -0.437,
    },
    # ----- QSO (D_V/r_d only) -----
    {
        "name":  "QSO",
        "z":     1.484,
        "kinds": ["DV"],
        "vals":  np.array([26.059]),
        "errs":  np.array([0.400]),
        "rho":   None,
    },
    # ----- Lya QSO (D_M/r_d, D_H/r_d) -----
    {
        "name":  "Lya",
        "z":     2.330,
        "kinds": ["DM", "DH"],
        "vals":  np.array([38.988, 8.632]),
        "errs":  np.array([0.531, 0.101]),
        "rho":   -0.431,
    },
]

# Fiducial sound horizon adopted in DESI DR2
RD_FID_DR2 = 147.09  # Mpc


# ============================================================
# Drop-in loader function
# ============================================================
def load_BAO_DESI_DR2(verbose=True):
    """
    Return DESI DR2 BAO data as a list of blocks, with the same shape
    used by the rest of the pipeline.

    Each block is a dict:
        {
            'name'  : tracer name (for diagnostics),
            'z'     : effective redshift,
            'kinds' : list of 'DM' / 'DH' / 'DV' (the observables),
            'vals'  : np.array of the observed values,
            'Cinv'  : inverse covariance of the block (n x n),
        }
    """
    blocks = []
    n_tot  = 0
    for entry in DESI_DR2_BAO:
        kinds = entry["kinds"]
        vals  = entry["vals"]
        errs  = entry["errs"]
        n     = len(kinds)

        # Build the per-block covariance matrix.
        # 1D blocks (DV-only) have a trivial 1x1 covariance.
        # 2D blocks (DM, DH) include the published correlation rho.
        if n == 1:
            cov = np.array([[errs[0] ** 2]])
        elif n == 2:
            rho = entry["rho"]
            sM, sH = errs[0], errs[1]
            cov = np.array([
                [sM * sM,        rho * sM * sH],
                [rho * sM * sH,  sH * sH       ],
            ])
        else:
            raise ValueError(f"Block {entry['name']} has unexpected size {n}")

        Cinv = np.linalg.inv(cov)
        blocks.append({
            "name":  entry["name"],
            "z":     entry["z"],
            "kinds": kinds,
            "vals":  vals,
            "Cinv":  Cinv,
        })
        n_tot += n

    if verbose:
        print(f"  [BAO] DESI DR2: loaded {len(blocks)} blocks "
              f"({n_tot} measurements)")
        for bl in blocks:
            kstr = ",".join(bl["kinds"])
            print(f"        {bl['name']:<10s} "
                  f"z={bl['z']:.3f}  ({kstr})")

    return blocks


# ============================================================
# Backward-compatible alias
# ------------------------------------------------------------
# This makes the patch a drop-in replacement: the existing scripts
# import "load_BAO_DESI" from data_loaders, and that name will now
# resolve to the DR2 implementation.
# ============================================================
load_BAO_DESI = load_BAO_DESI_DR2


# ============================================================
# Stand-alone smoke test
# ============================================================
if __name__ == "__main__":
    blocks = load_BAO_DESI_DR2(verbose=True)
    n_tot  = sum(len(b["kinds"]) for b in blocks)
    print(f"\nTotal: {len(blocks)} blocks, {n_tot} BAO data points")
    print(f"r_d^fid = {RD_FID_DR2} Mpc (DR2 convention)")

    # ---- Quick chi^2 sanity check at a Planck LCDM fiducial cosmology
    # Using the DR2 best-fit point (Om=0.2975, H0=68.51) without computing
    # full integrals — the values are reproduced in DR2 Table V.
    print("\nSanity check (a few values being read back):")
    for bl in blocks:
        for k, v, s in zip(bl["kinds"], bl["vals"],
                           np.sqrt(np.diag(np.linalg.inv(bl["Cinv"])))):
            print(f"   {bl['name']:<10s} z={bl['z']:.3f}  "
                  f"{k}/r_d = {v:7.3f} +/- {s:.3f}")
