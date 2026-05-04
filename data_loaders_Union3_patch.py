"""
data_loaders_Union3_patch.py — v3 SEMPLICE
==========================================
Legge i due file Union3 dal repo CobayaSampler/sn_data/Union3:
  - lcparam_full.txt   (22 righe + header con z e mb)
  - mag_covmat.txt     (prima riga = 22, poi 484 valori = covarianza 22x22)

Output: (z, mu, C_inv, A_inv_MB, B_inv_MB)
firma compatibile con load_PantheonPlus / load_DESY5 / etc.
"""
import os
import numpy as np
import scipy.linalg as sla


def _find(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, 'data', filename),
        os.path.join(here, '..', 'data', filename),
        os.path.join(here, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_Union3(verbose=True):
    """
    Carica Union3 binned (22 punti) dai file CobayaSampler.

    Returns:  z, mu, C_inv, A_inv_MB, B_inv_MB
    """
    lcpath = _find('lcparam_full.txt')
    cvpath = _find('mag_covmat.txt')
    if lcpath is None or cvpath is None:
        raise FileNotFoundError(
            "Servono lcparam_full.txt e mag_covmat.txt in data/ "
            "(scaricabili da github.com/CobayaSampler/sn_data/tree/master/Union3)")

    # --- lcparam: skip header line, prendo colonna 1 (zcmb) e colonna 4 (mb) ---
    # Formato:  #name zcmb zhel dz mb dmb x1 dx1 color dcolor 3rdvar d3rdvar ...
    z, mu = [], []
    with open(lcpath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.split()
            z.append(float(cols[1]))   # zcmb
            mu.append(float(cols[4]))  # mb (distance modulus)
    z = np.array(z)
    mu = np.array(mu)
    N = len(z)

    # --- covarianza: prima riga = N, poi N*N elementi (uno per riga) ---
    with open(cvpath) as f:
        first = f.readline().strip()
        nsz = int(first)
        if nsz != N:
            raise ValueError(f"mag_covmat.txt header dice {nsz}, lcparam ha {N}")
        rest = f.read().split()
        flat = np.array([float(x) for x in rest])
    if flat.size != N*N:
        raise ValueError(f"mag_covmat: attesi {N*N} elementi, trovati {flat.size}")
    C = flat.reshape(N, N)
    C = 0.5 * (C + C.T)

    # Inverto via Cholesky
    try:
        L, low = sla.cho_factor(C, lower=True)
    except sla.LinAlgError as e:
        raise RuntimeError(f"Covarianza Union3 non def-pos: {e}")
    Cinv = sla.cho_solve((L, low), np.eye(N))
    ones = np.ones(N)
    Binv = sla.cho_solve((L, low), ones)
    Ainv = float(ones @ Binv)

    if verbose:
        print(f"  load_Union3: {N} bin")
        print(f"    z range  : [{z.min():.4f}, {z.max():.4f}]")
        print(f"    mu range : [{mu.min():.2f}, {mu.max():.2f}]")
        print(f"    sqrt(diag(C)) typical: {np.sqrt(np.median(np.diag(C))):.4f} mag")
        print(f"    Ainv_MB = {Ainv:.4f}")

    return z, mu, Cinv, Ainv, Binv


if __name__ == "__main__":
    z, mu, Cinv, A, B = load_Union3()
    print(f"\nFirst 3 (z, mu): {list(zip(z[:3], mu[:3]))}")
    eigs = np.linalg.eigvalsh(Cinv)
    print(f"eig(Cinv) min={eigs.min():.4e}, max={eigs.max():.4e}")
