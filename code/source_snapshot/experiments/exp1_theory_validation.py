"""
exp1_theory_validation.py
=========================
Numerical validation of Section IV (no data, no networks):
(a) Pairwise spherical Theorem 1 across a 27-point (w, d0, eta) grid
    spanning the rho=1 transition: d_final -> d_star=sqrt(d0^2-w/eta)
    for rho>1, collapse for rho<=1.
(b) The eta-flip pair (w=0.6, d0=1.0, eta in {2.0, 0.5}) isolating eta.
(c) Gram-matrix PSD boundary; (d) auxiliary Euclidean contraction.
Run: python -m experiments.exp1_theory_validation
"""
from __future__ import annotations
import json, os
import numpy as np

from sml.reduced_dynamics import (simulate_spherical_pair, theorem_equilibrium,
                                  validate_all)

def main():
    rng = np.random.default_rng(0)
    rows = []
    for w in (0.05, 0.2, 0.5):
        for d0 in (0.5, 1.0, 1.5):
            for eta in (0.25, 1.0, 4.0):
                rho = eta * d0 ** 2 / w
                dstar = theorem_equilibrium(w, d0, eta)
                finals = [simulate_spherical_pair(rng.normal(size=8), rng.normal(size=8),
                                                   w, d0, eta=eta, lr=0.05, steps=4000)[-1]
                          for _ in range(3)]
                d_final = float(np.mean(finals))
                rows.append(dict(w=w, d0=d0, eta=eta, rho=rho, d_star=dstar,
                                 d_final=d_final,
                                 abs_err=abs(d_final - (dstar if dstar is not None else 0.0))))
    sup = [r for r in rows if r["rho"] > 1]
    sub = [r for r in rows if r["rho"] <= 1]
    report = {
        "grid": rows,
        "rho_min": min(r["rho"] for r in rows), "rho_max": max(r["rho"] for r in rows),
        "superthreshold_max_abs_err": max(r["abs_err"] for r in sup),
        "subthreshold_max_d_final": max(r["d_final"] for r in sub),
    }
    report.update(validate_all(verbose=False))
    os.makedirs("results", exist_ok=True)
    json.dump(report, open("results/theory_validation.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2))

if __name__ == "__main__":
    main()