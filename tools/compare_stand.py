"""Ankle stiffness check from a stand_test log: how far the ankles give under Albert's weight (measured - commanded pitch),
and the lean, per phase, next to the simulation's numbers for the same procedure (outputs/freestand/stand_ref_sim.csv).
The sim assumes the firmware's PD acts on the two ankle motors (~900 Nm/rad in pitch): if Albert's ankles give much more
than the sim's, they are softer, and standing without support has less margin than the sim says.

    python -m tools.compare_stand logs/B1_stand.csv
"""
from __future__ import annotations

import sys

import numpy as np

K_PITCH, K_ROLL = 2.02, 0.8
REF = "outputs/freestand/stand_ref_sim.csv"


def per_phase(path, stance_ankle=-0.21):
    r = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    out = {}
    for ph in ("to_stance", "lower_rope", "hold", "arm_replay", "raise_rope"):
        m = r["phase"] == ph
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        if ph == "to_stance":
            idx = idx[-15:]  # the stance reached, rope still carrying
        pl = (r["pos_L_Ankle_A"][idx] - r["pos_L_Ankle_B"][idx]) / (2 * K_PITCH)
        pr = (r["pos_R_Ankle_A"][idx] - r["pos_R_Ankle_B"][idx]) / (2 * K_PITCH)
        sl = 1000 * np.mean(pl - stance_ankle)            # ankle pitch, measured - commanded (mrad), left-leg convention
        sr = -1000 * np.mean(pr - (-stance_ankle))
        out[ph] = (float(np.mean(r["lean_deg"][m])), float(np.max(r["lean_deg"][m])), sl, sr)
    return out


def main():
    real = per_phase(sys.argv[1]); sim = per_phase(REF)
    print(f"{'phase':11s} | {'lean mean/max (deg)':>22s} | {'ankle give L/R (mrad)':>24s}")
    for ph in real:
        r, s = real[ph], sim.get(ph)
        print(f"{ph:11s} | Albert {r[0]:5.1f}/{r[1]:5.1f}  sim {s[0]:5.1f}/{s[1]:5.1f} | Albert {r[2]:+6.1f}/{r[3]:+6.1f}  sim {s[2]:+6.1f}/{s[3]:+6.1f}"
              if s else f"{ph:11s} | Albert {r[0]:5.1f}/{r[1]:5.1f} | Albert {r[2]:+6.1f}/{r[3]:+6.1f}")
    if all(ph in d for d in (real, sim) for ph in ("to_stance", "hold")):
        # extra give when the rope stops carrying: load / stiffness (encoder offsets cancel)
        dr = (real["hold"][2] - real["to_stance"][2]) + (real["hold"][3] - real["to_stance"][3])
        ds = (sim["hold"][2] - sim["to_stance"][2]) + (sim["hold"][3] - sim["to_stance"][3])
        print(f"\nextra ankle give once the rope is slack (L+R): Albert {dr:+.1f} mrad, sim {ds:+.1f} mrad")
        if abs(dr) > 1.0:
            k = ds / dr
            print(f"Albert's ankle stiffness ~ {k:.2f} x the model's (~1: as modelled; < 0.5: much softer: re-run the stance search "
                  f"in sim_freestand with the ankle gains scaled by {k:.2f} before going further)")
        print("(the rope's share during the takeover is not known exactly; the lean comparison above matters as much)")


if __name__ == "__main__":
    main()
