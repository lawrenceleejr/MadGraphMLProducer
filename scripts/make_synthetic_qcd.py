#!/usr/bin/env python3
"""
Synthetic "QCD-like" multijet event generator -> HDF5.

This does NOT run a matrix-element / parton-shower calculation. It fabricates
events that *roughly smell like* high-multiplicity QCD multijets and writes them
in the EXACT same HDF5 schema produced by the real pipeline
(run_docker.py :: process_lhe_to_hdf5), so the output is a drop-in for the same
training code (SPANet `/INPUTS/Source/`, PasswdABC `/source/`, `/EventVars/`,
and the legacy `jet_features` / `particle_features` / `event_features` arrays).

Use it when you need, e.g., a 7-jet QCD-flavoured sample that MadGraph cannot
generate directly (gg -> 7g is intractable at LO).

What is modelled (the "QCD smell"):
  * Steeply-falling event HT, partitioned into a pT *hierarchy* across the jets
    (leading jet dominates, soft tail) -> realistic pT ordering.
  * Momentum-balanced jets (vector pT sum ~ 0 at truth); MET arises only from a
    per-jet resolution smear -> small MET relative to HT, as in real QCD.
  * Central-ish rapidity (Gaussian in eta, clipped), uniform phi.
  * Jet mass ~ 0.1-0.15 * pT with spread.
  * Gluon-like fragmentation for the constituents:
      - multiplicity grows ~logarithmically with jet pT (high, gluon-like),
      - the constituent momentum fractions follow the QCD "hump-backed plateau"
        (Gaussian in xi = ln(1/z)) -> many soft particles + a few hard ones,
      - angular ordering: softer constituents sit further from the jet axis
        (still within the jet radius R).
      - realistic charged/neutral hadron + photon PDG-id mix.

It is a caricature, not a calculation -- good enough to pretrain / smoke-test /
augment, not to do physics measurements.

Dependencies: numpy, h5py  (no MadGraph, no Docker, no torch).

Examples:
    python scripts/make_synthetic_qcd.py -o qcd7.h5 -n 20000
    python scripts/make_synthetic_qcd.py -o qcd7.h5 -n 50000 --n-jets 7 --seed 1
    python scripts/make_synthetic_qcd.py -o qcd6_8.h5 -n 20000 --n-jets 7 --jet-spread 1
"""

import argparse
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("Error: h5py is required.  pip install h5py numpy")


# ----------------------------------------------------------------------------
# Schema (must match run_docker.py :: process_lhe_to_hdf5 exactly)
# ----------------------------------------------------------------------------
JET_FEATURES = ["pt", "eta", "phi", "mass", "n_constituents", "parent_pdg", "is_signal"]
PARTICLE_FEATURES = ["pt_rel", "eta_rel", "phi_rel", "energy", "pdg_id"]
EVENT_FEATURES = ["n_jets", "met_x", "met_y", "met_pt", "ht", "n_signal", "weight"]

GLUON_PDG = 21

# Rough hadron-level composition of a jet (particle-flow-like): charged pions
# dominate, photons from pi0, then kaons and baryons. (sign assigned at random
# for charged species.)
_HADRON_PDGS = np.array([211, 22, 321, 130, 2212, 2112])
_HADRON_PROB = np.array([0.50, 0.27, 0.09, 0.05, 0.05, 0.04])
_HADRON_CHARGED = np.array([True, False, True, False, True, False])


def sample_event(rng, n_jets, pt_min, eta_max, max_particles,
                 ht_index, ht_max, dirichlet_alpha, btag_rate):
    """Generate one synthetic QCD-like multijet event.

    Returns a dict of per-jet arrays plus event-level MET/HT.
    """
    # --- 1) Event energy scale: steeply-falling HT spectrum -----------------
    # Inverse-transform sample HT from dN/dHT ~ HT^-ht_index on [ht_min, ht_max].
    # Generate a touch above pt_min so the softest jet survives balancing+smear.
    gen_min = pt_min * 1.2
    ht_min = n_jets * gen_min * 2.0
    u = rng.random()
    p = ht_index - 1.0
    ht = (ht_min ** (-p) + u * (ht_max ** (-p) - ht_min ** (-p))) ** (-1.0 / p)

    # --- 2) Partition HT into a pT hierarchy (floor at gen_min) -------------
    # Dirichlet shares give one/few dominant jets + a soft tail; sort descending.
    shares = rng.dirichlet(np.full(n_jets, dirichlet_alpha))
    pt = gen_min + shares * (ht - n_jets * gen_min)
    pt = np.sort(pt)[::-1]

    # --- 3) Angles: central-ish eta, uniform phi ----------------------------
    eta = np.clip(rng.normal(0.0, 1.8, n_jets), -eta_max, eta_max)
    phi = rng.uniform(-np.pi, np.pi, n_jets)

    # --- 4) Momentum balance: subtract the transverse centroid --------------
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    px -= px.mean()
    py -= py.mean()
    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)

    # --- 5) Detector-like resolution smear -> this is where MET comes from --
    smear = np.clip(rng.normal(1.0, 0.08, n_jets), 0.3, None)
    pt = pt * smear
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    met_x, met_y = -px.sum(), -py.sum()          # MET = -(visible vector sum)

    # order by measured (smeared) pT, like a reconstructed jet collection
    order = np.argsort(pt)[::-1]
    pt, eta, phi = pt[order], eta[order], phi[order]

    # --- 6) Jet mass ~ 0.12 * pT --------------------------------------------
    mass = pt * np.clip(rng.normal(0.12, 0.04, n_jets), 0.02, 0.45)

    # --- 7) b-tag flags (small rate; gluon->bb splitting etc.) --------------
    btag = (rng.random(n_jets) < btag_rate).astype(np.float32)

    # --- 8) Constituents: gluon-like fragmentation per jet ------------------
    p_pt_rel = np.zeros((n_jets, max_particles), dtype=np.float32)
    p_eta_rel = np.zeros((n_jets, max_particles), dtype=np.float32)
    p_phi_rel = np.zeros((n_jets, max_particles), dtype=np.float32)
    p_energy = np.zeros((n_jets, max_particles), dtype=np.float32)
    p_pdg = np.zeros((n_jets, max_particles), dtype=np.float32)
    n_const = np.zeros(n_jets, dtype=np.int32)

    R = 0.4
    for j in range(n_jets):
        e_jet = np.hypot(pt[j] * np.cosh(eta[j]), mass[j])

        # multiplicity grows ~ logarithmically with pT (gluon jets are busy)
        mean_mult = 6.0 + 7.0 * np.log(max(pt[j] / pt_min, 1.0))
        n = int(np.clip(rng.poisson(mean_mult), 2, max_particles))

        # hump-backed plateau: xi = ln(1/z) ~ Normal(mu, sigma)
        mu_xi = 0.5 * np.log(max(pt[j] / 0.3, 2.0))
        xi = rng.normal(mu_xi, 1.3, n)
        z = np.clip(np.exp(-xi), 1e-3, 0.95)
        z = z / z.sum()                          # constituents share the jet pT
        z = np.sort(z)[::-1]                      # leading constituent first

        # angular ordering: softer constituents sit further from the axis
        dr = R * np.sqrt(rng.random(n)) * (0.25 + 0.75 * (1.0 - z))
        dr = np.clip(dr, 0.0, R)
        alpha = rng.uniform(-np.pi, np.pi, n)

        n_const[j] = n
        p_pt_rel[j, :n] = z                       # fraction of jet pT (==1.0 if single)
        p_eta_rel[j, :n] = dr * np.cos(alpha)
        p_phi_rel[j, :n] = dr * np.sin(alpha)
        p_energy[j, :n] = z * e_jet
        # PDG-id mix (sign charged species at random)
        cat = rng.choice(len(_HADRON_PDGS), size=n, p=_HADRON_PROB)
        pdg = _HADRON_PDGS[cat].astype(np.float32)
        sign = np.where(_HADRON_CHARGED[cat] & (rng.random(n) < 0.5), -1.0, 1.0)
        p_pdg[j, :n] = pdg * sign

    return {
        "pt": pt.astype(np.float32), "eta": eta.astype(np.float32),
        "phi": phi.astype(np.float32), "mass": mass.astype(np.float32),
        "btag": btag, "n_const": n_const,
        "p_pt_rel": p_pt_rel, "p_eta_rel": p_eta_rel, "p_phi_rel": p_phi_rel,
        "p_energy": p_energy, "p_pdg": p_pdg,
        "met_x": np.float32(met_x), "met_y": np.float32(met_y),
        "ht": np.float32(pt.sum()),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate a synthetic QCD-like multijet HDF5 (drop-in schema).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-o", "--output", default="synthetic_qcd.h5", help="Output HDF5 path")
    ap.add_argument("-n", "--events", type=int, default=10000, help="Number of events")
    ap.add_argument("--n-jets", type=int, default=7, help="Jets per event")
    ap.add_argument("--jet-spread", type=int, default=0,
                    help="If >0, vary jet count as n-jets +/- up to this (uniform); else fixed")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed")
    ap.add_argument("--max-jets", type=int, default=20, help="Array width for jets (schema)")
    ap.add_argument("--max-particles", type=int, default=100,
                    help="Array width for constituents per jet (schema)")
    ap.add_argument("--pt-min", type=float, default=20.0, help="Min jet pT (GeV)")
    ap.add_argument("--eta-max", type=float, default=4.5, help="Max |eta| for jets")
    ap.add_argument("--ht-index", type=float, default=4.5,
                    help="Falling HT spectrum exponent (dN/dHT ~ HT^-index)")
    ap.add_argument("--ht-max", type=float, default=3000.0, help="Max HT (GeV)")
    ap.add_argument("--dirichlet-alpha", type=float, default=2.0,
                    help="Jet pT-sharing concentration (smaller -> steeper hierarchy)")
    ap.add_argument("--btag-rate", type=float, default=0.03, help="Per-jet b-tag probability")
    ap.add_argument("--chunk", type=int, default=2000, help="Events written per HDF5 chunk")
    ap.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    args = ap.parse_args()

    if args.n_jets > args.max_jets:
        sys.exit(f"--n-jets ({args.n_jets}) cannot exceed --max-jets ({args.max_jets})")

    rng = np.random.default_rng(args.seed)
    N, J, P = args.events, args.max_jets, args.max_particles
    comp = None if args.compression == "none" else args.compression

    print(f"Generating {N} synthetic QCD-like events "
          f"({args.n_jets} jets, max_jets={J}, max_particles={P}) -> {args.output}")

    def ds(grp, name, shape, dtype):
        chunks = (min(args.chunk, N),) + tuple(shape[1:]) if shape[1:] else (min(args.chunk, N),)
        return grp.create_dataset(name, shape=shape, dtype=dtype,
                                  compression=comp, chunks=chunks)

    with h5py.File(args.output, "w") as f:
        # SPANet format -------------------------------------------------------
        src = f.create_group("INPUTS/Source")
        d_mask = ds(src, "MASK", (N, J), bool)
        d_pt = ds(src, "pt", (N, J), np.float32)
        d_eta = ds(src, "eta", (N, J), np.float32)
        d_phi = ds(src, "phi", (N, J), np.float32)
        d_mass = ds(src, "mass", (N, J), np.float32)
        d_btag = ds(src, "btag", (N, J), np.float32)

        # SPANet targets: QCD is background -> empty group (matches real output)
        f.create_group("TARGETS")

        # PasswdABC format ----------------------------------------------------
        s2 = f.create_group("source")
        d_e = ds(s2, "e", (N, J), np.float32)
        ds(s2, "pt", (N, J), np.float32)
        ds(s2, "eta", (N, J), np.float32)
        ds(s2, "phi", (N, J), np.float32)
        ds(s2, "mask", (N, J), bool)
        ev = f.create_group("EventVars")
        d_normweight = ds(ev, "normweight", (N,), np.float32)

        # Legacy format -------------------------------------------------------
        d_jetf = ds(f, "jet_features", (N, J, len(JET_FEATURES)), np.float32)
        d_partf = ds(f, "particle_features", (N, J, P, len(PARTICLE_FEATURES)), np.float32)
        d_evtf = ds(f, "event_features", (N, len(EVENT_FEATURES)), np.float32)
        d_jmask = ds(f, "jet_mask", (N, J), bool)
        d_pmask = ds(f, "particle_mask", (N, J, P), bool)

        # ---- fill in chunks (low memory) -----------------------------------
        for start in range(0, N, args.chunk):
            end = min(start + args.chunk, N)
            nb = end - start

            jetf = np.zeros((nb, J, len(JET_FEATURES)), np.float32)
            partf = np.zeros((nb, J, P, len(PARTICLE_FEATURES)), np.float32)
            evtf = np.zeros((nb, len(EVENT_FEATURES)), np.float32)
            jmask = np.zeros((nb, J), bool)
            pmask = np.zeros((nb, J, P), bool)
            spt = np.zeros((nb, J), np.float32)
            seta = np.zeros((nb, J), np.float32)
            sphi = np.zeros((nb, J), np.float32)
            smass = np.zeros((nb, J), np.float32)
            sbtag = np.zeros((nb, J), np.float32)
            se = np.zeros((nb, J), np.float32)

            for k in range(nb):
                nj = args.n_jets
                if args.jet_spread > 0:
                    nj = int(np.clip(args.n_jets + rng.integers(-args.jet_spread,
                                                                args.jet_spread + 1),
                                     1, J))
                evt = sample_event(rng, nj, args.pt_min, args.eta_max, P,
                                   args.ht_index, args.ht_max,
                                   args.dirichlet_alpha, args.btag_rate)

                jmask[k, :nj] = True
                spt[k, :nj] = evt["pt"]
                seta[k, :nj] = evt["eta"]
                sphi[k, :nj] = evt["phi"]
                smass[k, :nj] = evt["mass"]
                sbtag[k, :nj] = evt["btag"]
                e_jet = np.hypot(evt["pt"] * np.cosh(evt["eta"]), evt["mass"])
                se[k, :nj] = e_jet

                jetf[k, :nj, 0] = evt["pt"]
                jetf[k, :nj, 1] = evt["eta"]
                jetf[k, :nj, 2] = evt["phi"]
                jetf[k, :nj, 3] = evt["mass"]
                jetf[k, :nj, 4] = evt["n_const"]
                jetf[k, :nj, 5] = GLUON_PDG          # parent_pdg: gluon
                jetf[k, :nj, 6] = 0                  # is_signal: QCD background

                partf[k, :nj, :, 0] = evt["p_pt_rel"]
                partf[k, :nj, :, 1] = evt["p_eta_rel"]
                partf[k, :nj, :, 2] = evt["p_phi_rel"]
                partf[k, :nj, :, 3] = evt["p_energy"]
                partf[k, :nj, :, 4] = evt["p_pdg"]
                for j in range(nj):
                    pmask[k, j, :evt["n_const"][j]] = True

                met_pt = float(np.hypot(evt["met_x"], evt["met_y"]))
                evtf[k] = [nj, evt["met_x"], evt["met_y"], met_pt, evt["ht"], 0, 1.0]

            d_jetf[start:end] = jetf
            d_partf[start:end] = partf
            d_evtf[start:end] = evtf
            d_jmask[start:end] = jmask
            d_pmask[start:end] = pmask
            d_mask[start:end] = jmask
            d_pt[start:end] = spt
            d_eta[start:end] = seta
            d_phi[start:end] = sphi
            d_mass[start:end] = smass
            d_btag[start:end] = sbtag
            d_e[start:end] = se
            f["source/pt"][start:end] = spt
            f["source/eta"][start:end] = seta
            f["source/phi"][start:end] = sphi
            f["source/mask"][start:end] = jmask
            d_normweight[start:end] = 1.0
            print(f"  wrote events {start:>8}..{end:<8}", end="\r")

        # ---- attributes (match the real writer) ----------------------------
        f.attrs["config_name"] = "synthetic_qcd"
        f.attrs["n_events"] = N
        f.attrs["particle_features"] = PARTICLE_FEATURES
        f.attrs["jet_features"] = JET_FEATURES
        f.attrs["event_features"] = EVENT_FEATURES
        f.attrs["max_jets"] = J
        f.attrs["max_particles"] = P
        f.attrs["jet_algorithm"] = "synthetic (no clustering)"
        f.attrs["jet_radius"] = 0.4
        f.attrs["process_string"] = f"synthetic QCD-like multijet ({args.n_jets} jets)"
        f.attrs["model"] = "synthetic_qcd"
        f.attrs["spanet_inputs"] = ["Source"]
        f.attrs["spanet_jet_features"] = ["pt", "eta", "phi", "mass", "btag"]
        f.attrs["passwdabc_source_features"] = ["e", "pt", "eta", "phi"]
        f.attrs["passwdabc_event_vars"] = ["normweight"]
        f.attrs["synthetic"] = True

    print(f"\nDone. Wrote {N} events to {args.output}")


if __name__ == "__main__":
    main()
