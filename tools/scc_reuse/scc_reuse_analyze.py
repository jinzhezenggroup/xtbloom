#!/usr/bin/env python3
"""Phase-1 analyzer for issue #343: SCC subspace-reuse potential.

Reads the raw diagnostic stream emitted by ``scc_reuse_capture`` (either a
single-geometry ``single`` run or a warm-start ``traj`` run) and quantifies how
much the previous SCC / geometry eigenspace can be reused:

* relative Frobenius change of the effective Hamiltonian and density;
* S-metric principal angles between successive occupied subspaces;
* the fraction of the new occupied subspace captured by the previous one;
* the generalized residual of the previous eigenpairs against the new
  Hamiltonian (R = H_{k+1} C_k - S C_k diag(eps_k));
* per-iteration driver-step and isolated-eigensolve wall time.

For trajectory documents the analogous quantities are reported between the
converged states of consecutive geometries.

Document layout (see scc_reuse_capture.cpp):

    diagnostic xtbloom-scc-reuse-v1
    case <name> / nat / atomic_numbers / positions / molecular_charge /
    unpaired_electrons / temperature_kelvin / maximum_iterations
    geometry <generation>
      nao / nspin / overlap / core_hamiltonian
      iteration <k>: step_micros, eigensolve_micros, hamiltonian,
                     coefficients, eigenvalues, occupations, density
      converged: coefficients, eigenvalues, occupations, density
      converged_state <0|1>
    [case <name> ...  geometry <generation> ...]   (trajectory mode)
    end-of-diagnostics terminal single_iterations=<n>

Conventions, verified against the production CPU eigensolver: coefficients are
row-major C[r, orbital] with C^T S C = I; occupations are [2, nao] alpha then
beta; density P = C diag(f_alpha + f_beta) C^T. Matrices are reshaped row-major
here. All values are atomic units; times are microseconds.

This tool is experimental investigation tooling for issue #343, not part of
any acceptance or conformance gate.
"""

import argparse
import hashlib
import json
import math
import sys

import numpy as np


class DocumentError(ValueError):
    pass


class Parser:
    def __init__(self, path):
        with open(path, encoding="ascii") as handle:
            self.lines = [line.strip() for line in handle if line.strip()]
        self.index = 0

    def next(self):
        if self.index >= len(self.lines):
            raise DocumentError("unexpected end of document")
        line = self.lines[self.index]
        self.index += 1
        return line

    def peek(self):
        if self.index >= len(self.lines):
            return None
        return self.lines[self.index]

    def expect(self, label):
        line = self.next()
        if line != label:
            raise DocumentError(f"line {self.index}: expected {label!r}, got {line!r}")

    def scalar(self, label):
        """Read one scalar after (a) 'label' alone or (b) 'label <value>'."""
        line = self.next()
        if line == label:
            return self.next(), False
        if line.startswith(label + " "):
            return line[len(label) + 1 :], True
        raise DocumentError(f"line {self.index}: expected {label!r}, got {line!r}")

    def int_value(self, label):
        text, _ = self.scalar(label)
        try:
            return int(text)
        except ValueError as exc:
            raise DocumentError(
                f"line {self.index}: expected integer for {label!r}"
            ) from exc

    def float_value(self, label):
        text, _ = self.scalar(label)
        try:
            return float(text)
        except ValueError as exc:
            raise DocumentError(
                f"line {self.index}: expected float for {label!r}"
            ) from exc

    def ints(self, count):
        try:
            return [int(self.next()) for _ in range(count)]
        except ValueError as exc:
            raise DocumentError(f"line {self.index}: expected integer") from exc

    def floats(self, count):
        try:
            return [float(self.next()) for _ in range(count)]
        except ValueError as exc:
            raise DocumentError(f"line {self.index}: expected float") from exc


def matrix(p, n):
    return np.asarray(p.floats(n * n), dtype=np.float64).reshape(n, n)


def parse_document(path):
    p = Parser(path)
    header = p.next()
    if header != "diagnostic xtbloom-scc-reuse-v1":
        raise DocumentError(f"{path}: not an xtbloom-scc-reuse-v1 document")

    def read_case():
        line = p.next()
        if not line.startswith("case "):
            raise DocumentError(f"line {p.index}: expected 'case <name>', got {line!r}")
        case = line[len("case ") :]
        nat = p.int_value("nat")
        p.expect("atomic_numbers")
        atomic_numbers = p.ints(nat)
        p.expect("positions")
        positions = p.floats(3 * nat)
        charge = p.float_value("molecular_charge")
        unpaired = p.int_value("unpaired_electrons")
        temperature = p.float_value("temperature_kelvin")
        max_iter = p.int_value("maximum_iterations")
        return {
            "case": case,
            "nat": nat,
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "molecular_charge": charge,
            "unpaired_electrons": unpaired,
            "temperature_kelvin": temperature,
            "maximum_iterations": max_iter,
        }

    metadata = [read_case()]

    geometries = []

    def read_geometry():
        gen = p.int_value("geometry")
        nao = p.int_value("nao")
        nspin = p.int_value("nspin")
        p.expect("overlap")
        s = matrix(p, nao)
        p.expect("core_hamiltonian")
        h0 = matrix(p, nao)
        per_iter = []
        while True:
            line = p.next()
            if line.startswith("iteration "):
                k = int(line[len("iteration ") :])
                step = p.int_value("step_micros")
                eig = p.int_value("eigensolve_micros")
                p.expect("hamiltonian")
                h = matrix(p, nao)
                p.expect("coefficients")
                c = np.asarray(p.floats(nspin * nao * nao), dtype=np.float64).reshape(
                    nspin, nao, nao
                )
                p.expect("eigenvalues")
                eps = np.asarray(p.floats(nspin * nao), dtype=np.float64).reshape(
                    nspin, nao
                )
                p.expect("occupations")
                occ = np.asarray(p.floats(2 * nao), dtype=np.float64).reshape(2, nao)
                p.expect("density")
                dens = matrix(p, nao)
                per_iter.append(
                    {
                        "k": k,
                        "step_micros": step,
                        "eigensolve_micros": eig,
                        "H": h,
                        "C": c,
                        "eps": eps,
                        "occ": occ,
                        "P": dens,
                    }
                )
            elif line == "converged":
                p.expect("coefficients")
                c = np.asarray(p.floats(nspin * nao * nao), dtype=np.float64).reshape(
                    nspin, nao, nao
                )
                p.expect("eigenvalues")
                eps = np.asarray(p.floats(nspin * nao), dtype=np.float64).reshape(
                    nspin, nao
                )
                p.expect("occupations")
                occ = np.asarray(p.floats(2 * nao), dtype=np.float64).reshape(2, nao)
                p.expect("density")
                dens = matrix(p, nao)
                converged_state = p.int_value("converged_state") == 1
                return {
                    "generation": gen,
                    "nao": nao,
                    "nspin": nspin,
                    "overlap": s,
                    "core_hamiltonian": h0,
                    "iterations": per_iter,
                    "converged": {"C": c, "eps": eps, "occ": occ, "P": dens},
                    "converged_state": converged_state,
                }
            else:
                raise DocumentError(f"unexpected token {line!r} inside geometry body")

    while True:
        line = p.peek()
        if line is None:
            raise DocumentError("document missing end-of-diagnostics")
        if line.startswith("case "):
            metadata.append(read_case())
            continue
        if line == "geometry" or line.startswith("geometry "):
            geometries.append(read_geometry())
            continue
        if line.startswith("end-of-diagnostics"):
            p.next()
            break
        raise DocumentError(f"unexpected token {line!r} at top level")

    return {
        "case": metadata,
        "geometries": geometries,
        "terminal_line": p.lines[p.index - 1] if p.index <= len(p.lines) else "",
    }


def occupied_total(occ):
    return occ[0] + occ[1]


def occupied_mask(occ, spin, nspin):
    """Binary occupied mask for one spin channel: shared orbitals use the
    summed alpha+beta occupation, unrestricted channels use their own row.
    """
    if nspin == 1:
        return occupied_total(occ) > 0.5
    return occ[spin] > 0.5


def subspace_metrics(c_prev, c_cur, occ_prev, occ_cur, s_matrix, nspin):
    """Principal-angle and capture metrics between two occupied subspaces.

    Returns (cos_min, angle_max_deg, capture_fraction, chordal_distance) where
    cos_min is the smallest principal-angle cosine (worst paired direction),
    angle_max its angle in degrees, capture_fraction the fraction of the new
    occupied subspace reproduced by the previous one, and chordal_distance
    sqrt(sum_i sin^2(theta_i)) over the intersection dimension. Restricted
    (nspin == 1) uses the shared orbital subspace; unrestricted reports the
    average over spin channels.
    """
    cos_vals = []
    captures = []
    chordals = []
    for spin in range(nspin):
        mask_p = occupied_mask(occ_prev, spin, nspin)
        mask_c = occupied_mask(occ_cur, spin, nspin)
        c1 = np.asarray(c_prev)[spin][:, mask_p]
        c2 = np.asarray(c_cur)[spin][:, mask_c]
        n1, n2 = c1.shape[1], c2.shape[1]
        if n1 == 0 or n2 == 0:
            cos_vals.append(1.0)
            captures.append(1.0)
            chordals.append(0.0)
            continue
        cross = c1.T @ s_matrix @ c2
        sigma = np.linalg.svd(cross, compute_uv=False)
        cos_sq = np.clip(sigma * sigma, 0.0, 1.0)
        dim = min(n1, n2)
        captures.append(float(np.sum(cos_sq[:dim]) / n2))
        cos = np.clip(sigma[:dim], 0.0, 1.0)
        cos_vals.append(float(np.min(cos)))
        chordals.append(math.sqrt(float(np.sum(1.0 - cos * cos))))
    cos_min = min(cos_vals)
    angle_max = math.degrees(math.acos(max(cos_min, 0.0)))
    capture = sum(captures) / nspin
    chordal = sum(chordals) / nspin
    return cos_min, angle_max, capture, chordal


def density_overlap(p1, p2):
    den = math.sqrt(float(np.trace(p1 @ p1)) * float(np.trace(p2 @ p2)))
    if den <= 0.0:
        return 1.0
    return float(np.trace(p1 @ p2)) / den


def generalized_residual(h_new, c_old, eps_old, s_matrix, occ_old, nspin):
    """||H_new C_old - S C_old diag(eps_old)||_F / ||H_new||_F (full, then
    restricted to the occupied columns of C_old), averaged over spin channels
    for unrestricted systems.
    """
    h = np.asarray(h_new)
    s = np.asarray(s_matrix)
    hnorm = max(float(np.linalg.norm(h)), 1e-300)
    rel_full_total = 0.0
    rel_occ_total = 0.0
    for spin in range(nspin):
        c = np.asarray(c_old)[spin]
        eps = np.asarray(eps_old)[spin]
        full = h @ c - s @ c @ np.diag(eps)
        rel_full_total += float(np.linalg.norm(full)) / hnorm
        mask = occupied_mask(occ_old, spin, nspin)
        c_occ = c[:, mask]
        if c_occ.shape[1] == 0:
            rel_occ_total += float("nan")
        else:
            rel_occ_total += (
                float(np.linalg.norm(h @ c_occ - s @ c_occ @ np.diag(eps[mask])))
                / hnorm
            )
    return rel_full_total / nspin, rel_occ_total / nspin


def rr_eigenvalue_error(h_new, c_prev, eps_cur, occ_prev, occ_cur, nspin):
    """Max abs and RMS error of the Rayleigh-Ritz eigenvalues of H_new in the
    previous occupied subspace versus the new occupied eigenvalues.

    Because C^T S C = I on its own basis, the projected problem is
    C_prev_occ^T H_new C_prev_occ; its eigenvalues are the best approximation
    a recycled subspace can deliver. When the two occupied subspaces coincide
    this error is ~0 even if the individual eigenvalues drifted, separating
    "subspace capture" from "eigenvalue drift" that a cheap RR step cures.
    """
    maxima = []
    rmss = []
    for spin in range(nspin):
        h = np.asarray(h_new)
        c = np.asarray(c_prev)[spin][:, occupied_mask(occ_prev, spin, nspin)]
        eps_new = np.asarray(eps_cur)[spin][occupied_mask(occ_cur, spin, nspin)]
        if c.shape[1] == 0 or eps_new.size == 0:
            maxima.append(float("nan"))
            rmss.append(float("nan"))
            continue
        h_proj = c.T @ h @ c
        eig = np.linalg.eigvalsh((h_proj + h_proj.T) / 2.0)
        n = min(eig.size, eps_new.size)
        diff = np.abs(np.sort(eig)[:n] - np.sort(eps_new)[:n])
        maxima.append(float(np.max(diff)))
        rmss.append(math.sqrt(float(np.mean(diff * diff))))
    return max(maxima), math.sqrt(float(np.mean([r * r for r in rmss])))


def validate_eigenpairs(h, c, eps, s_matrix):
    """Max |C^T S C - I| and max |H C - S C diag(eps)| / ||H|| per spin."""
    ctsc_max = 0.0
    hc_max = 0.0
    hnorm = float(np.linalg.norm(h)) or 1.0
    for spin in range(c.shape[0]):
        csp = np.asarray(c[spin])
        epsp = eps[spin]
        ctsc = csp.T @ s_matrix @ csp - np.eye(csp.shape[0])
        hc = h @ csp - s_matrix @ csp @ np.diag(epsp)
        ctsc_max = max(ctsc_max, float(np.max(np.abs(ctsc))))
        hc_max = max(hc_max, float(np.max(np.abs(hc))) / hnorm)
    return ctsc_max, hc_max


def analyze_geometry(geo):
    n = geo["nao"]
    s = geo["overlap"]
    rows = []
    for idx, it in enumerate(geo["iterations"]):
        row = {
            "k": it["k"],
            "step_micros": it["step_micros"],
            "eigensolve_micros": it["eigensolve_micros"],
        }
        ctsc, hc = validate_eigenpairs(it["H"], it["C"], it["eps"], s)
        row["validation_ctsc_max"] = ctsc
        row["validation_he_max"] = hc
        if idx > 0:
            prev = geo["iterations"][idx - 1]
            d_h = float(np.linalg.norm(it["H"] - prev["H"]))
            row["rel_dH"] = d_h / (float(np.linalg.norm(prev["H"])) or 1.0)
            d_p = float(np.linalg.norm(it["P"] - prev["P"]))
            row["rel_dP"] = d_p / (float(np.linalg.norm(prev["P"])) or 1.0)
            row["density_overlap"] = density_overlap(prev["P"], it["P"])
            cos_min, angle_max, capture, chordal = subspace_metrics(
                prev["C"], it["C"], prev["occ"], it["occ"], s, geo["nspin"]
            )
            row["subspace_min_cos"] = cos_min
            row["subspace_max_angle_deg"] = angle_max
            row["subspace_capture_fraction"] = capture
            row["subspace_chordal_distance"] = chordal
            rel_full, rel_occ = generalized_residual(
                it["H"], prev["C"], prev["eps"], s, prev["occ"], geo["nspin"]
            )
            row["rel_residual_full"] = rel_full
            row["rel_residual_occupied"] = rel_occ
            rr_max, rr_rms = rr_eigenvalue_error(
                it["H"], prev["C"], it["eps"], prev["occ"], it["occ"], geo["nspin"]
            )
            row["rr_eigenvalue_max_err"] = rr_max
            row["rr_eigenvalue_rms_err"] = rr_rms
        rows.append(row)

    block = {
        "generation": geo["generation"],
        "nao": n,
        "iterations": rows,
        "converged_state": geo["converged_state"],
    }
    if rows:
        block["eigensolve_total_micros"] = sum(r["eigensolve_micros"] for r in rows)
        block["step_total_micros"] = sum(r["step_micros"] for r in rows)
        block["eigensolve_share"] = block["eigensolve_total_micros"] / max(
            block["step_total_micros"], 1
        )
    return block


def trajectory_metrics(g1, g2):
    s1, s2 = g1["overlap"], g2["overlap"]
    c1, c2 = g1["converged"], g2["converged"]
    rel_dh0 = float(np.linalg.norm(g2["core_hamiltonian"] - g1["core_hamiltonian"])) / (
        float(np.linalg.norm(g1["core_hamiltonian"])) or 1.0
    )
    rel_dp = float(np.linalg.norm(c2["P"] - c1["P"])) / (
        float(np.linalg.norm(c1["P"])) or 1.0
    )
    # Consecutive geometries share one basis, so S is identical; still use the
    # appropriate S for each subspace.
    cos_min, angle_max, capture, chordal = subspace_metrics(
        c1["C"], c2["C"], c1["occ"], c2["occ"], s1, g1["nspin"]
    )
    rel_full, rel_occ = generalized_residual(
        g2["core_hamiltonian"], c1["C"], c1["eps"], s1, c1["occ"], g1["nspin"]
    )
    return {
        "rel_dH0": rel_dh0,
        "rel_density": rel_dp,
        "density_overlap": density_overlap(c1["P"], c2["P"]),
        "subspace_min_cos": cos_min,
        "subspace_max_angle_deg": angle_max,
        "subspace_capture_fraction": capture,
        "subspace_chordal_distance": chordal,
        "rel_residual_full": rel_full,
        "rel_residual_occupied": rel_occ,
    }


def build_report(path, doc):
    geometries = [analyze_geometry(g) for g in doc["geometries"]]
    report = {
        "document": path,
        "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        "case": doc["case"],
        "geometries": geometries,
    }
    if len(geometries) >= 2:
        report["trajectory"] = trajectory_metrics(
            doc["geometries"][0], doc["geometries"][1]
        )
    return report


def summarize(report):
    lines = []
    for geo in report["geometries"]:
        rows = geo["iterations"]
        iters = len(rows)
        conv = (
            "converged"
            if geo["converged_state"]
            else "NOT converged (max iterations/failure)"
        )
        lines.append(
            f"geometry {geo['generation']}: nao={geo['nao']} iterations={iters} [{conv}]"
        )
        if iters:
            lines.append(
                f"  eigensolve total {geo['eigensolve_total_micros']} us, "
                f"share of step {geo['eigensolve_share']:.2f}"
            )

        if iters >= 2:
            lines.append(
                "  rel||dH||      " + " ".join(f"{r['rel_dH']:9.3e}" for r in rows[1:])
            )
            lines.append(
                "  rel||dP||      " + " ".join(f"{r['rel_dP']:9.3e}" for r in rows[1:])
            )
            lines.append(
                "  ang_max(deg)   "
                + " ".join(f"{r['subspace_max_angle_deg']:9.3f}" for r in rows[1:])
            )
            lines.append(
                "  capture frac   "
                + " ".join(f"{r['subspace_capture_fraction']:9.4f}" for r in rows[1:])
            )
            lines.append(
                "  resid occ      "
                + " ".join(f"{r['rel_residual_occupied']:9.3e}" for r in rows[1:])
            )
            lines.append(
                "  RR eig err     "
                + " ".join(f"{r['rr_eigenvalue_max_err']:9.3e}" for r in rows[1:])
            )
    if "trajectory" in report:
        t = report["trajectory"]
        lines.append(
            f"trajectory: rel||dH0||={t['rel_dH0']:.3e} rel||dP||={t['rel_density']:.3e} "
            f"capture={t['subspace_capture_fraction']:.4f} "
            f"ang_max={t['subspace_max_angle_deg']:.3f} deg"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "documents", nargs="+", help="scc_reuse_capture diagnostic stream(s)"
    )
    parser.add_argument("--report", help="write the JSON report to this path")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress the console summary"
    )
    args = parser.parse_args(argv)

    reports = []
    for path in args.documents:
        doc = parse_document(path)
        report = build_report(path, doc)
        reports.append(report)
        if not args.quiet:
            print(summarize(report))
            print()
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(
                reports if len(reports) > 1 else reports[0],
                handle,
                indent=2,
                allow_nan=True,
            )
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
