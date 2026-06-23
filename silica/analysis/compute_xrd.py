#!/usr/bin/env python3
"""
Compute simulated powder X-ray diffraction pattern using Debye formula.

Reads a LAMMPS dump or data file and computes I(2θ) for Cu Kα radiation.
Can be used to identify cristobalite, tridymite, and amorphous phases.

Reference peaks (Cu Kα, λ = 1.5406 Å):
  Cristobalite: 2θ ≈ 21.9° (101), 36.1° (200)
  Tridymite:    2θ ≈ 20.5°, 21.8°, 23.3°
  Amorphous:    broad hump at 2θ ≈ 22°

Usage:
    python compute_xrd.py dump.calc_pH7.0.final.lammpstrj
    python compute_xrd.py data.calc_pH7.0.final --format data
"""

import argparse
import numpy as np
import os
import sys


# ── Atomic Scattering Factors (Cromer-Mann coefficients) ────────────────────

SCATTERING_FACTORS = {
    'Si': {
        'a': [6.2915, 3.0353, 1.9891, 1.5410],
        'b': [2.4386, 32.3337, 0.6785, 81.6937],
        'c': 1.1407,
    },
    'O': {
        'a': [3.0485, 2.2868, 1.5463, 0.8670],
        'b': [13.2771, 5.7011, 0.3239, 32.9089],
        'c': 0.2508,
    },
}


def atomic_form_factor(q, element):
    """Compute atomic form factor f(q) using Cromer-Mann coefficients.

    Args:
        q: scattering vector magnitude (Å⁻¹), can be array
        element: 'Si' or 'O'

    Returns:
        f(q): form factor value(s)
    """
    params = SCATTERING_FACTORS[element]
    s = q / (4.0 * np.pi)  # sin(θ)/λ = Q/(4π)
    f = params['c']
    for a, b in zip(params['a'], params['b']):
        f = f + a * np.exp(-b * s**2)
    return f


# ── File Readers ─────────────────────────────────────────────────────────────

def read_dump_last_frame(filename):
    """Read the last frame from a LAMMPS dump file.

    Returns:
        positions: (N, 3) array
        types: list of atom types (1=Si, 2=O)
        box: (3,) array of box lengths
    """
    positions = []
    types = []
    box = np.zeros(3)

    with open(filename) as f:
        lines = f.readlines()

    # Find the last TIMESTEP
    last_ts_idx = -1
    for i, line in enumerate(lines):
        if 'ITEM: TIMESTEP' in line:
            last_ts_idx = i

    if last_ts_idx < 0:
        raise ValueError(f"No TIMESTEP found in {filename}")

    # Parse from last timestep
    i = last_ts_idx
    n_atoms = int(lines[i + 3].strip())

    # Box bounds
    for dim in range(3):
        parts = lines[i + 5 + dim].split()
        box[dim] = float(parts[1]) - float(parts[0])

    # Atoms
    header_line = lines[i + 8].strip()
    # Parse column indices from ITEM: ATOMS line
    cols = header_line.replace("ITEM: ATOMS", "").split()
    id_col = cols.index('id') if 'id' in cols else 0
    type_col = cols.index('type') if 'type' in cols else 1
    x_col = cols.index('x') if 'x' in cols else 2
    y_col = cols.index('y') if 'y' in cols else 3
    z_col = cols.index('z') if 'z' in cols else 4

    for j in range(n_atoms):
        parts = lines[i + 9 + j].split()
        types.append(int(parts[type_col]))
        positions.append([
            float(parts[x_col]),
            float(parts[y_col]),
            float(parts[z_col]),
        ])

    return np.array(positions), types, box


def read_data_file(filename):
    """Read positions from a LAMMPS data file.

    Returns:
        positions: (N, 3) array
        types: list of atom types
        box: (3,) array of box lengths
    """
    positions = []
    types = []
    box = np.zeros(3)

    with open(filename) as f:
        in_atoms = False
        for line in f:
            line = line.strip()
            if 'xlo xhi' in line:
                parts = line.split()
                box[0] = float(parts[1]) - float(parts[0])
            elif 'ylo yhi' in line:
                parts = line.split()
                box[1] = float(parts[1]) - float(parts[0])
            elif 'zlo zhi' in line:
                parts = line.split()
                box[2] = float(parts[1]) - float(parts[0])
            elif line.startswith('Atoms'):
                in_atoms = True
                continue
            elif in_atoms and line:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        types.append(int(parts[1]))
                        positions.append([
                            float(parts[3]),
                            float(parts[4]),
                            float(parts[5]),
                        ])
                    except ValueError:
                        if parts[0] in ('Velocities', 'Bonds'):
                            break

    return np.array(positions), types, box


# ── Debye XRD Calculation ────────────────────────────────────────────────────

def compute_xrd_debye(positions, types, box, q_range=None,
                      n_q_points=500, type_map=None):
    """Compute powder XRD pattern using Debye scattering formula.

    I(Q) = (1/N) Σᵢ Σⱼ fᵢ(Q) fⱼ(Q) sin(Q·rᵢⱼ)/(Q·rᵢⱼ) W(rᵢⱼ)

    with Lorch window function W(r) = sin(πr/Rc)/(πr/Rc)

    Args:
        positions: (N, 3) atom positions
        types: list of atom types
        box: (3,) box lengths
        q_range: (q_min, q_max) in Å⁻¹
        n_q_points: number of Q points
        type_map: dict mapping type index to element name

    Returns:
        two_theta: array of 2θ values (degrees, Cu Kα)
        intensity: array of I(2θ) values (normalized)
    """
    if type_map is None:
        type_map = {1: 'Si', 2: 'O'}

    if q_range is None:
        q_range = (0.5, 8.0)  # Å⁻¹, covers 2θ ~ 7° to 120° for Cu Kα

    wavelength = 1.5406  # Cu Kα (Å)
    N = len(positions)
    R_c = min(box) / 2.0  # Window function cutoff

    q_values = np.linspace(q_range[0], q_range[1], n_q_points)

    # Precompute form factors for each atom
    elements = [type_map.get(t, 'O') for t in types]
    unique_elements = list(set(elements))

    print(f"  Computing XRD for {N} atoms...")
    print(f"  Box: {box}")
    print(f"  R_c = {R_c:.2f} Å")
    print(f"  Q range: {q_range[0]:.2f} - {q_range[1]:.2f} Å⁻¹")

    # Compute all pairwise distances (with minimum image convention)
    intensity = np.zeros(n_q_points)

    # Self-scattering term
    for i in range(N):
        fi = atomic_form_factor(q_values, elements[i])
        intensity += fi**2

    # Cross terms (i != j) — optimized with vectorization
    batch_size = 100  # Process atoms in batches for memory efficiency
    n_batches = (N + batch_size - 1) // batch_size

    for batch in range(n_batches):
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, N)

        if (batch + 1) % 5 == 0 or batch == n_batches - 1:
            print(f"    Processing atoms {i_start}-{i_end} / {N}...")

        for i in range(i_start, i_end):
            # Compute distances from atom i to all atoms j > i
            dr = positions[i+1:] - positions[i]
            # Minimum image convention
            for dim in range(3):
                dr[:, dim] -= box[dim] * np.round(dr[:, dim] / box[dim])
            rij = np.linalg.norm(dr, axis=1)

            # Window function (Lorch)
            mask = rij < R_c
            rij_masked = rij[mask]
            if len(rij_masked) == 0:
                continue

            w = np.sinc(rij_masked / R_c)  # np.sinc(x) = sin(πx)/(πx)

            # Form factors
            fi = atomic_form_factor(q_values, elements[i])
            for j_local, j_global in enumerate(range(i+1, N)):
                if not mask[j_local]:
                    continue
                r = rij_masked[j_local] if j_local < len(rij_masked) else None
                if r is None:
                    continue

            # Vectorized: sum over all j for this i
            for k, q in enumerate(q_values):
                fi_q = atomic_form_factor(np.array([q]), elements[i])[0]
                qr = q * rij_masked
                sinc_qr = np.where(qr > 1e-10, np.sin(qr) / qr, 1.0)

                # Sum fj * sinc(qr) * W(r) for all j
                fj_q = np.array([
                    atomic_form_factor(np.array([q]),
                                       elements[i + 1 + idx])[0]
                    for idx, m in enumerate(mask) if m
                ])
                if len(fj_q) > 0:
                    contribution = fi_q * np.sum(fj_q * sinc_qr * w)
                    intensity[k] += 2.0 * contribution  # factor 2 for i,j + j,i

    intensity /= N

    # Convert Q to 2θ
    two_theta = 2.0 * np.degrees(np.arcsin(q_values * wavelength / (4.0 * np.pi)))

    return two_theta, intensity


def main():
    parser = argparse.ArgumentParser(
        description="Compute simulated powder XRD from LAMMPS output")
    parser.add_argument("input", help="LAMMPS dump or data file")
    parser.add_argument("--format", choices=["dump", "data"], default="dump",
                        help="Input file format (default: dump)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file prefix (default: xrd_<input>)")
    parser.add_argument("--q-min", type=float, default=0.5,
                        help="Min Q (Å⁻¹, default: 0.5)")
    parser.add_argument("--q-max", type=float, default=8.0,
                        help="Max Q (Å⁻¹, default: 8.0)")
    parser.add_argument("--n-points", type=int, default=500,
                        help="Number of Q points (default: 500)")
    args = parser.parse_args()

    # Read input
    if args.format == "dump":
        positions, types, box = read_dump_last_frame(args.input)
    else:
        positions, types, box = read_data_file(args.input)

    print(f"Read {len(positions)} atoms from {args.input}")

    # Compute XRD
    two_theta, intensity = compute_xrd_debye(
        positions, types, box,
        q_range=(args.q_min, args.q_max),
        n_q_points=args.n_points,
    )

    # Save data
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"xrd_{base}"

    np.savetxt(f"{args.output}.dat",
               np.column_stack([two_theta, intensity]),
               header="2theta(deg)  I(arb.units)",
               fmt="%.4f  %.6e")
    print(f"XRD data saved to {args.output}.dat")

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(two_theta, intensity, 'b-', linewidth=0.8)
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Intensity (arb. units)")
        ax.set_title(f"Simulated XRD — {os.path.basename(args.input)}")
        ax.set_xlim(10, 80)

        # Mark reference peaks
        ax.axvline(21.9, color='r', linestyle='--', alpha=0.5,
                   label='Cristobalite (101)')
        ax.axvline(36.1, color='r', linestyle=':', alpha=0.5,
                   label='Cristobalite (200)')
        ax.axvline(20.5, color='g', linestyle='--', alpha=0.5,
                   label='Tridymite')
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(f"{args.output}.png", dpi=150)
        print(f"XRD plot saved to {args.output}.png")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
