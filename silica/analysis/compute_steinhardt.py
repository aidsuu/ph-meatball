#!/usr/bin/env python3
"""
Compute per-atom Steinhardt order parameters (Q4, Q6) and classify
each Si atom as amorphous, cristobalite-like, or tridymite-like.

Reference Q values from ideal crystal structures:
  Phase          Q4       Q6
  Amorphous     ~0.03    ~0.03
  β-Cristobalite ~0.19    ~0.57
  Tridymite      ~0.13    ~0.48
  α-Quartz       ~0.14    ~0.54

Usage:
    python compute_steinhardt.py dump.calc_pH7.0.final.lammpstrj
    python compute_steinhardt.py data.calc_pH7.0.final --format data
"""

import argparse
import numpy as np
from scipy.special import sph_harm
import os
import sys


# ─── File Readers (shared with compute_xrd.py) ──────────────────────────────

def read_dump_last_frame(filename):
    """Read the last frame from a LAMMPS dump file."""
    positions = []
    types = []
    box = np.zeros(3)

    with open(filename) as f:
        lines = f.readlines()

    last_ts_idx = -1
    for i, line in enumerate(lines):
        if 'ITEM: TIMESTEP' in line:
            last_ts_idx = i

    if last_ts_idx < 0:
        raise ValueError(f"No TIMESTEP found in {filename}")

    i = last_ts_idx
    n_atoms = int(lines[i + 3].strip())

    for dim in range(3):
        parts = lines[i + 5 + dim].split()
        box[dim] = float(parts[1]) - float(parts[0])

    header_line = lines[i + 8].strip()
    cols = header_line.replace("ITEM: ATOMS", "").split()
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
    """Read positions from a LAMMPS data file."""
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


# ─── Steinhardt Order Parameters ────────────────────────────────────────────

def get_neighbors(positions, box, i, cutoff=3.5):
    """Find neighbor indices within cutoff using minimum image convention."""
    dr = positions - positions[i]
    for dim in range(3):
        dr[:, dim] -= box[dim] * np.round(dr[:, dim] / box[dim])
    dists = np.linalg.norm(dr, axis=1)

    # Exclude self
    mask = (dists > 0.1) & (dists < cutoff)
    return np.where(mask)[0], dr[mask], dists[mask]


def compute_qlm(positions, box, i, l, cutoff=3.5):
    """Compute complex q_lm(i) for atom i.

    q_lm(i) = (1/N_b) Σ_j Y_lm(θ_ij, φ_ij)

    where the sum is over N_b neighbors within cutoff.
    """
    neighbors, dr, dists = get_neighbors(positions, box, i, cutoff)
    n_neighbors = len(neighbors)

    if n_neighbors == 0:
        return np.zeros(2 * l + 1, dtype=complex)

    # Convert to spherical coordinates
    r = dists
    theta = np.arccos(np.clip(dr[:, 2] / r, -1, 1))
    phi = np.arctan2(dr[:, 1], dr[:, 0])

    # Compute spherical harmonics for m = -l to l
    qlm = np.zeros(2 * l + 1, dtype=complex)
    for m_idx, m in enumerate(range(-l, l + 1)):
        # scipy sph_harm convention: sph_harm(m, l, phi, theta)
        ylm = sph_harm(m, l, phi, theta)
        qlm[m_idx] = np.mean(ylm)

    return qlm


def compute_ql(positions, box, i, l, cutoff=3.5):
    """Compute rotationally invariant Q_l for atom i.

    Q_l = sqrt( (4π)/(2l+1) Σ_m |q_lm|² )
    """
    qlm = compute_qlm(positions, box, i, l, cutoff)
    ql = np.sqrt(4.0 * np.pi / (2 * l + 1) * np.sum(np.abs(qlm)**2))
    return ql


def compute_all_steinhardt(positions, types, box, si_type=1,
                           cutoff=3.5):
    """Compute Q4 and Q6 for all Si atoms.

    Args:
        positions: (N, 3) atom positions
        types: atom type list
        box: (3,) box lengths
        si_type: atom type for Si
        cutoff: neighbor cutoff (Å) — should capture first Si-Si shell
                (~3.1 Å in cristobalite, using Si-O-Si through O)

    Returns:
        si_indices: indices of Si atoms
        q4_values: Q4 for each Si atom
        q6_values: Q6 for each Si atom
    """
    # Get Si atom indices
    si_indices = [i for i, t in enumerate(types) if t == si_type]
    n_si = len(si_indices)

    # For Steinhardt analysis of SiO2, we look at Si-Si neighbors
    # The Si-Si distance in cristobalite is ~3.1 Å (through O bridge)
    # But for the Steinhardt parameter, we typically look at Si-O neighbors
    # and use the O positions to define bond angles.
    #
    # Alternative: compute Q_l using only Si-Si pairs with a larger cutoff
    # (~5.0 Å to capture the first Si-Si coordination shell)

    # Use Si-only positions with larger cutoff for Si-Si shell
    si_positions = positions[si_indices]
    si_cutoff = 5.5  # First Si-Si coordination shell in SiO2 (~5.1 Å)

    print(f"  Computing Steinhardt Q4, Q6 for {n_si} Si atoms...")
    print(f"  Si-Si cutoff: {si_cutoff:.1f} Å")

    q4_values = np.zeros(n_si)
    q6_values = np.zeros(n_si)

    for idx in range(n_si):
        q4_values[idx] = compute_ql(si_positions, box, idx, l=4,
                                     cutoff=si_cutoff)
        q6_values[idx] = compute_ql(si_positions, box, idx, l=6,
                                     cutoff=si_cutoff)

        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx + 1}/{n_si} Si atoms...")

    return si_indices, q4_values, q6_values


# ─── Phase Classification ───────────────────────────────────────────────────

def classify_phases(q4_values, q6_values):
    """Classify each Si atom based on Q4, Q6 values.

    Uses simple distance-based classification to reference values.

    Returns:
        labels: array of strings ('amorphous', 'cristobalite', 'tridymite')
        percentages: dict with phase percentages
    """
    # Reference points in (Q4, Q6) space
    refs = {
        'amorphous':    (0.03, 0.03),
        'cristobalite': (0.19, 0.57),
        'tridymite':    (0.13, 0.48),
    }

    n = len(q4_values)
    labels = []

    for q4, q6 in zip(q4_values, q6_values):
        # Compute distance to each reference
        dists = {}
        for phase, (ref_q4, ref_q6) in refs.items():
            # Weighted distance (Q6 is more discriminating)
            d = np.sqrt((q4 - ref_q4)**2 + (q6 - ref_q6)**2)
            dists[phase] = d

        # Assign to nearest reference
        nearest = min(dists, key=dists.get)
        labels.append(nearest)

    labels = np.array(labels)
    percentages = {}
    for phase in refs:
        count = np.sum(labels == phase)
        percentages[phase] = 100.0 * count / n

    return labels, percentages


def main():
    parser = argparse.ArgumentParser(
        description="Compute Steinhardt Q4/Q6 and classify silica phases")
    parser.add_argument("input", help="LAMMPS dump or data file")
    parser.add_argument("--format", choices=["dump", "data"], default="dump",
                        help="Input file format (default: dump)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file prefix")
    parser.add_argument("--cutoff", type=float, default=5.5,
                        help="Si-Si neighbor cutoff in Å (default: 5.5)")
    args = parser.parse_args()

    # Read
    if args.format == "dump":
        positions, types, box = read_dump_last_frame(args.input)
    else:
        positions, types, box = read_data_file(args.input)

    print(f"Read {len(positions)} atoms from {args.input}")

    # Compute
    si_indices, q4, q6 = compute_all_steinhardt(
        positions, types, box, cutoff=args.cutoff)

    # Classify
    labels, percentages = classify_phases(q4, q6)

    # Print results
    print(f"\n  Phase Classification Results:")
    print(f"  {'Phase':<15} {'Count':>6} {'Percentage':>10}")
    print(f"  {'-'*35}")
    for phase in ['amorphous', 'cristobalite', 'tridymite']:
        count = np.sum(labels == phase)
        pct = percentages[phase]
        print(f"  {phase:<15} {count:>6} {pct:>9.1f}%")

    # Save
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"steinhardt_{base}"

    # Per-atom data
    np.savetxt(f"{args.output}.dat",
               np.column_stack([si_indices, q4, q6]),
               header="si_index  Q4  Q6",
               fmt="%d  %.6f  %.6f")

    # Summary
    with open(f"{args.output}_summary.txt", 'w') as f:
        f.write(f"Source: {args.input}\n")
        f.write(f"N_Si: {len(si_indices)}\n")
        f.write(f"Q4_mean: {np.mean(q4):.4f} ± {np.std(q4):.4f}\n")
        f.write(f"Q6_mean: {np.mean(q6):.4f} ± {np.std(q6):.4f}\n\n")
        for phase in ['amorphous', 'cristobalite', 'tridymite']:
            f.write(f"{phase}: {percentages[phase]:.1f}%\n")

    print(f"\n  Data saved to {args.output}.dat")
    print(f"  Summary saved to {args.output}_summary.txt")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = {'amorphous': '#888888', 'cristobalite': '#e74c3c',
                  'tridymite': '#2ecc71'}

        for phase in ['amorphous', 'cristobalite', 'tridymite']:
            mask = labels == phase
            ax.scatter(q4[mask], q6[mask], c=colors[phase],
                      label=f"{phase} ({percentages[phase]:.1f}%)",
                      s=15, alpha=0.6, edgecolors='none')

        # Reference points
        refs = {
            'Amorphous': (0.03, 0.03),
            'β-Cristobalite': (0.19, 0.57),
            'Tridymite': (0.13, 0.48),
        }
        for name, (rq4, rq6) in refs.items():
            ax.plot(rq4, rq6, 'k*', markersize=15, zorder=5)
            ax.annotate(name, (rq4, rq6), fontsize=8,
                       xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel("Q₄")
        ax.set_ylabel("Q₆")
        ax.set_title(f"Steinhardt Order Parameters — "
                     f"{os.path.basename(args.input)}")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 0.30)
        ax.set_ylim(-0.02, 0.70)

        fig.tight_layout()
        fig.savefig(f"{args.output}.png", dpi=150)
        print(f"  Plot saved to {args.output}.png")
        plt.close()
    except ImportError:
        print("  matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
