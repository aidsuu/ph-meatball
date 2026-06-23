#!/usr/bin/env python3
"""
Ring-size distribution analysis for SiO2 network.

Counts primitive rings in the Si-O-Si network using shortest-path
ring finding. Crystalline phases (cristobalite, tridymite) are dominated
by 6-membered rings, while amorphous silica has a broad distribution
(3-12 membered rings).

A "ring" here counts the number of Si atoms in a closed Si-O-Si-O-...-Si path.

Usage:
    python ring_analysis.py dump.calc_pH7.0.final.lammpstrj
    python ring_analysis.py data.calc_pH7.0.final --format data
"""

import argparse
import numpy as np
from collections import defaultdict, deque
import os
import sys


# ─── File Readers ────────────────────────────────────────────────────────────

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


# ─── Network Building ───────────────────────────────────────────────────────

def build_sio_network(positions, types, box, si_type=1, o_type=2,
                      sio_cutoff=2.0):
    """Build the Si-O bonding network.

    Returns:
        si_neighbors: dict mapping Si index → list of neighboring Si indices
                      (connected through bridging O atoms)
        si_indices: list of Si atom indices
        o_indices: list of O atom indices
    """
    si_indices = [i for i, t in enumerate(types) if t == si_type]
    o_indices = [i for i, t in enumerate(types) if t == o_type]

    n_si = len(si_indices)
    n_o = len(o_indices)

    print(f"  Building Si-O network: {n_si} Si, {n_o} O atoms")
    print(f"  Si-O cutoff: {sio_cutoff:.2f} Å")

    # For each O atom, find which Si atoms it bridges
    o_to_si = {}  # O index → list of bonded Si indices

    for o_idx in o_indices:
        o_pos = positions[o_idx]
        bonded_si = []

        for si_idx in si_indices:
            dr = positions[si_idx] - o_pos
            for dim in range(3):
                dr[dim] -= box[dim] * round(dr[dim] / box[dim])
            dist = np.linalg.norm(dr)

            if dist < sio_cutoff:
                bonded_si.append(si_idx)

        if len(bonded_si) == 2:
            # Bridging oxygen → connects two Si atoms
            o_to_si[o_idx] = bonded_si

    # Build Si-Si adjacency through bridging O
    si_neighbors = defaultdict(set)
    for o_idx, si_pair in o_to_si.items():
        si_neighbors[si_pair[0]].add(si_pair[1])
        si_neighbors[si_pair[1]].add(si_pair[0])

    n_bridging = len(o_to_si)
    n_connected = sum(1 for s in si_indices if len(si_neighbors[s]) > 0)
    print(f"  Bridging O atoms: {n_bridging}")
    print(f"  Connected Si atoms: {n_connected}/{n_si}")

    return dict(si_neighbors), si_indices


# ─── Ring Finding (Shortest-Path) ───────────────────────────────────────────

def find_shortest_rings(si_neighbors, si_indices, max_ring_size=12):
    """Find shortest-path rings in the Si-Si network.

    Uses BFS from each Si atom to find the shortest cycle that
    passes through each of its bonds.

    A ring of size n means n Si atoms connected in a cycle through
    bridging O atoms.

    Returns:
        ring_counts: dict {ring_size: count}
    """
    ring_counts = defaultdict(int)
    found_rings = set()  # Track unique rings (as frozensets)

    n_si = len(si_indices)
    si_set = set(si_indices)

    print(f"  Finding rings (max size {max_ring_size})...")

    for start_idx, start in enumerate(si_indices):
        if start not in si_neighbors:
            continue

        neighbors = si_neighbors[start]

        for neighbor in neighbors:
            # BFS to find shortest path from neighbor back to start
            # without using the direct start-neighbor edge
            visited = {start, neighbor}
            queue = deque()

            # Initialize with neighbor's neighbors (excluding start)
            for nn in si_neighbors.get(neighbor, set()):
                if nn != start and nn in si_set:
                    queue.append((nn, [neighbor, nn]))
                    visited.add(nn)

            ring_found = False
            while queue and not ring_found:
                current, path = queue.popleft()

                if len(path) >= max_ring_size:
                    continue

                for next_node in si_neighbors.get(current, set()):
                    if next_node == start and len(path) >= 2:
                        # Found a ring
                        ring = frozenset(path)
                        if ring not in found_rings:
                            found_rings.add(ring)
                            ring_size = len(path) + 1  # +1 for start atom
                            ring_counts[ring_size] += 1
                        ring_found = True
                        break
                    elif next_node not in visited and next_node in si_set:
                        visited.add(next_node)
                        queue.append((next_node, path + [next_node]))

        if (start_idx + 1) % 100 == 0:
            print(f"    Processed {start_idx + 1}/{n_si} Si atoms...")

    return dict(ring_counts)


def main():
    parser = argparse.ArgumentParser(
        description="Ring-size distribution analysis for SiO2 network")
    parser.add_argument("input", help="LAMMPS dump or data file")
    parser.add_argument("--format", choices=["dump", "data"], default="dump",
                        help="Input file format (default: dump)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file prefix")
    parser.add_argument("--sio-cutoff", type=float, default=2.0,
                        help="Si-O bond cutoff in Å (default: 2.0)")
    parser.add_argument("--max-ring", type=int, default=12,
                        help="Maximum ring size to search (default: 12)")
    args = parser.parse_args()

    # Read
    if args.format == "dump":
        positions, types, box = read_dump_last_frame(args.input)
    else:
        positions, types, box = read_data_file(args.input)

    print(f"Read {len(positions)} atoms from {args.input}")

    # Build network
    si_neighbors, si_indices = build_sio_network(
        positions, types, box, sio_cutoff=args.sio_cutoff)

    # Find rings
    ring_counts = find_shortest_rings(
        si_neighbors, si_indices, max_ring_size=args.max_ring)

    # Print results
    total_rings = sum(ring_counts.values())
    print(f"\n  Ring-Size Distribution:")
    print(f"  {'Size':>6} {'Count':>8} {'Fraction':>10}")
    print(f"  {'-'*28}")
    for size in sorted(ring_counts.keys()):
        count = ring_counts[size]
        frac = count / total_rings if total_rings > 0 else 0
        print(f"  {size:>6} {count:>8} {frac:>9.3f}")
    print(f"  {'Total':>6} {total_rings:>8}")

    # Crystallinity metric: fraction of 6-membered rings
    n_6ring = ring_counts.get(6, 0)
    crystallinity = n_6ring / total_rings if total_rings > 0 else 0
    print(f"\n  6-ring fraction (crystallinity indicator): {crystallinity:.3f}")

    # Save
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"rings_{base}"

    with open(f"{args.output}.dat", 'w') as f:
        f.write("# Ring-size distribution\n")
        f.write("# ring_size  count  fraction\n")
        for size in range(2, args.max_ring + 1):
            count = ring_counts.get(size, 0)
            frac = count / total_rings if total_rings > 0 else 0
            f.write(f"{size}  {count}  {frac:.6f}\n")

    with open(f"{args.output}_summary.txt", 'w') as f:
        f.write(f"Source: {args.input}\n")
        f.write(f"Total rings: {total_rings}\n")
        f.write(f"6-ring fraction: {crystallinity:.4f}\n")
        f.write(f"Mean ring size: "
                f"{sum(s*c for s,c in ring_counts.items())/total_rings:.2f}\n"
                if total_rings > 0 else "Mean ring size: N/A\n")

    print(f"  Data saved to {args.output}.dat")

    # Plot
    try:
        import matplotlib.pyplot as plt

        sizes = list(range(2, args.max_ring + 1))
        counts = [ring_counts.get(s, 0) for s in sizes]
        fracs = [c / total_rings if total_rings > 0 else 0 for c in counts]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(sizes, fracs, color='steelblue', edgecolor='navy',
                      alpha=0.8)

        # Highlight 6-membered rings
        if 6 in ring_counts:
            idx_6 = sizes.index(6)
            bars[idx_6].set_color('#e74c3c')
            bars[idx_6].set_edgecolor('darkred')

        ax.set_xlabel("Ring Size (number of Si atoms)")
        ax.set_ylabel("Fraction")
        ax.set_title(f"Ring-Size Distribution — "
                     f"{os.path.basename(args.input)}")
        ax.set_xticks(sizes)

        # Annotate crystallinity
        ax.text(0.95, 0.95,
                f"6-ring fraction: {crystallinity:.3f}",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

        fig.tight_layout()
        fig.savefig(f"{args.output}.png", dpi=150)
        print(f"  Plot saved to {args.output}.png")
        plt.close()
    except ImportError:
        print("  matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
