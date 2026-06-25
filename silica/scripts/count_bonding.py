#!/usr/bin/env python3
"""
Count atom bonding types in LAMMPS data file.
Calculates SiO2, SiO3, H2O, free oxygen, etc.
Works for files before and after hydrogen stripping.
"""

import argparse
import numpy as np
import sys

def read_lammps_data(filename):
    header = {}
    atoms = []
    masses = {}

    with open(filename) as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if 'atoms' in line and 'atom' not in line.lower().split()[0]:
            header['natoms'] = int(line.split()[0])
        elif 'atom types' in line:
            header['ntypes'] = int(line.split()[0])
        elif 'xlo xhi' in line:
            parts = line.split()
            header['xlo'] = float(parts[0])
            header['xhi'] = float(parts[1])
        elif 'ylo yhi' in line:
            parts = line.split()
            header['ylo'] = float(parts[0])
            header['yhi'] = float(parts[1])
        elif 'zlo zhi' in line:
            parts = line.split()
            header['zlo'] = float(parts[0])
            header['zhi'] = float(parts[1])

        if line.startswith('Masses'):
            section = 'masses'
            continue
        elif line.startswith('Atoms'):
            section = 'atoms'
            continue

        if section == 'masses':
            parts = line.split()
            if len(parts) >= 2:
                try:
                    masses[int(parts[0])] = float(parts[1])
                except ValueError:
                    pass
        elif section == 'atoms':
            parts = line.split()
            if len(parts) >= 6:
                try:
                    atoms.append({
                        'id': int(parts[0]),
                        'type': int(parts[1]),
                        'charge': float(parts[2]),
                        'x': float(parts[3]),
                        'y': float(parts[4]),
                        'z': float(parts[5]),
                    })
                except ValueError:
                    pass

    return header, atoms, masses

def count_bonding(atoms, header, si_type=1, o_type=2, h_type=3, oh_cutoff=1.2, sio_cutoff=2.0):
    box = np.array([
        header['xhi'] - header['xlo'],
        header['yhi'] - header['ylo'],
        header['zhi'] - header['zlo'],
    ])

    positions = {}
    for atom in atoms:
        t = atom['type']
        if t not in positions:
            positions[t] = []
        positions[t].append([atom['x'], atom['y'], atom['z']])

    for t in positions:
        positions[t] = np.array(positions[t])

    si_pos = positions.get(si_type, np.empty((0, 3)))
    o_pos = positions.get(o_type, np.empty((0, 3)))
    h_pos = positions.get(h_type, np.empty((0, 3)))
    
    print(f"Total atoms: Si={len(si_pos)}, O={len(o_pos)}, H={len(h_pos)}")

    # Count Si environments
    si_counts = []
    for si in si_pos:
        if len(o_pos) > 0:
            dr = o_pos - si
            dr -= box * np.round(dr / box)
            dist = np.linalg.norm(dr, axis=1)
            n_o = np.sum(dist < sio_cutoff)
        else:
            n_o = 0
        si_counts.append(n_o)

    # Count O environments
    o_si_counts = []
    o_h_counts = []
    for o in o_pos:
        if len(si_pos) > 0:
            dr = si_pos - o
            dr -= box * np.round(dr / box)
            dist = np.linalg.norm(dr, axis=1)
            n_si = np.sum(dist < sio_cutoff)
        else:
            n_si = 0
            
        if len(h_pos) > 0:
            dr = h_pos - o
            dr -= box * np.round(dr / box)
            dist = np.linalg.norm(dr, axis=1)
            n_h = np.sum(dist < oh_cutoff)
        else:
            n_h = 0
            
        o_si_counts.append(n_si)
        o_h_counts.append(n_h)

    print("\n--- Silicon Coordinations (O neighbors) ---")
    for i in range(10):
        c = si_counts.count(i)
        if c > 0:
            print(f"  Si with {i} O (SiO{i}): {c}")

    print("\n--- Oxygen Environments (Si neighbors, H neighbors) ---")
    env_counts = {}
    for si_c, h_c in zip(o_si_counts, o_h_counts):
        pair = (si_c, h_c)
        env_counts[pair] = env_counts.get(pair, 0) + 1
        
    for (si_c, h_c), count in sorted(env_counts.items()):
        name = ""
        if si_c == 0 and h_c >= 2: name = " (Water / H2O)"
        elif si_c == 0 and h_c == 1: name = " (Hydroxyl / OH-)"
        elif si_c == 0 and h_c == 0: name = " (Completely Free Oxygen)"
        elif si_c == 1 and h_c == 1: name = " (Silanol / Si-OH)"
        elif si_c == 1 and h_c == 0: name = " (Dangling / Non-bridging Oxygen)"
        elif si_c == 2 and h_c == 0: name = " (Bridging Oxygen)"
        elif si_c >= 3: name = " (Over-coordinated Oxygen)"
        
        print(f"  O with {si_c} Si and {h_c} H: {count}{name}")

    print("\n--- Molecular Fragments (Silica Polymerization) ---")
    # Build a graph of Si atoms connected by bridging O atoms
    adj = {i: set() for i in range(len(si_pos))}
    
    for o in o_pos:
        if len(si_pos) > 0:
            dr = si_pos - o
            dr -= box * np.round(dr / box)
            dist = np.linalg.norm(dr, axis=1)
            si_neighbors = np.where(dist < sio_cutoff)[0]
            if len(si_neighbors) >= 2:
                # Add edges between all pairs of Si neighbors of this O
                for i in range(len(si_neighbors)):
                    for j in range(i+1, len(si_neighbors)):
                        adj[si_neighbors[i]].add(si_neighbors[j])
                        adj[si_neighbors[j]].add(si_neighbors[i])

    # Find connected components
    visited = set()
    cluster_sizes = []
    
    for i in range(len(si_pos)):
        if i not in visited:
            size = 0
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                size += 1
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            cluster_sizes.append(size)
            
    if not cluster_sizes:
        print("  No Silicon atoms found.")
    else:
        size_counts = {}
        for size in cluster_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
            
        for size, count in sorted(size_counts.items()):
            name = ""
            if size == 1: name = " (Monomer, e.g. Si(OH)4)"
            elif size == 2: name = " (Dimer)"
            elif size == 3: name = " (Trimer)"
            elif size == 4: name = " (Tetramer)"
            elif size > 4: name = " (Polymer network)"
            
            print(f"  Cluster of {size} Si atoms: {count}{name}")

def main():
    parser = argparse.ArgumentParser(description="Count atom bonding types in LAMMPS data file")
    parser.add_argument("input", help="Input LAMMPS data file")
    parser.add_argument("--si-type", type=int, default=1, help="Atom type for Si (default: 1)")
    parser.add_argument("--o-type", type=int, default=2, help="Atom type for O (default: 2)")
    parser.add_argument("--h-type", type=int, default=3, help="Atom type for H (default: 3)")
    parser.add_argument("--oh-cutoff", type=float, default=1.2, help="O-H bond cutoff in Angstrom (default: 1.2)")
    parser.add_argument("--sio-cutoff", type=float, default=2.0, help="Si-O bond cutoff in Angstrom (default: 2.0)")
    args = parser.parse_args()

    print(f"\nAnalyzing: {args.input}")
    header, atoms, masses = read_lammps_data(args.input)
    count_bonding(atoms, header, si_type=args.si_type, o_type=args.o_type, h_type=args.h_type,
                  oh_cutoff=args.oh_cutoff, sio_cutoff=args.sio_cutoff)
    print("")

if __name__ == "__main__":
    main()
