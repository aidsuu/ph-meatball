#!/usr/bin/env python3
"""
Strip hydrogen, water, and free oxygen from sol-gel output to create pure SiO2
for Vashishta calcination simulation.

Reads a LAMMPS data file (ReaxFF output with Si, O, H atoms),
removes all H atoms, any O atoms that were part of water molecules,
AND any completely free O atoms (O atoms with 0 Si neighbors).
reassigns charges for Vashishta potential (Si=+2.4, O=-1.2),
and writes a clean LAMMPS data file.
"""

import argparse
import numpy as np
import os
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

def identify_unbonded_oxygen(atoms, header, si_type=1, o_type=2, sio_cutoff=2.0):
    """
    Identify O atoms to remove:
    We remove ANY O atom that has 0 Si neighbors within sio_cutoff.
    This naturally includes water (H2O), hydronium (H3O+), hydroxyls (OH-)
    not bonded to the silica network, and completely free oxygen atoms.
    """
    box = np.array([
        header['xhi'] - header['xlo'],
        header['yhi'] - header['ylo'],
        header['zhi'] - header['zlo'],
    ])

    positions = {}
    ids_by_type = {}
    for atom in atoms:
        t = atom['type']
        if t not in positions:
            positions[t] = []
            ids_by_type[t] = []
        positions[t].append([atom['x'], atom['y'], atom['z']])
        ids_by_type[t].append(atom['id'])

    for t in positions:
        positions[t] = np.array(positions[t])

    to_remove_o_ids = set()

    if o_type not in positions:
        return to_remove_o_ids

    o_pos = positions[o_type]
    o_ids = ids_by_type[o_type]
    si_pos = positions.get(si_type, np.empty((0, 3)))

    for i, (o, oid) in enumerate(zip(o_pos, o_ids)):
        if len(si_pos) > 0:
            dr_si = si_pos - o
            dr_si -= box * np.round(dr_si / box)
            dist_si = np.linalg.norm(dr_si, axis=1)
            n_si = np.sum(dist_si < sio_cutoff)
        else:
            n_si = 0

        # Remove if not bonded to Si network
        if n_si == 0:
            to_remove_o_ids.add(oid)

    return to_remove_o_ids

def strip_and_convert(atoms, header, masses, to_remove_o_ids, si_type=1, o_type=2, h_type=3):
    kept_atoms = []

    n_removed_h = 0
    n_removed_o = 0

    for atom in atoms:
        if atom['type'] == h_type:
            n_removed_h += 1
            continue
        if atom['type'] == o_type and atom['id'] in to_remove_o_ids:
            n_removed_o += 1
            continue
        kept_atoms.append(atom)

    n_si = sum(1 for a in kept_atoms if a['type'] == si_type)
    n_o = sum(1 for a in kept_atoms if a['type'] == o_type)

    print(f"  Atoms removed:")
    print(f"    H atoms:                   {n_removed_h}")
    print(f"    O atoms (not bonded to Si): {n_removed_o}")
    print(f"  Remaining atoms:")
    print(f"    Si: {n_si}")
    print(f"    O:  {n_o}")
    if n_si > 0:
        print(f"    O/Si ratio: {n_o/n_si:.2f} (ideal SiO2 = 2.00)")

    new_masses = {1: 28.0855, 2: 15.9994}

    new_atoms = []
    for i, atom in enumerate(kept_atoms):
        new_type = 1 if atom['type'] == si_type else 2
        new_atoms.append({
            'id': i + 1,
            'type': new_type,
            'charge': 0.0,
            'x': atom['x'],
            'y': atom['y'],
            'z': atom['z'],
        })

    new_header = dict(header)
    new_header['natoms'] = len(new_atoms)
    new_header['ntypes'] = 2

    return new_atoms, new_header, new_masses

def write_lammps_data(filename, atoms, header, masses, comment=""):
    with open(filename, 'w') as f:
        f.write(f"LAMMPS data file - {comment}\n\n")
        f.write(f"{header['natoms']} atoms\n")
        f.write(f"{header['ntypes']} atom types\n\n")
        f.write(f"{header['xlo']:.6f} {header['xhi']:.6f} xlo xhi\n")
        f.write(f"{header['ylo']:.6f} {header['yhi']:.6f} ylo yhi\n")
        f.write(f"{header['zlo']:.6f} {header['zhi']:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for t, m in sorted(masses.items()):
            label = {1: "Si", 2: "O"}.get(t, "?")
            f.write(f"{t} {m:.4f}  # {label}\n")
        f.write("\nAtoms # charge\n\n")
        for atom in atoms:
            f.write(f"{atom['id']} {atom['type']} {atom['charge']:.4f} "
                    f"{atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")

    print(f"  Written {header['natoms']} atoms to {filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Strip H/H2O/free O from ReaxFF output, convert for Vashishta")
    parser.add_argument("input", help="Input LAMMPS data file (ReaxFF output)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output LAMMPS data file (default: auto-named)")
    parser.add_argument("--sio-cutoff", type=float, default=2.0,
                        help="Si-O bond cutoff in Angstrom (default: 2.0)")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        base = base.replace("solgel", "vashishta").replace("data.", "data.vash_freeO_")
        if base == os.path.splitext(args.input)[0]:
            base = base + ".vash_freeO_stripped"
        args.output = base

    print(f"\n{'='*60}")
    print(f"  Stripping hydrogen and unbonded oxygen from: {args.input}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")

    header, atoms, masses = read_lammps_data(args.input)
    print(f"\n  Input: {header['natoms']} atoms, {header['ntypes']} types")

    to_remove_ids = identify_unbonded_oxygen(atoms, header, sio_cutoff=args.sio_cutoff)
    print(f"  Unbonded O atoms (including water) identified: {len(to_remove_ids)}")

    new_atoms, new_header, new_masses = strip_and_convert(
        atoms, header, masses, to_remove_ids)

    ph_label = ""
    if "pH" in args.input:
        for part in args.input.split("pH"):
            if len(part) > 1:
                ph_label = part.split(".")[0].replace("_", "")
    comment = f"Pure SiO2 from sol-gel (pH {ph_label}), stripped free O, Vashishta format"
    write_lammps_data(args.output, new_atoms, new_header, new_masses, comment=comment)

    print(f"\n{'='*60}")
    print(f"  Conversion complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
