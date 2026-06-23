#!/usr/bin/env python3
"""
Strip hydrogen and water from sol-gel output to create pure SiO2
for Vashishta calcination simulation.

Reads a LAMMPS data file (ReaxFF output with Si, O, H atoms),
removes all H atoms and any O atoms that were part of water molecules,
reassigns charges for Vashishta potential (Si=+2.4, O=-1.2),
and writes a clean LAMMPS data file.

Usage:
    python strip_hydrogen.py data.solgel_pH7.0.final
    python strip_hydrogen.py data.solgel_pH7.0.final --output data.vash_pH7.0
"""

import argparse
import numpy as np
import os
import sys


def read_lammps_data(filename):
    """Read a LAMMPS data file (atom_style charge).

    Returns:
        header: dict with metadata (natoms, ntypes, box bounds)
        atoms: list of dicts with {id, type, charge, x, y, z}
        masses: dict {type_id: mass}
    """
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

        # Parse header
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

        # Detect sections
        if line.startswith('Masses'):
            section = 'masses'
            continue
        elif line.startswith('Atoms'):
            section = 'atoms'
            continue

        # Parse sections
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


def identify_water(atoms, header, si_type=1, o_type=2, h_type=3,
                   oh_cutoff=1.2, sio_cutoff=2.0):
    """Identify water molecules (O atoms bonded to 2H and no Si).

    An O atom is classified as "water oxygen" if:
    1. It has 2 or more H neighbors within oh_cutoff
    2. It has 0 Si neighbors within sio_cutoff

    Args:
        atoms: list of atom dicts
        header: dict with box dimensions
        si_type, o_type, h_type: atom type indices
        oh_cutoff: O-H bond cutoff (Angstrom)
        sio_cutoff: Si-O bond cutoff (Angstrom)

    Returns:
        water_o_ids: set of atom IDs that are water oxygens
    """
    # Build position arrays by type
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

    water_o_ids = set()

    if o_type not in positions:
        return water_o_ids

    o_pos = positions[o_type]
    o_ids = ids_by_type[o_type]

    h_pos = positions.get(h_type, np.empty((0, 3)))
    si_pos = positions.get(si_type, np.empty((0, 3)))

    for i, (o, oid) in enumerate(zip(o_pos, o_ids)):
        # Count H neighbors
        if len(h_pos) > 0:
            dr_h = h_pos - o
            dr_h -= box * np.round(dr_h / box)
            dist_h = np.linalg.norm(dr_h, axis=1)
            n_h = np.sum(dist_h < oh_cutoff)
        else:
            n_h = 0

        # Count Si neighbors
        if len(si_pos) > 0:
            dr_si = si_pos - o
            dr_si -= box * np.round(dr_si / box)
            dist_si = np.linalg.norm(dr_si, axis=1)
            n_si = np.sum(dist_si < sio_cutoff)
        else:
            n_si = 0

        # Water: 2 H neighbors, 0 Si neighbors
        if n_h >= 2 and n_si == 0:
            water_o_ids.add(oid)

    return water_o_ids


def strip_and_convert(atoms, header, masses, water_o_ids,
                      si_type=1, o_type=2, h_type=3):
    """Remove all H atoms and water O atoms, convert to Vashishta format.

    For Vashishta:
        Type 1 = Si (charge +2.4 |e|, but Vashishta handles charges internally)
        Type 2 = O  (charge -1.2 |e|, but Vashishta handles charges internally)
        Charges in data file are set to 0.0 (Vashishta computes its own)
    """
    kept_atoms = []

    n_removed_h = 0
    n_removed_water_o = 0
    n_removed_other_o = 0

    for atom in atoms:
        if atom['type'] == h_type:
            n_removed_h += 1
            continue
        if atom['type'] == o_type and atom['id'] in water_o_ids:
            n_removed_water_o += 1
            continue
        kept_atoms.append(atom)

    # Check stoichiometry
    n_si = sum(1 for a in kept_atoms if a['type'] == si_type)
    n_o = sum(1 for a in kept_atoms if a['type'] == o_type)

    print(f"  Atoms removed:")
    print(f"    H atoms:          {n_removed_h}")
    print(f"    Water O atoms:    {n_removed_water_o}")
    print(f"  Remaining atoms:")
    print(f"    Si: {n_si}")
    print(f"    O:  {n_o}")
    print(f"    O/Si ratio: {n_o/n_si:.2f} (ideal SiO2 = 2.00)")

    # Check for dangling O (non-bridging O from former Si-OH)
    # These are fine — they represent structural defects from the sol-gel process

    # Remap types: Si=1, O=2 (remove H type)
    new_masses = {1: 28.0855, 2: 15.9994}

    # Renumber atoms and set charges to 0.0 (Vashishta)
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
    """Write a LAMMPS data file (atom_style charge)."""
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
        description="Strip H/H2O from ReaxFF output, convert for Vashishta")
    parser.add_argument("input", help="Input LAMMPS data file (ReaxFF output)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output LAMMPS data file (default: auto-named)")
    parser.add_argument("--oh-cutoff", type=float, default=1.2,
                        help="O-H bond cutoff in Angstrom (default: 1.2)")
    parser.add_argument("--sio-cutoff", type=float, default=2.0,
                        help="Si-O bond cutoff in Angstrom (default: 2.0)")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        base = base.replace("solgel", "vashishta").replace("data.", "data.vash_")
        if base == os.path.splitext(args.input)[0]:
            base = base + ".vashishta"
        args.output = base

    print(f"\n{'='*60}")
    print(f"  Stripping hydrogen from: {args.input}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")

    # Read
    header, atoms, masses = read_lammps_data(args.input)
    print(f"\n  Input: {header['natoms']} atoms, {header['ntypes']} types")

    # Identify water
    water_ids = identify_water(atoms, header,
                               oh_cutoff=args.oh_cutoff,
                               sio_cutoff=args.sio_cutoff)
    print(f"  Water molecules identified: {len(water_ids)}")

    # Strip and convert
    new_atoms, new_header, new_masses = strip_and_convert(
        atoms, header, masses, water_ids)

    # Write
    ph_label = ""
    if "pH" in args.input:
        for part in args.input.split("pH"):
            if len(part) > 1:
                ph_label = part.split(".")[0].replace("_", "")
    comment = f"Pure SiO2 from sol-gel (pH {ph_label}), Vashishta format"
    write_lammps_data(args.output, new_atoms, new_header, new_masses,
                      comment=comment)

    print(f"\n{'='*60}")
    print(f"  Conversion complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
