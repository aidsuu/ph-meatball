#!/usr/bin/env python3
"""
Generate PLUMED NDX (index) file for Si atoms from a LAMMPS data file.

PLUMED needs to know which atoms are Si to compute Steinhardt order
parameters only on the Si sub-lattice.

Usage:
    python gen_plumed_ndx.py data.vash_pH7.0
    # Creates si_atoms.ndx in the same directory
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate PLUMED NDX file for Si atoms")
    parser.add_argument("input", help="LAMMPS data file")
    parser.add_argument("--output", "-o", default="si_atoms.ndx",
                        help="Output NDX file (default: si_atoms.ndx)")
    parser.add_argument("--si-type", type=int, default=1,
                        help="Atom type for Si (default: 1)")
    args = parser.parse_args()

    si_ids = []

    with open(args.input) as f:
        in_atoms = False
        for line in f:
            line = line.strip()
            if line.startswith("Atoms"):
                in_atoms = True
                continue
            if in_atoms and line:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        atom_id = int(parts[0])
                        atom_type = int(parts[1])
                        if atom_type == args.si_type:
                            si_ids.append(atom_id)
                    except ValueError:
                        if parts[0] in ('Velocities', 'Bonds', 'Angles'):
                            break

    print(f"Found {len(si_ids)} Si atoms (type {args.si_type})")

    # Write NDX file (GROMACS-style index file for PLUMED)
    with open(args.output, 'w') as f:
        f.write("[ si ]\n")
        for i, aid in enumerate(si_ids):
            f.write(f"{aid}")
            if (i + 1) % 15 == 0:
                f.write("\n")
            else:
                f.write(" ")
        f.write("\n")

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
