#!/usr/bin/env python3
"""
Convert pt3/sio2_amorf.data (Tersoff output) to Vashishta-compatible format.

Changes:
  - Set charges: type 1 (Si) = +2.4, type 2 (O) = -1.2
  - Verify box & atom counts
  - Output to data.vash_pH7.0 in simulations/ directory

Usage:
    python3 convert_to_vashishta.py
"""
import os, sys

# === Paths ===
PT3 = "/data/mahasiswa/adi/nope/pt3/sio2_amorf.data"
OUT_DIR = "/data/mahasiswa/adi/nope/silica/simulations"
OUT = os.path.join(OUT_DIR, "data.vash_pH7.0")

# Vashishta 1990 formal charges for SiO2
SI_CHARGE = 2.4
O_CHARGE = -1.2

def main():
    if not os.path.exists(PT3):
        sys.exit(f"ERROR: input not found: {PT3}")

    with open(PT3) as f:
        lines = f.readlines()

    # Find Atoms section — LAMMPS "Atoms # charge" has 1 comment line after
    atoms_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            # Next line is a comment (e.g. " # charge"); data starts after
            atoms_start = i + 2
            break

    if atoms_start is None:
        sys.exit("ERROR: 'Atoms' section not found")

    # Verify atom count
    natoms_line = next(l for l in lines if l.strip().endswith("atoms"))
    natoms = int(natoms_line.split()[0])
    print(f"  Atoms: {natoms}")

    # Set charges per atom — pt3 columns: id type charge x y z ix iy iz
    type_count = {1: 0, 2: 0}
    out_atoms = []
    for i in range(atoms_start, atoms_start + natoms):
        parts = lines[i].split()
        atom_id = int(parts[0])
        atom_type = int(parts[1])
        old_charge = float(parts[2])
        x, y, z = parts[3], parts[4], parts[5]
        # preserve image flags if present
        image_flags = " ".join(parts[6:]) if len(parts) > 6 else ""

        new_charge = SI_CHARGE if atom_type == 1 else O_CHARGE
        type_count[atom_type] += 1

        # atom-id type charge x y z [image flags]
        out_atoms.append(f"{atom_id} {atom_type} {new_charge:.4f} {x} {y} {z} {image_flags}\n".rstrip() + "\n")

    # Verify stoichiometry
    si_count = type_count[1]
    o_count = type_count[2]
    expected_o = si_count * 2
    if abs(o_count - expected_o) > si_count * 0.1:
        print(f"  WARNING: O count ({o_count}) not 2x Si ({si_count})")
    else:
        print(f"  Stoichiometry OK: {si_count} Si, {o_count} O (SiO2)")

    # Write output — preserve header, replace Atoms section
    out_lines = lines[:atoms_start] + out_atoms + lines[atoms_start + natoms:]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        f.writelines(out_lines)

    print(f"  Wrote: {OUT}")
    print(f"  Charges set: Si={SI_CHARGE}, O={O_CHARGE}")

if __name__ == "__main__":
    main()
