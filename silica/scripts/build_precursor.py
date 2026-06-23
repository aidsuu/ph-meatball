#!/usr/bin/env python3
"""
Build precursor systems for sol-gel simulation.

Creates Si(OH)4 + H2O periodic boxes with varying silanol deprotonation
levels to represent different pH conditions (6.0 - 8.0).

Usage:
    python build_precursor.py                    # Build all 5 pH conditions
    python build_precursor.py --ph 7.0           # Build single pH condition
    python build_precursor.py --n-silica 200     # Larger system

Output:
    ../simulations/data.pH_6.0  (etc.)
"""

import argparse
import numpy as np
import os
import sys


# ─── Molecule Geometries ────────────────────────────────────────────────────

def make_sioh4():
    """Create a single Si(OH)4 molecule (tetrahedral geometry).

    Returns:
        positions: (9, 3) array — [Si, O1, O2, O3, O4, H1, H2, H3, H4]
        types:     list of atom type indices (1=Si, 2=O, 3=H)
    """
    d_sio = 1.63   # Si-O bond length (Angstrom)
    d_oh = 0.96    # O-H bond length (Angstrom)
    angle_sioh = 115.0  # Si-O-H angle (degrees)

    # Si at origin
    si = np.array([0.0, 0.0, 0.0])

    # 4 O atoms at tetrahedral positions
    tet_dirs = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float)
    tet_dirs = tet_dirs / np.linalg.norm(tet_dirs[0])
    o_atoms = tet_dirs * d_sio

    # H atoms bonded to O, with Si-O-H angle
    h_atoms = []
    angle_rad = np.radians(angle_sioh)
    for i, o in enumerate(o_atoms):
        # Direction from O back to Si
        si_dir = -o / np.linalg.norm(o)
        # Find a perpendicular vector
        perp = np.cross(si_dir, [0, 0, 1])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(si_dir, [0, 1, 0])
        perp = perp / np.linalg.norm(perp)
        # H position: rotate si_dir by (180 - angle) around perp
        rot_angle = np.pi - angle_rad
        # Rodrigues rotation
        h_dir = (si_dir * np.cos(rot_angle)
                 + np.cross(perp, si_dir) * np.sin(rot_angle)
                 + perp * np.dot(perp, si_dir) * (1 - np.cos(rot_angle)))
        h = o + h_dir * d_oh
        h_atoms.append(h)

    positions = np.vstack([si, o_atoms, np.array(h_atoms)])
    types = [1, 2, 2, 2, 2, 3, 3, 3, 3]

    return positions, types


def make_sioh3_ominus():
    """Create Si(OH)3(O-) — one silanol deprotonated.

    Same as Si(OH)4 but with the 4th H removed.
    Returns (8, 3) positions and 8-element type list.
    """
    pos, types = make_sioh4()
    # Remove the last H atom (index 8 = H bonded to O4)
    pos = pos[:8]
    types = types[:8]
    return pos, types


def make_water():
    """Create a single H2O molecule.

    Returns:
        positions: (3, 3) array — [O, H1, H2]
        types:     list of atom type indices (2=O, 3=H)
    """
    d_oh = 0.96
    angle = np.radians(104.52)

    o = np.array([0.0, 0.0, 0.0])
    h1 = np.array([d_oh * np.sin(angle / 2), d_oh * np.cos(angle / 2), 0.0])
    h2 = np.array([-d_oh * np.sin(angle / 2), d_oh * np.cos(angle / 2), 0.0])

    positions = np.vstack([o, h1, h2])
    types = [2, 3, 3]

    return positions, types


def make_hydronium():
    """Create H3O+ molecule (for charge compensation of deprotonated silanols).

    Trigonal pyramidal geometry.
    Returns (4, 3) positions and type list.
    """
    d_oh = 0.98
    angle = np.radians(113.0)  # H-O-H angle in H3O+

    o = np.array([0.0, 0.0, 0.0])
    # 3 H atoms in trigonal arrangement, tilted from the plane
    tilt = np.radians(20.0)  # tilt from xy-plane
    h_pos = []
    for k in range(3):
        phi = k * 2 * np.pi / 3
        h = np.array([
            d_oh * np.cos(tilt) * np.cos(phi),
            d_oh * np.cos(tilt) * np.sin(phi),
            d_oh * np.sin(tilt)
        ])
        h_pos.append(h)

    positions = np.vstack([o, np.array(h_pos)])
    types = [2, 3, 3, 3]

    return positions, types


# ─── Random Rotation ────────────────────────────────────────────────────────

def random_rotation_matrix(rng):
    """Generate a random 3D rotation matrix."""
    # Uniform random quaternion
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    # Quaternion to rotation matrix
    R = np.array([
        [1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
        [2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[0]*q[1])],
        [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)],
    ])
    return R


# ─── Packing ────────────────────────────────────────────────────────────────

def pack_molecules(molecules, box_length, rng, min_dist=2.0, max_attempts=5000):
    """Pack molecules randomly into a periodic cubic box.

    Args:
        molecules: list of (positions, types) tuples for each molecule
        box_length: side length of the cubic box (Angstrom)
        rng: numpy random generator
        min_dist: minimum distance between atoms of different molecules
        max_attempts: max placement attempts per molecule

    Returns:
        all_positions: (N, 3) array of all atom positions
        all_types: list of atom type indices
        success: bool
    """
    all_positions = []
    all_types = []
    placed_centers = []  # centers of mass for quick pre-check

    for mol_idx, (mol_pos, mol_types) in enumerate(molecules):
        placed = False
        mol_radius = np.max(np.linalg.norm(mol_pos - mol_pos.mean(axis=0), axis=1))

        for attempt in range(max_attempts):
            # Random position and orientation
            center = rng.random(3) * box_length
            R = random_rotation_matrix(rng)
            new_pos = (mol_pos - mol_pos.mean(axis=0)) @ R.T + center

            # Wrap into box
            new_pos = new_pos % box_length

            # Quick center-of-mass distance check
            too_close = False
            for prev_center, prev_radius in placed_centers:
                dr = new_pos.mean(axis=0) - prev_center
                # Minimum image convention
                dr = dr - box_length * np.round(dr / box_length)
                dist = np.linalg.norm(dr)
                if dist < (mol_radius + prev_radius + min_dist * 0.5):
                    too_close = True
                    break

            if too_close:
                continue

            # Detailed atom-atom distance check (only if passed quick check)
            if len(all_positions) > 0:
                existing = np.array(all_positions)
                overlap = False
                for atom_pos in new_pos:
                    dr = existing - atom_pos
                    dr = dr - box_length * np.round(dr / box_length)
                    dists = np.linalg.norm(dr, axis=1)
                    if np.any(dists < min_dist):
                        overlap = True
                        break
                if overlap:
                    continue

            # Place the molecule
            all_positions.extend(new_pos.tolist())
            all_types.extend(mol_types)
            placed_centers.append((new_pos.mean(axis=0), mol_radius))
            placed = True

            if (mol_idx + 1) % 50 == 0:
                print(f"  Placed {mol_idx + 1}/{len(molecules)} molecules...")
            break

        if not placed:
            print(f"  WARNING: Could not place molecule {mol_idx + 1} after "
                  f"{max_attempts} attempts. Try a larger box or lower density.")
            return np.array(all_positions), all_types, False

    return np.array(all_positions), all_types, True


# ─── LAMMPS Data File Writer ────────────────────────────────────────────────

def write_lammps_data(filename, positions, types, box_length,
                      masses=None, comment=""):
    """Write LAMMPS data file (atom_style charge).

    For ReaxFF, initial charges are set to 0.0 (QEq will equilibrate).
    """
    if masses is None:
        masses = {1: 28.0855, 2: 15.9994, 3: 1.00794}

    n_atoms = len(types)
    n_types = len(masses)

    with open(filename, 'w') as f:
        f.write(f"LAMMPS data file - {comment}\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_types} atom types\n\n")
        f.write(f"0.0 {box_length:.6f} xlo xhi\n")
        f.write(f"0.0 {box_length:.6f} ylo yhi\n")
        f.write(f"0.0 {box_length:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for t, m in sorted(masses.items()):
            label = {1: "Si", 2: "O", 3: "H"}[t]
            f.write(f"{t} {m:.4f}  # {label}\n")
        f.write("\nAtoms # charge\n\n")
        for i, (pos, atype) in enumerate(zip(positions, types)):
            # charge = 0.0 for ReaxFF (QEq equilibrates)
            f.write(f"{i+1} {atype} 0.0 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    print(f"  Written {n_atoms} atoms to {filename}")


# ─── pH-to-Composition Mapping ──────────────────────────────────────────────

def get_composition(ph, n_silica=100, water_ratio=4.0):
    """Determine molecular composition for a given pH.

    Args:
        ph: pH value (6.0 - 8.0)
        n_silica: number of Si(OH)4 / Si(OH)3O- molecules
        water_ratio: H2O molecules per silica molecule

    Returns:
        n_sioh4: number of fully protonated Si(OH)4
        n_sioh3: number of deprotonated Si(OH)3(O-)
        n_water: number of H2O molecules
        n_hydronium: number of H3O+ (charge compensation)
    """
    # Fraction of deprotonated silanols increases with pH
    # pKa of silanol ~ 6.8, so at pH 6.0 very few are deprotonated
    # Using Henderson-Hasselbalch: fraction = 1 / (1 + 10^(pKa - pH))
    pKa = 6.8
    frac_deprotonated = 1.0 / (1.0 + 10.0**(pKa - ph))

    # Each Si(OH)4 has 4 silanol groups; we deprotonate one per molecule
    # for the fraction of molecules that are deprotonated
    n_sioh3 = int(round(n_silica * frac_deprotonated))
    n_sioh4 = n_silica - n_sioh3

    n_water = int(round(n_silica * water_ratio))

    # Each deprotonated silanol removes 1 H; add H3O+ for charge neutrality
    n_hydronium = n_sioh3

    return n_sioh4, n_sioh3, n_water, n_hydronium


def estimate_box_length(n_sioh4, n_sioh3, n_water, n_hydronium,
                        target_density=1.5):
    """Estimate cubic box length for target density (g/cm³)."""
    # Molecular weights (g/mol)
    mw_sioh4 = 96.115
    mw_sioh3 = 95.107  # Si(OH)3O- (missing one H)
    mw_water = 18.015
    mw_h3o = 19.023    # H3O+

    total_mass = (n_sioh4 * mw_sioh4 + n_sioh3 * mw_sioh3
                  + n_water * mw_water + n_hydronium * mw_h3o)

    # Volume in cm³, then convert to Å³
    avogadro = 6.02214076e23
    volume_cm3 = total_mass / (target_density * avogadro)
    volume_ang3 = volume_cm3 * 1e24  # 1 cm = 1e8 Å → 1 cm³ = 1e24 ų

    box_length = volume_ang3 ** (1.0 / 3.0)
    return box_length


# ─── Main ────────────────────────────────────────────────────────────────────

def build_system(ph, n_silica=100, water_ratio=4.0, target_density=1.5,
                 seed=None, output_dir="../simulations"):
    """Build a precursor system for a given pH value."""

    print(f"\n{'='*60}")
    print(f"  Building precursor system for pH {ph:.1f}")
    print(f"{'='*60}")

    # Composition
    n_sioh4, n_sioh3, n_water, n_hydronium = get_composition(
        ph, n_silica, water_ratio)

    print(f"  Si(OH)4:      {n_sioh4}")
    print(f"  Si(OH)3(O-):  {n_sioh3}")
    print(f"  H2O:          {n_water}")
    print(f"  H3O+:         {n_hydronium}")

    total_atoms = n_sioh4 * 9 + n_sioh3 * 8 + n_water * 3 + n_hydronium * 4
    print(f"  Total atoms:  {total_atoms}")

    # Box size
    box_length = estimate_box_length(n_sioh4, n_sioh3, n_water, n_hydronium,
                                     target_density)
    print(f"  Box length:   {box_length:.2f} Å")
    print(f"  Target ρ:     {target_density} g/cm³")

    # Build molecule list
    rng = np.random.default_rng(seed)
    molecules = []

    sioh4_template = make_sioh4()
    sioh3_template = make_sioh3_ominus()
    water_template = make_water()
    h3o_template = make_hydronium()

    for _ in range(n_sioh4):
        molecules.append(sioh4_template)
    for _ in range(n_sioh3):
        molecules.append(sioh3_template)
    for _ in range(n_water):
        molecules.append(water_template)
    for _ in range(n_hydronium):
        molecules.append(h3o_template)

    # Shuffle to randomize placement order
    rng.shuffle(molecules)

    # Pack
    print(f"\n  Packing {len(molecules)} molecules into box...")
    positions, types, success = pack_molecules(
        molecules, box_length, rng, min_dist=1.4, max_attempts=50000)

    if not success:
        attempts = 0
        while not success and attempts < 5:
            print(f"  ERROR: Packing failed. Trying with {(attempts+1)*5}% larger box...")
            box_length *= 1.05
            attempts += 1
            positions, types, success = pack_molecules(
                molecules, box_length, rng, min_dist=1.3, max_attempts=50000)
            
        if not success:
            print("  FATAL: Packing failed even with much larger box.")
            sys.exit(1)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"data.pH_{ph:.1f}")
    comment = (f"pH={ph:.1f}, {n_sioh4} Si(OH)4 + {n_sioh3} Si(OH)3O- "
               f"+ {n_water} H2O + {n_hydronium} H3O+")
    write_lammps_data(filename, positions, types, box_length, comment=comment)

    # Write metadata
    meta_file = os.path.join(output_dir, f"meta.pH_{ph:.1f}.txt")
    with open(meta_file, 'w') as f:
        f.write(f"pH = {ph:.1f}\n")
        f.write(f"n_sioh4 = {n_sioh4}\n")
        f.write(f"n_sioh3 = {n_sioh3}\n")
        f.write(f"n_water = {n_water}\n")
        f.write(f"n_hydronium = {n_hydronium}\n")
        f.write(f"total_atoms = {total_atoms}\n")
        f.write(f"box_length = {box_length:.6f}\n")
        f.write(f"target_density = {target_density}\n")
        f.write(f"seed = {seed}\n")

    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Build Si(OH)4 + H2O precursor systems for sol-gel simulation")
    parser.add_argument("--ph", type=float, nargs="+", default=None,
                        help="pH values to build (default: 6.0 6.5 7.0 7.5 8.0)")
    parser.add_argument("--n-silica", type=int, default=100,
                        help="Number of silica molecules (default: 100)")
    parser.add_argument("--water-ratio", type=float, default=4.0,
                        help="H2O per silica molecule (default: 4.0)")
    parser.add_argument("--density", type=float, default=1.0,
                        help="Target density in g/cm³ (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ../simulations)")
    args = parser.parse_args()

    if args.ph is None:
        ph_values = [6.0, 6.5, 7.0, 7.5, 8.0]
    else:
        ph_values = args.ph

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "simulations")
    else:
        output_dir = args.output_dir

    print("Sol-Gel Precursor System Builder")
    print(f"pH values: {ph_values}")
    print(f"Silica molecules: {args.n_silica}")
    print(f"Water ratio: {args.water_ratio}")
    print(f"Target density: {args.density} g/cm³")

    # Print composition table
    print(f"\n{'pH':>6} {'Si(OH)4':>8} {'Si(OH)3O-':>10} {'H2O':>6} "
          f"{'H3O+':>6} {'Atoms':>7}")
    print("-" * 50)
    for ph in ph_values:
        n1, n2, n3, n4 = get_composition(ph, args.n_silica, args.water_ratio)
        total = n1 * 9 + n2 * 8 + n3 * 3 + n4 * 4
        print(f"{ph:6.1f} {n1:8d} {n2:10d} {n3:6d} {n4:6d} {total:7d}")
    print()

    for ph in ph_values:
        build_system(ph, args.n_silica, args.water_ratio, args.density,
                     args.seed, output_dir)

    print(f"\n{'='*60}")
    print(f"  All systems built successfully!")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
