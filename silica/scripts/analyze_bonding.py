import sys
import numpy as np
from scipy.spatial import KDTree

def analyze_bonding(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()

    atoms = []
    # Parse atoms from LAMMPS data file (id type q x y z)
    reading_atoms = False
    for line in lines:
        if line.startswith('Atoms'):
            reading_atoms = True
            continue
        if reading_atoms:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    atom_id = int(parts[0])
                    atom_type = int(parts[1])
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    atoms.append((atom_id, atom_type, x, y, z))
                except ValueError:
                    pass
            elif len(parts) == 0 and len(atoms) > 0:
                break # end of Atoms block

    # Type 1 = Si, Type 2 = O
    si_positions = np.array([ [a[2], a[3], a[4]] for a in atoms if a[1] == 1 ])
    o_positions = np.array([ [a[2], a[3], a[4]] for a in atoms if a[1] == 2 ])

    print(f"Total Si atoms: {len(si_positions)}")
    print(f"Total O atoms: {len(o_positions)}")
    print(f"O/Si Ratio: {len(o_positions)/len(si_positions):.2f}")

    if len(si_positions) == 0 or len(o_positions) == 0:
        return

    si_tree = KDTree(si_positions)
    o_tree = KDTree(o_positions)

    # Si-O bond cutoff = 1.9 Angstroms
    cutoff = 1.9

    # For each Si, count how many O neighbors
    si_neighbors = si_tree.query_ball_tree(o_tree, cutoff)
    si_coordination = [len(n) for n in si_neighbors]
    
    # For each O, count how many Si neighbors
    o_neighbors = o_tree.query_ball_tree(si_tree, cutoff)
    o_coordination = [len(n) for n in o_neighbors]

    print("\n--- Silicon Coordination (Si-O bonds) ---")
    for i in range(1, 6):
        count = si_coordination.count(i)
        print(f"Si with {i} O neighbors: {count} ({count/len(si_coordination)*100:.1f}%)")

    print("\n--- Oxygen Coordination (O-Si bonds) ---")
    for i in range(0, 4):
        count = o_coordination.count(i)
        if i == 1:
            name = "Non-Bridging Oxygen (NBO)"
        elif i == 2:
            name = "Bridging Oxygen (BO)"
        elif i == 0:
            name = "Free Oxygen"
        else:
            name = "Overcoordinated"
        print(f"O with {i} Si neighbors: {count} ({count/len(o_coordination)*100:.1f}%) - {name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_bonding.py <data_file>")
        sys.exit(1)
    analyze_bonding(sys.argv[1])
