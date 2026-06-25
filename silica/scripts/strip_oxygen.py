import sys
import numpy as np
import random
from scipy.spatial import KDTree

def strip_excess_oxygen(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    header = []
    atoms = []
    masses_lines = []
    
    reading_atoms = False
    reading_masses = False
    
    for line in lines:
        if line.strip() == "Masses":
            reading_masses = True
            reading_atoms = False
            masses_lines.append(line)
            continue
        elif line.strip() == "Atoms # charge":
            reading_atoms = True
            reading_masses = False
            header.append(line)
            continue
            
        if reading_masses:
            masses_lines.append(line)
            if line.strip() == "":
                reading_masses = False
        elif reading_atoms:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    atom_id = int(parts[0])
                    atom_type = int(parts[1])
                    q = float(parts[2])
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    atoms.append((atom_id, atom_type, q, x, y, z))
                except ValueError:
                    header.append(line)
            else:
                if len(atoms) == 0:
                    header.append(line)
        else:
            header.append(line)

    si_atoms = [a for a in atoms if a[1] == 1]
    o_atoms = [a for a in atoms if a[1] == 2]

    target_o = len(si_atoms) * 2

    if len(o_atoms) <= target_o:
        print(f"Skipping {input_file}, O atoms ({len(o_atoms)}) <= target ({target_o})")
        return

    si_positions = np.array([[a[3], a[4], a[5]] for a in si_atoms])
    o_positions = np.array([[a[3], a[4], a[5]] for a in o_atoms])

    si_tree = KDTree(si_positions)
    o_tree = KDTree(o_positions)

    cutoff = 1.9
    o_neighbors = o_tree.query_ball_tree(si_tree, cutoff)
    
    o_free = []
    o_nbo = []
    o_bo = []

    for i, neighbors in enumerate(o_neighbors):
        if len(neighbors) == 0:
            o_free.append(o_atoms[i])
        elif len(neighbors) == 1:
            o_nbo.append(o_atoms[i])
        else:
            o_bo.append(o_atoms[i])

    # Try to keep all BOs
    kept_o = list(o_bo)
    
    # Shuffle NBOs to remove randomly
    random.seed(42) # For reproducibility
    random.shuffle(o_nbo)
    
    # We need (target_o - len(kept_o)) more oxygen atoms
    needed = target_o - len(kept_o)
    
    if needed > 0:
        if needed <= len(o_nbo):
            kept_o.extend(o_nbo[:needed])
        else:
            # If we need more than we have NBOs, take all NBOs and some free O
            kept_o.extend(o_nbo)
            needed_free = needed - len(o_nbo)
            random.shuffle(o_free)
            kept_o.extend(o_free[:needed_free])

    final_atoms = si_atoms + kept_o
    # Sort by ID to maintain some order, though renumbering
    final_atoms.sort(key=lambda a: a[0])
    
    # Update header counts
    new_header = []
    for line in header:
        if "atoms" in line and "Atoms" not in line:
            new_header.append(f"{len(final_atoms)} atoms\n")
        else:
            new_header.append(line)

    with open(output_file, 'w') as f:
        for line in new_header:
            f.write(line)
        for line in masses_lines:
            f.write(line)
        
        # Write Atoms block
        for i, a in enumerate(final_atoms):
            new_id = i + 1
            f.write(f"{new_id} {a[1]} {a[2]:.4f} {a[3]:.6f} {a[4]:.6f} {a[5]:.6f}\n")

    print(f"Processed {input_file} -> {output_file}")
    print(f"  Original: {len(si_atoms)} Si, {len(o_atoms)} O")
    print(f"  Final   : {len(si_atoms)} Si, {len(kept_o)} O (Total: {len(final_atoms)})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 strip_oxygen.py <input.data> <output.data>")
        sys.exit(1)
    strip_excess_oxygen(sys.argv[1], sys.argv[2])
