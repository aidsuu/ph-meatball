import sys
import numpy as np
from scipy.spatial import KDTree

def check(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    atoms = []
    box_size = 27.0
    in_atoms = False
    for line in lines:
        if 'xhi' in line:
            parts = line.split()
            box_size = float(parts[1]) - float(parts[0])
            
        if line.strip().startswith("Atoms"):
            in_atoms = True
            continue
        if in_atoms and line.strip() == "":
            if len(atoms) > 0:
                break
            continue
        if in_atoms:
            parts = line.split()
            if len(parts) >= 6:
                atoms.append((int(parts[0]), int(parts[1]), float(parts[3]), float(parts[4]), float(parts[5])))

    si_atoms = [a for a in atoms if a[1] == 1]
    si_pos = np.array([[a[2], a[3], a[4]] for a in si_atoms])

    si_tree = KDTree(si_pos, boxsize=[box_size, box_size, box_size])
    
    pairs = si_tree.query_pairs(r=2.5)
    close_pairs = 0
    for i, j in pairs:
        # Check if > 2.2
        p1 = si_pos[i]
        p2 = si_pos[j]
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        dx = dx - box_size * round(dx / box_size)
        dy = dy - box_size * round(dy / box_size)
        dz = dz - box_size * round(dz / box_size)
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        if 2.2 < d <= 2.5:
            close_pairs += 1
            
    print(f"File: {input_file}")
    print(f"True Si-Si pairs between 2.2 and 2.5 A: {close_pairs}")

check(sys.argv[1])
