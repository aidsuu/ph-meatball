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
    o_atoms = [a for a in atoms if a[1] == 2]

    si_pos = np.array([[a[2], a[3], a[4]] for a in si_atoms])
    o_pos = np.array([[a[2], a[3], a[4]] for a in o_atoms])

    si_tree = KDTree(si_pos, boxsize=[box_size, box_size, box_size])
    o_tree = KDTree(o_pos, boxsize=[box_size, box_size, box_size])
    
    o_neighbors = o_tree.query_ball_tree(si_tree, 1.9)
    o_free = sum(1 for n in o_neighbors if len(n) == 0)
    o_nbo = sum(1 for n in o_neighbors if len(n) == 1)
    o_bo = sum(1 for n in o_neighbors if len(n) >= 2)
    
    si_neighbors = si_tree.query_ball_tree(o_tree, 1.9)
    si_0 = sum(1 for n in si_neighbors if len(n) == 0)
    si_1 = sum(1 for n in si_neighbors if len(n) == 1)
    si_2 = sum(1 for n in si_neighbors if len(n) == 2)
    si_3 = sum(1 for n in si_neighbors if len(n) == 3)
    si_4 = sum(1 for n in si_neighbors if len(n) >= 4)

    print(f"File: {input_file}")
    print(f"O atoms: Free={o_free}, NBO={o_nbo}, BO={o_bo}")
    print(f"Si atoms: 0-coord={si_0}, 1-coord={si_1}, 2-coord={si_2}, 3-coord={si_3}, 4+-coord={si_4}")

check(sys.argv[1])
