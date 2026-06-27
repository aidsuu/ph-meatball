import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def compute_rdf(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    atoms = []
    box_size = 16.0
    in_atoms = False
    for line in lines:
        if 'xhi' in line:
            parts = line.split()
            box_size = float(parts[1]) - float(parts[0])
            
        if line.startswith('Atoms'):
            in_atoms = True
            continue
        if in_atoms and line.strip() == '':
            if len(atoms) > 0:
                break
            continue
        if in_atoms:
            parts = line.split()
            if len(parts) >= 6:
                atoms.append((int(parts[1]), float(parts[3]), float(parts[4]), float(parts[5])))
                
    si_atoms = np.array([[a[1], a[2], a[3]] for a in atoms if a[0] == 1])
    o_atoms = np.array([[a[1], a[2], a[3]] for a in atoms if a[0] == 2])
    
    dists = cdist(si_atoms, o_atoms)
    # minimum image convention
    dists = dists - box_size * np.round(dists / box_size)
    dists = np.abs(dists)
    
    hist, bin_edges = np.histogram(dists, bins=100, range=(1.0, 3.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    peak_count_normal = np.sum(hist[(bin_centers > 1.4) & (bin_centers < 1.9)])
    peak_count_far = np.sum(hist[(bin_centers > 1.9) & (bin_centers < 2.5)])
    
    print(f"File: {data_file} | Box: {box_size}")
    print(f"Si-O pairs between 1.4 and 1.9 A: {peak_count_normal}")
    print(f"Si-O pairs between 1.9 and 2.5 A: {peak_count_far}")
    
if __name__ == '__main__':
    compute_rdf(sys.argv[1])
