import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

def compute_rdf(dump_file):
    with open(dump_file, 'r') as f:
        lines = f.readlines()
    
    # Find the last frame
    last_timestep_idx = -1
    for i in range(len(lines)-1, -1, -1):
        if lines[i].startswith('ITEM: TIMESTEP'):
            last_timestep_idx = i
            break
            
    if last_timestep_idx == -1:
        print("No frames found.")
        return
        
    frame_lines = lines[last_timestep_idx:]
    
    # Parse box bounds
    xlo, xhi = map(float, frame_lines[5].split())
    ylo, yhi = map(float, frame_lines[6].split())
    zlo, zhi = map(float, frame_lines[7].split())
    box_size = xhi - xlo
    
    atoms = []
    # Parse atoms
    for line in frame_lines[9:]:
        parts = line.split()
        if len(parts) >= 5:
            atoms.append((int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
            
    si_atoms = np.array([[a[1], a[2], a[3]] for a in atoms if a[0] == 1])
    
    dists = pdist(si_atoms)
    # minimum image convention
    dists = dists - box_size * np.round(dists / box_size)
    dists = np.abs(dists)
    
    hist, bin_edges = np.histogram(dists, bins=100, range=(1.0, 5.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    peak_count = np.sum(hist[(bin_centers > 2.2) & (bin_centers < 2.5)])
    print(f"File: {dump_file} | Box: {box_size}")
    print(f"Si-Si pairs between 2.2 and 2.5 A: {peak_count}")
    
if __name__ == '__main__':
    compute_rdf(sys.argv[1])
