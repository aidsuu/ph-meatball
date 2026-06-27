import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

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
    
    dists = pdist(si_atoms)
    # minimum image convention
    dists = dists - box_size * np.round(dists / box_size)
    dists = np.abs(dists)
    
    hist, bin_edges = np.histogram(dists, bins=100, range=(1.0, 5.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # plot
    plt.plot(bin_centers, hist)
    plt.xlabel('Si-Si Distance (A)')
    plt.ylabel('Count')
    plt.title('Si-Si RDF')
    plt.savefig('../results/sisi_rdf.png')
    
    # check for peak at 2.3-2.5
    peak_count = np.sum(hist[(bin_centers > 2.2) & (bin_centers < 2.5)])
    print(f"File: {data_file} | Box: {box_size}")
    print(f"Si-Si pairs between 2.2 and 2.5 A: {peak_count}")
    
if __name__ == '__main__':
    compute_rdf(sys.argv[1])
