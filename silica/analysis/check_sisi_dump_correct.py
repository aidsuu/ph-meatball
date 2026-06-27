import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform

def check(dump_file):
    with open(dump_file, 'r') as f:
        lines = f.readlines()
    
    last_idx = -1
    for i in range(len(lines)-1, -1, -1):
        if lines[i].startswith('ITEM: TIMESTEP'):
            last_idx = i
            break
            
    frame_lines = lines[last_idx:]
    xlo, xhi = map(float, frame_lines[5].split())
    ylo, yhi = map(float, frame_lines[6].split())
    zlo, zhi = map(float, frame_lines[7].split())
    box_x = xhi - xlo
    box_y = yhi - ylo
    box_z = zhi - zlo
    
    atoms = []
    for line in frame_lines[9:]:
        parts = line.split()
        if len(parts) >= 5:
            atoms.append((int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
            
    si_atoms = np.array([[a[1], a[2], a[3]] for a in atoms if a[0] == 1])
    
    N = len(si_atoms)
    close_pairs = 0
    for i in range(N):
        for j in range(i+1, N):
            dx = si_atoms[i,0] - si_atoms[j,0]
            dy = si_atoms[i,1] - si_atoms[j,1]
            dz = si_atoms[i,2] - si_atoms[j,2]
            
            dx -= box_x * round(dx / box_x)
            dy -= box_y * round(dy / box_y)
            dz -= box_z * round(dz / box_z)
            
            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            if 2.2 < d <= 2.5:
                close_pairs += 1
                
    print(f"File: {dump_file}")
    print(f"True Si-Si pairs between 2.2 and 2.5 A: {close_pairs}")

check(sys.argv[1])
