import sys
import numpy as np
from scipy.spatial import KDTree

def evaporate(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    atoms = []
    header_lines = []
    velocities = []
    
    in_atoms = False
    in_velocities = False
    
    box_x = box_y = box_z = 0.0
    xlo = ylo = zlo = 0.0
    
    for line in lines:
        if 'xlo xhi' in line:
            xlo, xhi = map(float, line.split()[:2])
            box_x = xhi - xlo
        if 'ylo yhi' in line:
            ylo, yhi = map(float, line.split()[:2])
            box_y = yhi - ylo
        if 'zlo zhi' in line:
            zlo, zhi = map(float, line.split()[:2])
            box_z = zhi - zlo
            
        if line.startswith('Atoms'):
            in_atoms = True
            in_velocities = False
            header_lines.append(line)
            continue
            
        if line.startswith('Velocities'):
            in_velocities = True
            in_atoms = False
            continue
            
        if in_atoms:
            if line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 6:
                atoms.append([int(parts[0]), int(parts[1]), float(parts[2]), 
                              float(parts[3]), float(parts[4]), float(parts[5])])
                              
        elif in_velocities:
            if line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 4:
                velocities.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
                
        else:
            header_lines.append(line)
            
    atoms = np.array(atoms)
    
    if len(atoms) == 0:
        print("No atoms found!")
        return
        
    si_mask = atoms[:, 1] == 1
    o_mask = atoms[:, 1] == 2
    h_mask = atoms[:, 1] == 3
    
    offset = np.array([xlo, ylo, zlo])
    si_coords = atoms[si_mask, 3:6] - offset
    o_coords = atoms[o_mask, 3:6] - offset
    h_coords = atoms[h_mask, 3:6] - offset
    
    si_ids = atoms[si_mask, 0]
    o_ids = atoms[o_mask, 0]
    h_ids = atoms[h_mask, 0]
    
    si_tree = KDTree(si_coords, boxsize=[box_x, box_y, box_z]) if len(si_coords) > 0 else None
    h_tree = KDTree(h_coords, boxsize=[box_x, box_y, box_z]) if len(h_coords) > 0 else None
    
    to_delete = set()
    water_count = 0
    
    for i, o_c in enumerate(o_coords):
        h_neighbors = h_tree.query_ball_point(o_c, 1.3) if h_tree else []
        si_neighbors = si_tree.query_ball_point(o_c, 2.0) if si_tree else []
        
        if len(h_neighbors) == 2 and len(si_neighbors) == 0:
            to_delete.add(o_ids[i])
            for h_idx in h_neighbors:
                to_delete.add(h_ids[h_idx])
            water_count += 1
            
    print(f"Evaporated {water_count} water molecules from {input_file}")
    
    new_atoms = [a for a in atoms if a[0] not in to_delete]
    
    vel_dict = {v[0]: v for v in velocities}
    new_velocities = []
    
    for i in range(len(new_atoms)):
        old_id = new_atoms[i][0]
        new_id = i + 1
        new_atoms[i][0] = new_id
        if old_id in vel_dict:
            new_velocities.append([new_id, vel_dict[old_id][1], vel_dict[old_id][2], vel_dict[old_id][3]])
            
    with open(output_file, 'w') as f:
        for i, line in enumerate(header_lines):
            if 'atoms' in line and 'Atoms' not in line:
                f.write(f"{len(new_atoms)} atoms\n")
            else:
                f.write(line)
        f.write("\n")
        
        for a in new_atoms:
            f.write(f"{int(a[0])} {int(a[1])} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f} {a[5]:.6f}\n")
            
        if len(new_velocities) > 0:
            f.write("\nVelocities\n\n")
            for v in new_velocities:
                f.write(f"{int(v[0])} {v[1]:.6f} {v[2]:.6f} {v[3]:.6f}\n")

if __name__ == '__main__':
    evaporate(sys.argv[1], sys.argv[2])
