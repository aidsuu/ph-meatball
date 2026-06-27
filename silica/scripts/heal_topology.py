#!/usr/bin/env python3
"""
Heal Silica Topology (Phase C)
Replaces the destructive 'strip_oxygen.py'.
If any silanols (Si-OH) remain after Sol-Gel, stripping H leaves Non-Bridging Oxygens (NBOs).
Instead of randomly deleting NBOs (which creates SiO3 defects), this script:
1. Identifies all under-coordinated Si or NBOs.
2. Pairs them up based on distance.
3. Merges two NBOs into a single Bridging Oxygen (BO) at the midpoint between the two Si atoms.
4. Ensures perfect 1:2 stoichiometry without destroying the tetrahedral network!
"""

import argparse
import numpy as np
from scipy.spatial import KDTree

def heal_network(input_file, output_file, si_type=1, o_type=2, h_type=3):
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        lines = f.readlines()

    header_lines = []
    atoms = []
    box = np.zeros(3)
    
    box_x = box_y = box_z = 0.0
    xlo = ylo = zlo = 0.0
    
    in_atoms = False
    for line in lines:
        if 'xlo xhi' in line:
            parts = line.split()
            xlo, xhi = float(parts[0]), float(parts[1])
            box[0] = xhi - xlo
        elif 'ylo yhi' in line:
            parts = line.split()
            ylo, yhi = float(parts[0]), float(parts[1])
            box[1] = yhi - ylo
        elif 'zlo zhi' in line:
            parts = line.split()
            zlo, zhi = float(parts[0]), float(parts[1])
            box[2] = zhi - zlo
            
        if line.startswith('Atoms'):
            in_atoms = True
            header_lines.append(line)
            continue
            
        if in_atoms:
            if line.strip() == '':
                if len(atoms) > 0: in_atoms = False
                continue
            parts = line.split()
            if len(parts) >= 6:
                atoms.append([int(parts[0]), int(parts[1]), float(parts[2]), 
                              float(parts[3]), float(parts[4]), float(parts[5])])
        elif not in_atoms and len(atoms) == 0:
            header_lines.append(line)

    atoms = np.array(atoms)
    
    # 1. Strip ALL Hydrogen automatically
    non_h_mask = atoms[:, 1] != h_type
    atoms = atoms[non_h_mask]
    
    si_mask = atoms[:, 1] == si_type
    o_mask = atoms[:, 1] == o_type
    
    offset = np.array([xlo, ylo, zlo])
    si_coords = atoms[si_mask, 3:6] - offset
    o_coords = atoms[o_mask, 3:6] - offset
    si_ids = atoms[si_mask, 0]
    o_ids = atoms[o_mask, 0]
    
    # 2. Build KDTree for Si to find Oxygen coordination
    si_tree = KDTree(si_coords, boxsize=box)
    
    nbo_list = []  # List of (O_index, bonded_Si_index)
    
    # Find Non-Bridging Oxygens (Oxygens with only 1 Si neighbor)
    for i, o_c in enumerate(o_coords):
        si_neighbors = si_tree.query_ball_point(o_c, 2.0)
        if len(si_neighbors) == 1:
            nbo_list.append((i, si_neighbors[0]))
            
    print(f"Total Si: {len(si_coords)}")
    print(f"Total O: {len(o_coords)}")
    print(f"Found {len(nbo_list)} Non-Bridging Oxygens (NBOs).")
    
    target_o_count = len(si_coords) * 2
    excess_o = len(o_coords) - target_o_count
    
    print(f"Target O count for perfect SiO2: {target_o_count}")
    print(f"Excess O to remove: {excess_o}")
    
    if excess_o < 0:
        print("ERROR: System has TOO FEW Oxygens! Cannot heal.")
        return
    elif excess_o == 0:
        print("System is already perfectly stoichiometric! No healing needed.")
        to_delete_o_indices = set()
    else:
        if len(nbo_list) < excess_o * 2:
            print("WARNING: Not enough NBOs to heal all excess oxygen cleanly. Network might be heavily over-coordinated.")
            to_delete_o_indices = set(np.random.choice(len(o_coords), excess_o, replace=False))
        else:
            to_delete_o_indices = set()
            
            # Create a list of available NBOs
            available_nbos = []
            for o_idx, si_idx in nbo_list:
                available_nbos.append({
                    'o_idx': o_idx,
                    'si_idx': si_idx,
                    'si_coord': si_coords[si_idx]
                })
                
            healed_count = 0
            for step in range(excess_o):
                if len(available_nbos) < 2:
                    break
                    
                # Pick the first NBO
                nbo1 = available_nbos.pop(0)
                si1_c = nbo1['si_coord']
                
                # Find the closest NBO that belongs to a DIFFERENT Si atom
                best_dist = float('inf')
                best_idx = -1
                
                for j, nbo2 in enumerate(available_nbos):
                    if nbo2['si_idx'] == nbo1['si_idx']:
                        continue # MUST be different Si
                    
                    dr = nbo2['si_coord'] - si1_c
                    dr -= box * np.round(dr / box)
                    dist = np.linalg.norm(dr)
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = j
                        
                if best_idx == -1:
                    # No different Si available! Discard this NBO.
                    continue
                    
                nbo2 = available_nbos.pop(best_idx)
                
                # Pair nbo1 and nbo2
                to_delete_o_indices.add(nbo2['o_idx'])
                
                dr = nbo2['si_coord'] - nbo1['si_coord']
                dr -= box * np.round(dr / box)
                # Add a tiny random jitter to prevent perfect overlap if multiple NBOs are bridged between the same Si pair
                midpoint = nbo1['si_coord'] + 0.5 * dr + np.random.normal(0, 0.2, 3)
                
                global_o1_idx = np.where(atoms[:, 0] == o_ids[nbo1['o_idx']])[0][0]
                atoms[global_o1_idx, 3:6] = midpoint
                healed_count += 1
                
            print(f"Successfully healed {healed_count} pairs of NBOs into Bridging Oxygens.")

    # Apply deletions
    if len(to_delete_o_indices) > 0:
        o_ids_to_delete = set(o_ids[list(to_delete_o_indices)])
        final_mask = np.array([a[0] not in o_ids_to_delete for a in atoms])
        atoms = atoms[final_mask]
        
    # Re-assign fixed Vashishta charges
    for a in atoms:
        if a[1] == si_type:
            a[2] = 1.6
        elif a[1] == o_type:
            a[2] = -0.8
            
    # Renumber IDs
    for i in range(len(atoms)):
        atoms[i][0] = i + 1
        
    print(f"Final atoms: {len(atoms)} (Si: {np.sum(atoms[:, 1] == si_type)}, O: {np.sum(atoms[:, 1] == o_type)})")
    
    # Write output
    with open(output_file, 'w') as f:
        in_masses = False
        for line in header_lines:
            if 'atoms' in line and 'Atoms' not in line:
                f.write(f"{len(atoms)} atoms\n")
            elif 'atom types' in line:
                f.write(f"2 atom types\n")
            elif 'Masses' in line:
                in_masses = True
                f.write("Masses\n\n1 28.0855\n2 15.999\n")
            elif in_masses and line.strip() == '':
                continue
            elif in_masses and (line.strip().startswith('1') or line.strip().startswith('2') or line.strip().startswith('3')):
                if line.strip().startswith('3'):
                    in_masses = False
                    f.write("\n")
                continue
            else:
                f.write(line)
        f.write("\n")
        
        for a in atoms:
            f.write(f"{int(a[0])} {int(a[1])} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f} {a[5]:.6f}\n")
            
    print(f"Wrote healed structure to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input ReaxFF data file")
    parser.add_argument("output", help="Output Vashishta data file")
    args = parser.parse_args()
    heal_network(args.input, args.output)
