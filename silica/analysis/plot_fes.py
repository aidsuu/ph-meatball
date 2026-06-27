import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_fes(dat_file, out_file):
    with open(dat_file, 'r') as f:
        lines = f.readlines()
        
    x_bins, y_bins = set(), set()
    data = []
    
    for line in lines:
        if line.startswith('#!'):
            continue
        parts = line.split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            x_bins.add(x)
            y_bins.add(y)
            data.append((x, y, z))
            
    nx = len(x_bins)
    ny = len(y_bins)
    
    X = np.zeros((ny, nx))
    Y = np.zeros((ny, nx))
    Z = np.zeros((ny, nx))
    
    x_sorted = sorted(list(x_bins))
    y_sorted = sorted(list(y_bins))
    
    x_map = {val: i for i, val in enumerate(x_sorted)}
    y_map = {val: i for i, val in enumerate(y_sorted)}
    
    for x, y, z in data:
        ix = x_map[x]
        iy = y_map[y]
        X[iy, ix] = x
        Y[iy, ix] = y
        Z[iy, ix] = z
        
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Free Energy (kJ/mol)')
    plt.xlabel('Averaged Steinhardt $q_4$')
    plt.ylabel('Averaged Steinhardt $q_6$')
    plt.title(f'Free Energy Surface ({dat_file})')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Saved FES plot to {out_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 plot_fes.py <input.dat> <output.png>")
        sys.exit(1)
    plot_fes(sys.argv[1], sys.argv[2])
