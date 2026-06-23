#!/usr/bin/env python3
"""
Generate PLUMED NDX file (Si atoms) for metadynamics.

Reads LAMMPS data file, finds all atoms of given type, writes GROMACS-style NDX
with one line per 15 atom IDs.

Usage:
    python3 gen_ndx.py <data_file> <si_type> <output_ndx>
"""
import sys, os

def main():
    if len(sys.argv) < 4:
        print("Usage: gen_ndx.py <data_file> <si_type> <output_ndx>")
        sys.exit(1)

    data_file = sys.argv[1]
    si_type = int(sys.argv[2])
    out_ndx = sys.argv[3]

    si_ids = []
    in_atoms = False
    skip_one = 0
    with open(data_file) as f:
        for line in f:
            if line.strip().startswith("Atoms"):
                in_atoms = True
                skip_one = 1   # skip the " # charge" comment line
                continue
            if in_atoms and skip_one > 0:
                skip_one -= 1
                continue
            if in_atoms and line.strip() == "":
                break
            if in_atoms:
                parts = line.split()
                if len(parts) >= 2 and int(parts[1]) == si_type:
                    si_ids.append(int(parts[0]))

    with open(out_ndx, "w") as f:
        f.write("[ si ]\n")
        for i in range(0, len(si_ids), 15):
            f.write(" ".join(str(x) for x in si_ids[i:i+15]) + "\n")

    print(f"  Wrote {len(si_ids)} Si atom IDs to {out_ndx}")

if __name__ == "__main__":
    main()
