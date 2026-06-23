# Silica Sol-Gel & Calcination Simulation

## Project Overview
This project simulates the sol-gel synthesis and subsequent high-temperature calcination of silica ($\text{SiO}_2$) to study how pH affects the resulting crystal structures. By varying the initial ratio of protonated to deprotonated silanol groups, we model the chemical environment corresponding to different pH conditions (pH 6.0–8.0).

To overcome the challenges of simulating complex chemical reactions and long crystallization timescales, this project employs a **hybrid MD strategy**:
1. **Sol-Gel Condensation (ReaxFF)**: Captures the chemical reactions (bond-breaking/forming) of the sol-gel process where orthosilicic acid molecules condense to form a silica network and water byproducts.
2. **Calcination (Vashishta + PLUMED2 Metadynamics)**: Simulates calcination at $900^\circ\text{C}$ ($1173\text{ K}$) using the fast, accurate Vashishta potential. Crystallization is accelerated using well-tempered metadynamics with Steinhardt order parameters ($Q_4$ and $Q_6$) as Collective Variables.

---

## Project Goals & Checklist
- [x] **Phase A (Precursor Building)**: Generate initial configuration boxes containing $\text{Si(OH)}_4$, $\text{Si(OH)}_3\text{O}^-$, $\text{H}_2\text{O}$, and $\text{H}_3\text{O}^+$ corresponding to pH 6.0, 6.5, 7.0, 7.5, and 8.0.
- [ ] **Phase B (Sol-Gel Condensation)**: Perform reactive MD runs with ReaxFF to simulate condensation chemistry.
- [ ] **Phase C (Drying & Conversion)**: Post-process the sol-gel output to remove water and hydrogen, preparing a dry, charge-balanced $\text{SiO}_2$ network.
- [ ] **Phase D (Calcination)**: Run accelerated MD with PLUMED2 well-tempered metadynamics to transition amorphous silica to cristobalite/tridymite.
- [ ] **Phase E (Post-Processing & Analysis)**:
  - [ ] Compute powder X-ray diffraction (XRD) profiles using the Debye scattering formula.
  - [ ] Analyze per-atom Steinhardt parameters ($Q_4, Q_6$) to quantify crystalline polymorph fractions.
  - [ ] Perform primitive ring-size network topology analysis.
  - [ ] Generate comparison plots showing phase composition vs. pH.

---

## Environment Setup
### Full Tools (Local Machine)
The full environment is defined in `environment-full.yml` (includes GPU, visualization, etc.):
```bash
conda env create -f environment-full.yml
```

### HPC
A minimal environment is provided in `environment.yml` (CPU-only, no GUI):
```bash
# Install miniconda if not available
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh

conda env create -f environment.yml
conda activate rki
```

Both environments provide sufficcient simulation capabilities:
- **LAMMPS 29Aug2024** with: REAXFF, PLUMED, MANYBODY (Vashishta), KSPACE, OPENMP, ML-PACE, REPLICA
- **PLUMED 2.9.2** with MPI support

### Verify Installation
```bash
# Check LAMMPS packages
lmp -h | grep "Installed packages" -A 5

# Check PLUMED
plumed info --version   # Should print: 2.9

# Quick test (should complete without errors)
echo "units metal
atom_style charge
region box block 0 10 0 10 0 10
create_box 1 box
mass 1 28.0855
run 0" | lmp -in -
```

### Job Submission Example (SLURM)
```bash
#!/bin/bash
#SBATCH --job-name=silica_metad
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --partition=compute

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rki

# Run LAMMPS with MPI
mpirun -np $SLURM_NTASKS lmp -in calcination_metad.in
```

## Step-by-Step Execution Guide

To simulate the sol-gel process, calcination, and analyze the resulting silica structure for each pH condition (6.0, 6.5, 7.0, 7.5, 8.0), follow these steps:

### Phase 1: Build Precursor Systems (Phase A)
Generate initial starting boxes containing orthosilicic acid $\text{Si(OH)}_4$ and water.
```bash
# Run from the repository root:
python structures/build_precursor.py --n-silica 100 --density 1.0 --seed 42
```
* **Arguments**:
  * `--n-silica`: Number of silica molecules (default: 100). Use `100` for a fast local run or `250` for larger systems.
  * `--density`: Initial packing density in $\text{g/cm}^3$ (default: 1.0).
  * `--seed`: Random seed for coordinates generation (default: 42).
* **Output**: Generates configuration files like `data.pH_7.0` in the `simulations/` directory.

---

### Phase 2: Sol-Gel Condensation (Phase B - ReaxFF MD)
Run reactive MD to simulate the condensation reaction $\text{Si-OH} + \text{HO-Si} \rightarrow \text{Si-O-Si} + \text{H}_2\text{O}$ at high temperature.
```bash
cd simulations

# Run the ReaxFF simulation for a specific pH (e.g., pH 7.0)
mpirun -np 16 lmp -in solgel_reaxff.in -var pH 7.0 -var seed 12345
```
* **Variables**:
  * `-var pH`: Specifies the input structure (`data.pH_X.Y`) to load.
  * `-var seed`: Velocity initialization seed.
* **Output**: Generates `data.solgel_pH7.0.final` and species logs in the `simulations/` directory.

---

### Phase 3: Dry & Convert to Pure $\text{SiO}_2$ (Phase C)
Remove all water molecules and strip hydrogen atoms to prepare a clean amorphous $\text{SiO}_2$ network for Vashishta MD.
```bash
cd ../scripts

# Run stripping on the sol-gel output file
python strip_hydrogen.py ../simulations/data.solgel_pH7.0.final -o ../simulations/data.vash_pH7.0
```
* **Output**: Generates the Vashishta-compatible structure file `data.vash_pH7.0` (types: 1=Si, 2=O; charges set to 0.0).

---

### Phase 4: Calcination with Metadynamics (Phase D - Vashishta + PLUMED2)
Perform well-tempered metadynamics to accelerate crystallization under calcination conditions ($1173\text{ K}$).

1. **Generate the PLUMED index file** containing Silicon atom IDs (needed to restrict Steinhardt order parameter calculation to the Si sub-lattice):
   ```bash
   # From the scripts directory:
   python gen_plumed_ndx.py ../simulations/data.vash_pH7.0 -o ../simulations/si_atoms.ndx
   ```

2. **Run calcination**:
   ```bash
   cd ../simulations
   mpirun -np 16 lmp -in calcination_metad.in -var pH 7.0 -var seed 54321
   ```
* **Output**: Generates the calcined structure `data.calc_pH7.0.final`, trajectory `dump.calc_pH7.0.lammpstrj`, and PLUMED outputs `COLVAR` and `HILLS`.

---

### Phase 5: Post-Processing & Analysis (Phase E)
Analyze the crystalline polymorph fractions (amorphous, cristobalite, tridymite).

Ensure the `results/` directory exists first:
```bash
cd ../
mkdir -p results
cd analysis
```

1. **Simulated X-ray Diffraction (XRD)**:
   ```bash
   python compute_xrd.py ../simulations/data.calc_pH7.0.final --format data -o ../results/xrd_pH7.0
   ```

2. **Steinhardt Order Parameters ($Q_4$/$Q_6$)**:
   ```bash
   python compute_steinhardt.py ../simulations/data.calc_pH7.0.final --format data -o ../results/steinhardt_pH7.0
   ```

3. **Ring-Size Distribution**:
   ```bash
   python ring_analysis.py ../simulations/data.calc_pH7.0.final --format data -o ../results/rings_pH7.0
   ```

4. **Plot Combined Comparison Results**:
   Once you have run the analysis steps for all pH values (e.g. 6.0, 6.5, 7.0, 7.5, 8.0), generate overlay and comparison plots:
   ```bash
   python plot_results.py --results-dir ../results
   ```
* **Output**: Generates comparison plots (`xrd_comparison.png`, `phase_vs_pH.png`, `rings_comparison.png`, `steinhardt_comparison.png`) in the `results/` folder.

### Portability Note
All binaries (LAMMPS, PLUMED, OpenMPI) are provided pre-built by conda-forge.

