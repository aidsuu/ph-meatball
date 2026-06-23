#!/bin/bash
#SBATCH --job-name=solgel_pH6.5
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=24:00:00
#SBATCH --output=out/slurm_solgel_pH6.5_%j.out

echo "=== Starting LAMMPS Sol-Gel Run ==="
echo "pH value: 6.5"
echo "Date:     $(date)"

# Sourcing conda
if [ -f $HOME/.local/miniconda3/etc/profile.d/conda.sh ]; then
    source $HOME/.local/miniconda3/etc/profile.d/conda.sh
elif [ -f $HOME/miniconda3/etc/profile.d/conda.sh ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi
conda activate rki

# Run LAMMPS with MPI
mpirun -np $SLURM_NTASKS lmp -in solgel_reaxff.in -var pH 6.5 -var seed 12345

echo "=== LAMMPS Sol-Gel Run Completed ==="
echo "Date:     $(date)"
