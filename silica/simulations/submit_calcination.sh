#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=168:00:00

PH=$1

if [ -z "$PH" ]; then
    echo "Usage: sbatch --job-name=calc_pHX.X --output=out/slurm_calc_pHX.X_%j.out submit_calcination.sh <pH_value>"
    exit 1
fi

echo "=== Starting Calcination for pH $PH ==="
echo "Date: $(date)"

# Sourcing conda
if [ -f $HOME/.local/miniconda3/etc/profile.d/conda.sh ]; then
    source $HOME/.local/miniconda3/etc/profile.d/conda.sh
elif [ -f $HOME/miniconda3/etc/profile.d/conda.sh ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi
conda activate rki

# Create specific PLUMED dat to prevent race conditions
cat plumed.dat | sed "s/si_atoms.ndx/si_atoms_pH${PH}.ndx/g" | sed "s/out\/HILLS/out\/HILLS_pH${PH}/g" | sed "s/out\/COLVAR/out\/COLVAR_pH${PH}/g" > plumed_pH${PH}.dat

# Generate index file for PLUMED
python3 ../scripts/gen_plumed_ndx.py out/data.vash_vashishta_pH${PH} -o si_atoms_pH${PH}.ndx

# Run LAMMPS with MPI
export PLUMED_KERNEL="/clusterfs/students/asaputra/opt/plumed-2.10.0-conda/lib/libplumedKernel.so"
mpirun -np $SLURM_NTASKS lmp -in calcination_metad.in -var pH $PH -var seed $RANDOM

echo "=== Calcination Completed for pH $PH ==="
echo "Date: $(date)"
