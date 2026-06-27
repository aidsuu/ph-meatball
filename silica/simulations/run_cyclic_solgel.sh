#!/bin/bash
#SBATCH --job-name=cyc_solgel
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=24:00:00
#SBATCH --partition=qdisk

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rki

pH=$1
cycles=10

if [ -z "$pH" ]; then
    echo "Usage: sbatch run_cyclic_solgel.sh <pH>"
    exit 1
fi

echo "=== Initializing Phase (Compression & Heating) for pH $pH ==="
mpirun -np $SLURM_NTASKS lmp -in solgel_init.in -var pH $pH -log out/log.solgel_pH${pH}.init

for ((i=1; i<=cycles; i++)); do
    echo "=== Cyclic Evaporation: Cycle $i / $cycles ==="
    
    # 1. Evaporate water using Python FIRST (creates vacuum voids)
    python ../scripts/evaporate_water.py current_pH${pH}.data current_pH${pH}.data.tmp
    mv current_pH${pH}.data.tmp current_pH${pH}.data
    
    # 2. Run ReaxFF NPT condensation for 20 ps (compacts the voids and reacts)
    mpirun -np $SLURM_NTASKS lmp -in solgel_cycle.in -var pH $pH -log out/log.solgel_pH${pH}.cycle$i
done

echo "=== Finishing Phase (Cooling to 300 K) ==="
mpirun -np $SLURM_NTASKS lmp -in solgel_finish.in -var pH $pH -log out/log.solgel_pH${pH}.finish

# Clean up temporary current file
rm -f current_pH${pH}.data

echo "=== Cyclic Sol-Gel Complete for pH $pH ==="
