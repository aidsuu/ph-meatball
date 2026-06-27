#!/bin/bash
set -e
source /clusterfs/students/asaputra/miniconda3/etc/profile.d/conda.sh
conda activate rki

SOURCE_DIR="/clusterfs/students/asaputra/opt/sources"
INSTALL_DIR="/clusterfs/students/asaputra/opt/plumed-2.10.0-conda"

mkdir -p "$SOURCE_DIR"
cd "$SOURCE_DIR"

if [ ! -d "plumed-2.10.0" ]; then
    tar xf plumed-src-2.10.0.tgz
fi

cd plumed-2.10.0
make clean || true
./configure --prefix="$INSTALL_DIR" --enable-modules=all CXX=mpicxx CC=mpicc
make -j 12
make install
echo "Done!"
