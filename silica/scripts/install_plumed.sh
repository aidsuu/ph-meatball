#!/bin/bash
set -e

# Define directories
SOURCE_DIR="/clusterfs/students/asaputra/opt/sources"
INSTALL_DIR="/clusterfs/students/asaputra/opt/plumed-2.10.0-symfunc"
TARBALL="plumed-src-2.10.0.tgz"
URL="https://github.com/plumed/plumed2/releases/download/v2.10.0/plumed-src-2.10.0.tgz"

echo "=== Starting PLUMED 2.10.0 compilation with all modules ==="
mkdir -p "$SOURCE_DIR"
cd "$SOURCE_DIR"

# Download tarball if not present
if [ ! -f "$TARBALL" ]; then
    echo "Downloading PLUMED source..."
    wget -q "$URL"
fi

# Extract
echo "Extracting source..."
rm -rf plumed-2.10.0
tar xf "$TARBALL"
cd plumed-2.10.0

# Configure with all modules enabled (enables symfunc)
echo "Configuring PLUMED..."
./configure --prefix="$INSTALL_DIR" --enable-modules=all

# Build and Install
echo "Compiling PLUMED (using 8 threads)..."
make -j 8

echo "Installing PLUMED..."
make install

echo "=== PLUMED installation complete! ==="
echo "Installed at: $INSTALL_DIR"
