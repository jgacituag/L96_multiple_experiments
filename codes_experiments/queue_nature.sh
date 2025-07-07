#!/bin/bash
#PBS -N nature_run_batch
#PBS -l nodes=1:ppn=64
#PBS -j oe
#PBS -o logs/nature_batch.log
#PBS -V

cd /home/jorge.gacitua/salidas/L96_multiple_experiments/codes_experiments/

# Cargar entorno
source /opt/load-libs.sh 3
mkdir -p logs
export OMP_NUM_THREADS=5
export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"


LOGFILE="logs/nature.log"
python3 -u ./multiple_nature_run.py > $LOGFILE 2>&1 &


wait
echo "=== All experiments finished ==="