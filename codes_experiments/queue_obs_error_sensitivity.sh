#!/bin/bash
#PBS -N obs_error_batch
#PBS -l nodes=1:ppn=64
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o logs/obs_error_batch.log
#PBS -V

cd /home/jorge.gacitua/experimentos/L96_multiple_experiments/experiment_codes/

# Cargar entorno
source /opt/load-libs.sh 3
mkdir -p logs
export OMP_NUM_THREADS=5
export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"

OBSERRS=("ObsErr0.3" "ObsErr1" "ObsErr5" "ObsErr25")
NTEMPS=(1 2 3)

for OBS in "${OBSERRS[@]}"; do
  for TEMP in "${NTEMPS[@]}"; do
    LOGFILE="logs/${OBS}_NTemp${TEMP}.log"
    echo "Running $OBS NTemp=$TEMP"
    python3 -u ./obs_error_sensitivity.py $OBS $TEMP > $LOGFILE 2>&1 &
  done
done

wait
echo "=== All experiments finished ==="