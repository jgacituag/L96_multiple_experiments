#!/bin/bash
#PBS -N obs_error_batch_n60
#PBS -l nodes=1:ppn=64
#PBS -j oe
#PBS -o logs/obs_error_batch.log
#PBS -V

cd /home/jorge.gacitua/experimentos/L96_multiple_experiments/codes_experiments/

# Cargar entorno
source /opt/load-libs.sh 3
mkdir -p logs
export OMP_NUM_THREADS=20
export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"

#OBSERRS=("ObsErr0.3" "ObsErr1" "ObsErr5" "ObsErr25")
OBSERRS=("ObsErr5")
NTEMPS=(1 2 3)
NALPHA=(2)
NENS=(60)

for NENS_VAL in "${NENS[@]}"; do
  for ALPHA in "${NALPHA[@]}"; do
    for OBS in "${OBSERRS[@]}"; do
      for TEMP in "${NTEMPS[@]}"; do
        LOGFILE="logs/${OBS}_NTemp${TEMP}_Nens${NENS_VAL}_Alpha${ALPHA}.log"
        echo "Running ${OBS} NTemp=$TEMP Nens=$NENS_VAL Alpha=$ALPHA"
        python3 -u ./obs_error_sensitivity.py $OBS $TEMP $NENS_VAL $ALPHA > $LOGFILE 2>&1 &
      done
    done
  done
done

wait
echo "=== All experiments finished ==="