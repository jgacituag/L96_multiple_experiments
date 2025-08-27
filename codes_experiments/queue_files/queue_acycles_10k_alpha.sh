#!/bin/bash
#PBS -N acycles_10k_alpha
#PBS -l nodes=1:ppn=120
#PBS -j oe
#PBS -o logs/obs_error_batch.log
#PBS -V

cd /home/jorge.gacitua/salidas/L96_multiple_experiments/codes_experiments/

# Cargar entorno
source /opt/load-libs.sh 3
mkdir -p logs
export OMP_NUM_THREADS=10
export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"

OBSERRS=("ObsErr5")
#OBSERRS=("ObsErr0.3" "ObsErr1" "ObsErr5" "ObsErr25")
NTEMPS=(1 2 3) 
NALPHA=(0 1 2 3)
NENS=(20)
frec=(4)
den=(1.0)

for NENS_VAL in "${NENS[@]}"; do
  for ALPHA in "${NALPHA[@]}"; do
    for OBS in "${OBSERRS[@]}"; do
      for TEMP in "${NTEMPS[@]}"; do
        for FREC in "${frec[@]}"; do
          for DEN in "${den[@]}"; do
            LOGFILE="logs/${OBS}_NTemp${TEMP}_Nens${NENS_VAL}_Alpha${ALPHA}_FREC${FREC}_DEN${DEN}_10k.log"
            echo "Running ${OBS} NTemp=$TEMP Nens=$NENS_VAL Alpha=$ALPHA FREC=$FREC DEN=$DEN"
            python3 -u ./run_L96_ac10k.py $OBS $TEMP $NENS_VAL $ALPHA $FREC $DEN> $LOGFILE 2>&1 &
          done
        done
      done
    done
  done
done


wait
echo "=== All experiments finished ==="
