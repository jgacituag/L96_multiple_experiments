#!/bin/bash
#!/bin/bash
#PBS -N 80k_series
#PBS -l nodes=1:ppn=48
#PBS -j oe
#PBS -o logs/obs_error_batch.log
#PBS -V

cd /home/jorge.gacitua/salidas/L96_multiple_experiments/codes_experiments/

# Cargar entorno
source /opt/load-libs.sh 3
mkdir -p logs

export OMP_NUM_THREADS=48
export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"

OBSERRS=("ObsErr5")
NTEMPS=(1 2 3) 
NALPHA=(2)
NENS=(80)
frec=(4)
den=(1.0)
PRESPINUP_CYCLES=(100)
PRESPINUP_INFS=(1.0 1.2)

for NENS_VAL in "${NENS[@]}"; do
  for ALPHA in "${NALPHA[@]}"; do
    for OBS in "${OBSERRS[@]}"; do
      for TEMP in "${NTEMPS[@]}"; do
        for FREC in "${frec[@]}"; do
          for DEN in "${den[@]}"; do
            for PRESPINUP in "${PRESPINUP_CYCLES[@]}"; do
              for PRESPINUP_INF in "${PRESPINUP_INFS[@]}"; do
                LOGFILE="logs/${OBS}_NTemp${TEMP}_Nens${NENS_VAL}_Alpha${ALPHA}_FREC${FREC}_DEN${DEN}_PreSpinup${PRESPINUP}_PreSpinupInf${PRESPINUP_INF}.log"
                echo "Running ${OBS} NTemp=$TEMP Nens=$NENS_VAL Alpha=$ALPHA FREC=$FREC DEN=$DEN PreSpinup=$PRESPINUP PreSpinupInf=$PRESPINUP_INF"
                python3 -u ./run_L96_prespinup_series.py $OBS $TEMP $NENS_VAL $ALPHA $FREC $DEN $PRESPINUP $PRESPINUP_INF> $LOGFILE 2>&1 &
              done
            done
          done
        done
      done
    done
  done
done

wait
echo "=== All experiments finished ==="
