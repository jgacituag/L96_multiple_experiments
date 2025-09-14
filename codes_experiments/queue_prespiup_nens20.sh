#!/bin/bash
#!/bin/bash
#PBS -N Full_ObsErr_Prespup_nens20
#PBS -l nodes=1:ppn=128
#PBS -j oe
#PBS -o logs/obs_error_batch_nens20.log
#PBS -V

cd /home/jorge.gacitua/salidas/L96_multiple_experiments/codes_experiments/

# Cargar entorno
source /opt/load-libs.sh 3
mkdir -p logs

export OMP_NUM_THREADS=2
export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"

OBSERRS=("ObsErr0.3" "ObsErr1" "ObsErr5" "ObsErr25")
NTEMPS=(1 2 3) 
NALPHA=(0 1 2 3)
NENS=(20)
frec=(4)
den=(0.5)
PRESPINUP_CYCLES=(200)
PRESPINUP_INFS=(1.2)

THREADS_PER_RUN=8
export OMP_NUM_THREADS=$THREADS_PER_RUN
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"

CORES=128
RESERVE=8                       # keep a few for OS / filesystem
MAX_PAR=$(( ( CORES - RESERVE ) / THREADS_PER_RUN ))
[ "$MAX_PAR" -lt 1 ] && MAX_PAR=1
echo "Node CPUs : $CORES   |  Max concurrent runs: $MAX_PAR"
echo "Thread caps -> OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS"

running=0

# --- Launch ---------------------------------------------------------------
for NENS_VAL in "${NENS[@]}"; do
  for ALPHA in "${NALPHA[@]}"; do
    for OBS in "${OBSERRS[@]}"; do
      for TEMP in "${NTEMPS[@]}"; do
        for FREC in "${frec[@]}"; do
          for DEN in "${den[@]}"; do
            for PRESPINUP in "${PRESPINUP_CYCLES[@]}"; do
              for PRESPINUP_INF in "${PRESPINUP_INFS[@]}"; do

                LOGFILE="logs/${OBS}_NTemp${TEMP}_Nens${NENS_VAL}_Alpha${ALPHA}_FREC${FREC}_DEN${DEN}_PreSpinup${PRESPINUP}_PreSpinupInf${PRESPINUP_INF}.log"
                echo "Launching: ${OBS} NTemp=$TEMP Nens=$NENS_VAL Alpha=$ALPHA FREC=$FREC DEN=$DEN PreSpinup=$PRESPINUP PreSpinupInf=$PRESPINUP_INF  -> $LOGFILE"

                python3 -u ./run_L96_prespinup_series.py \
                  "$OBS" "$TEMP" "$NENS_VAL" "$ALPHA" "$FREC" "$DEN" "$PRESPINUP" "$PRESPINUP_INF" \
                  > "$LOGFILE" 2>&1 &

                running=$((running+1))
                if [ "$running" -ge "$MAX_PAR" ]; then
                  wait    # hold until some finish
                  running=0
                fi

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
