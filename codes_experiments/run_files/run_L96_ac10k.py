#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

# Añadir rutas de módulos
sys.path.append('/home/jorge.gacitua/salidas/L96_multiple_experiments/model/')
sys.path.append('/home/jorge.gacitua/salidas/L96_multiple_experiments/data_assimilation/')

import sensitivity_conf_default as conf
import assimilation_letkf_module as alm

# Leer argumentos
if len(sys.argv) < 3:
    print("Usage: python obs_error_sensitivity.py <ObsErr> <NTemp>")
    sys.exit(1)

obs_err = sys.argv[1]        # e.g., 'ObsErr0.3'
ntemp = int(sys.argv[2])     # e.g., 1, 2, or 3
nens = int(sys.argv[3])      # Number of ensemble members
AlphaTempScale = int(sys.argv[4])  # e.g., 1, 2, or 3
frec = int(sys.argv[5])
den = f"{float(sys.argv[6]):.1f}"  # format to decimal with one decimal place, e.g., '1.0'
base_nature_prefix = f'Paper_Nature_Freq4_Den{den}_Type3'
nature_name = f'{base_nature_prefix}_{obs_err}'
gec = '_NOGEC'
conf.GeneralConf['NatureName'] = nature_name
conf.GeneralConf['ObsFile'] = f'/home/jorge.gacitua/salidas/L96_multiple_experiments/data/Nature/{nature_name}.npz'
conf.DAConf['NTemp'] = ntemp
conf.DAConf['ExpLength'] = 10000                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = nens                                #Number of ensemble members
conf.DAConf['Twin'] = True                                #When True, model configuration will be replaced by the model configuration in the nature run.
conf.DAConf['Freq'] = 4                                   #Assimilation frequency (in number of time steps)
conf.DAConf['TSFreq'] = 4                                 #Intra window ensemble output frequency (for 4D Data assimilation)
conf.DAConf['LocScalesLETKF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['BridgeParam']=0.0                            #Bridging parameter for the hybrid 0-pure LETKF, 1.0-pure ETPF

conf.DAConf['AddaptiveTemp']=False                        #Enable addaptive tempering time step in pseudo time.
conf.DAConf['AlphaTempScale'] = AlphaTempScale            #Scale factor to obtain the tempering factors on each tempering iteration.

conf.DAConf['GrossCheckFactor'] = 1000.0
conf.DAConf['LowDbzPerThresh']  = 1.1

out_filename = f'/home/jorge.gacitua/salidas/L96_multiple_experiments/data/LETKF/LETKF_{nature_name}_Nens{nens}_NTemp{ntemp}_alpha{AlphaTempScale}{gec}_10k.npz'

print(f'\n=== Running experiment: {nature_name} with NTemp={ntemp} ===\n')

# Inicialización
#results = []
AlphaTempList = []
AlphaTempList=[np.array([1]) , np.array([90,1]) , np.array([90,5,1])  , np.array([90,10,5,1]) ]
mult_inf_range = np.arange(1.0, 1.6, 0.05)
loc_scale_range = np.arange(0.5, 5.0, 0.5)

total_analysis_rmse = np.zeros((len(mult_inf_range), len(loc_scale_range)))
total_analysis_sprd = np.zeros((len(mult_inf_range), len(loc_scale_range)))
total_analysis_bias = np.zeros((len(mult_inf_range), len(loc_scale_range)))
total_forecast_rmse = np.zeros((len(mult_inf_range), len(loc_scale_range)))
total_forecast_sprd = np.zeros((len(mult_inf_range), len(loc_scale_range)))
total_forecast_bias = np.zeros((len(mult_inf_range), len(loc_scale_range)))
NormalEnd = np.zeros((len(mult_inf_range), len(loc_scale_range)))
# Loop principal
iiters = 1
for iinf, mult_inf in enumerate(mult_inf_range):
    for iloc, loc_scale in enumerate(loc_scale_range):

        print(f'iteration {iiters} of {len(mult_inf_range) * len(loc_scale_range)}')
        conf.DAConf['InfCoefs'] = np.array([mult_inf, 0.0, 0.0, 0.0, 0.0])
        conf.DAConf['LocScalesLETKF'] = np.array([loc_scale, -1.0])

        result = alm.assimilation_letkf_run(conf)
        #results.append(result)
        NormalEnd[iinf, iloc] = result['NormalEnd']
        AlphaTempList.append(alm.get_temp_steps(ntemp, conf.DAConf['AlphaTempScale']))

        total_analysis_rmse[iinf, iloc] = np.mean(result['XASRmse'])
        total_forecast_rmse[iinf, iloc] = np.mean(result['XFSRmse'])
        total_analysis_sprd[iinf, iloc] = np.mean(result['XASSprd'])
        total_forecast_sprd[iinf, iloc] = np.mean(result['XFSSprd'])
        total_analysis_bias[iinf, iloc] = np.mean(result['XASBias'])
        total_forecast_bias[iinf, iloc] = np.mean(result['XFSBias'])
        iiters += 1

# Guardar resultados
np.savez_compressed(
    out_filename,
    NormalEnd=NormalEnd,
    mult_inf_range=mult_inf_range,
    loc_scale_range=loc_scale_range,
    AlphaTempList=AlphaTempList,
    total_analysis_rmse=total_analysis_rmse,
    total_forecast_rmse=total_forecast_rmse,
    total_analysis_sprd=total_analysis_sprd,
    total_forecast_sprd=total_forecast_sprd,
    total_analysis_bias=total_analysis_bias,
    total_forecast_bias=total_forecast_bias
)

print(f'\n=== Experiment completed and saved to {out_filename} ===\n')
