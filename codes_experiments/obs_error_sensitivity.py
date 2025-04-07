#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

# Añadir rutas de módulos
sys.path.append('/home/jorge.gacitua/experimentos/L96_multiple_experiments/model/')
sys.path.append('/home/jorge.gacitua/experimentos/L96_multiple_experiments/data_assimilation/')

import sensitivity_conf_default as conf
import assimilation_letkf_module as alm

# Leer argumentos
if len(sys.argv) < 3:
    print("Usage: python obs_error_sensitivity.py <ObsErr> <NTemp>")
    sys.exit(1)

obs_err = sys.argv[1]        # e.g., 'ObsErr0.3'
ntemp = int(sys.argv[2])     # e.g., 1, 2, or 3

base_nature_prefix = 'Paper_Nature_Freq4_Den1.0_Type3'
nature_name = f'{base_nature_prefix}_{obs_err}'
AlphaTempScale = 2.0
gec = '_GEC'
conf.GeneralConf['NatureName'] = nature_name
conf.GeneralConf['ObsFile'] = f'/home/jorge.gacitua/experimentos/L96_multiple_experiments/data/Nature/{nature_name}.npz'
conf.DAConf['NTemp'] = ntemp

conf.DAConf['ExpLength'] = None                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 20                                  #Number of ensemble members
conf.DAConf['Twin'] = True                                #When True, model configuration will be replaced by the model configuration in the nature run.
conf.DAConf['Freq'] = 4                                   #Assimilation frequency (in number of time steps)
conf.DAConf['TSFreq'] = 4                                 #Intra window ensemble output frequency (for 4D Data assimilation)
conf.DAConf['LocScalesLETKF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['BridgeParam']=0.0                            #Bridging parameter for the hybrid 0-pure LETKF, 1.0-pure ETPF

conf.DAConf['AddaptiveTemp']=False                        #Enable addaptive tempering time step in pseudo time.
conf.DAConf['AlphaTempScale'] = AlphaTempScale            #Scale factor to obtain the tempering factors on each tempering iteration.
conf.DAConf['GrossCheckFactor'] = 15.0                    #Optimized gross error check
conf.DAConf['LowDbzPerThresh']  = 1.1                     #Optimized Low ref thresh.

out_filename = f'/home/jorge.gacitua/experimentos/L96_multiple_experiments/data/LETKF/LETKF_{nature_name}_NTemp{ntemp}_alpha{AlphaTempScale}{gec}.npz'

print(f'\n=== Running experiment: {nature_name} with NTemp={ntemp} ===\n')

# Inicialización
results = []
AlphaTempList = []

mult_inf_range = np.arange(1.0, 1.6, 0.05)
loc_scale_range = np.arange(0.5, 5.0, 0.5)

total_analysis_rmse = np.zeros((len(mult_inf_range), len(loc_scale_range)))
total_analysis_sprd = np.zeros((len(mult_inf_range), len(loc_scale_range)))
total_forecast_rmse = np.zeros((len(mult_inf_range), len(loc_scale_range)))
total_forecast_sprd = np.zeros((len(mult_inf_range), len(loc_scale_range)))

# Loop principal
iiters = 1
for iinf, mult_inf in enumerate(mult_inf_range):
    for iloc, loc_scale in enumerate(loc_scale_range):
        print(f'iteration {iiters} of {len(mult_inf_range) * len(loc_scale_range)}')
        conf.DAConf['InfCoefs'] = np.array([mult_inf, 0.0, 0.0, 0.0, 0.0])
        conf.DAConf['LocScalesLETKF'] = np.array([loc_scale, -1.0])

        result = alm.assimilation_letkf_run(conf)
        results.append(result)
        AlphaTempList.append(alm.get_temp_steps(ntemp, conf.DAConf['AlphaTempScale']))

        total_analysis_rmse[iinf, iloc] = np.mean(result['XASRmse'])
        total_forecast_rmse[iinf, iloc] = np.mean(result['XFSRmse'])
        total_analysis_sprd[iinf, iloc] = np.mean(result['XASSprd'])
        total_forecast_sprd[iinf, iloc] = np.mean(result['XFSSprd'])

        iiters += 1

# Guardar resultados
np.savez_compressed(
    out_filename,
    results=results,
    mult_inf_range=mult_inf_range,
    loc_scale_range=loc_scale_range,
    AlphaTempList=AlphaTempList,
    total_analysis_rmse=total_analysis_rmse,
    total_forecast_rmse=total_forecast_rmse,
    total_analysis_sprd=total_analysis_sprd,
    total_forecast_sprd=total_forecast_sprd
)

print(f'\n=== Experiment completed and saved to {out_filename} ===\n')