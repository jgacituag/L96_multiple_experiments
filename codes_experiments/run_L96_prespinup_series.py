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
conf.DAConf['ExpLength'] = 40000                           #None use the full nature run experiment. Else use this length.
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
PreSpinupCycles = int(sys.argv[7])
PreSpinupInflation = f"{float(sys.argv[8]):.1f}"
conf.DAConf['PreSpinupCycles']   =   PreSpinupCycles# Number of pre-spinup cycles
conf.DAConf['PreSpinupInflation'] = PreSpinupInflation
out_filename = f'/home/jorge.gacitua/salidas/L96_multiple_experiments/data/LETKF/LETKF_{nature_name}_Nens{nens}_NTemp{ntemp}_alpha{AlphaTempScale}{gec}_40k_Prespinup{PreSpinupCycles}_inf{PreSpinupInflation}.npz'

print(f'\n=== Running experiment: {nature_name} with NTemp={ntemp} ===\n')

# Inicialización
#results = []
AlphaTempList = []

series_analysis_rmse = np.zeros((conf.DAConf['ExpLength']))
series_analysis_sprd = np.zeros((conf.DAConf['ExpLength']))
series_analysis_bias = np.zeros((conf.DAConf['ExpLength']))
series_forecast_rmse = np.zeros((conf.DAConf['ExpLength']))
series_forecast_sprd = np.zeros((conf.DAConf['ExpLength']))
series_forecast_bias = np.zeros((conf.DAConf['ExpLength']))

# BEST Parameter for 80 members and error 5
mult_inf = 1.2
loc_scale = 4.0

conf.DAConf['InfCoefs'] = np.array([mult_inf, 0.0, 0.0, 0.0, 0.0])
conf.DAConf['LocScalesLETKF'] = np.array([loc_scale, -1.0])



result = alm.assimilation_letkf_run(conf)
NormalEnd = result['NormalEnd']
AlphaTempList.append(alm.get_temp_steps(ntemp, conf.DAConf['AlphaTempScale']))

# Extract full time series instead of only averages
series_analysis_rmse = result['XATRmse']
series_forecast_rmse = result['XFTRmse']
series_analysis_bias = result['XATBias']
series_forecast_bias = result['XFTBias']
series_analysis_sprd = result['XATSprd']
series_forecast_sprd = result['XATSprd']

# Optionally save ensemble mean too
analysis_mean = result['XAMean']  # (Nx, time)
forecast_mean = result['XFMean']


# Guardar resultados
np.savez_compressed(
    out_filename,
    NormalEnd=NormalEnd,
    AlphaTempList=AlphaTempList,
    series_analysis_rmse=series_analysis_rmse,
    series_forecast_rmse=series_forecast_rmse,
    series_analysis_sprd=series_analysis_sprd,
    series_forecast_sprd=series_forecast_sprd,
    series_analysis_bias=series_analysis_bias,
    series_forecast_bias=series_forecast_bias,
    analysis_mean=analysis_mean,
    forecast_mean=forecast_mean
)
print(f'\n=== Experiment completed and saved to {out_filename} ===\n')
