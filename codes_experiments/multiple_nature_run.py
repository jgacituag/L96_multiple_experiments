#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import sys
sys.path.append('/home/jorge.gacitua/salidas/L96_multiple_experiments/model/')
sys.path.append('/home/jorge.gacitua/salidas/L96_multiple_experiments/data_assimilation/')
import numpy as np
import nature_module as nature
import default_nature_conf as conf

####################################
# PARAMETERS THAT WILL BE ITERATED
####################################

FreqList=[4]
SpaceDensityList=[1.0]
ObsOpe=[3]
ObsError=[0.3, 1, 5, 25] 
conf.GeneralConf['RandomSeed'] = 10  #Fix random seed

for MyFreq in FreqList :
    for MyDen in SpaceDensityList :
       for MyOO in ObsOpe :
          for MyOE in ObsError :
              conf.ObsConf['Freq'] = MyFreq  
              conf.ObsConf['SpaceDensity'] = MyDen
              conf.ObsConf['Type'] = MyOO
              conf.ObsConf['Error'] = MyOE
              conf.GeneralConf['ExpName']='Freq' + str(MyFreq) + '_Den' + str(MyDen) + '_Type' + str(MyOO) + '_ObsErr' + str(MyOE)  
              conf.GeneralConf['NatureFileName']='Paper_Nature_' + conf.GeneralConf['ExpName'] + '.npz'
              nature.nature_run( conf )
