#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:22:20 2023
To calculate rates with SCT and ZCT
@author: selinbac
"""

#INPUTS
Trange=[200]
calculate_ZCT=True
calculate_SCT=False

use_trueVAGandE0=True #false if trial VAG and E0 wanted in ZCT calculations


######


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import seaborn as sns
from numpy.linalg import norm
from calcFreeEnergy import h, c, Na, kcaltokJ, kb, R, amu, htokcal
from calcFreeEnergy import ZPE_calc, enthalpy_entropy, free_energy, au_to_wvno
from calcFreeEnergy import free_energy_vtst
import numpy as np
from matplotlib.lines import Line2D
from scipy.io import loadmat
import scipy.io as spio
import SCT
import os
pd.options.mode.chained_assignment = None
system='MADH'
names=[]
for x in os.listdir('.'):
    if ".mat" in x: 
        names.append(x)




names=['results3.mat']

        
for filename in names:
    print(filename)
    print('pkls/'+filename.split('.')[0].split('results_')[-1].split('_K')[0]+'.pkl')
    alldf = pd.read_pickle('MatPKL/'+system+'.pkl')
            
            
    evals_proj=alldf['Eigenvalues']
    ZCT_true=[]
    SCT_true=[]
    
    svals=alldf['s']
    Vmep=alldf['Energy']
    proj=alldf['Projection'] 
    Hessians=alldf['Hessian']
    #evals_proj=alldf['Eigenvalues']
    
    #Hessians=[]
    evecs_proj=alldf['Eigenvectors']

    results=spio.loadmat(filename)       
    Xfinal = np.array(results['Xvmc2'])
    Xtrue=  np.array(results['Xtrue'])
    
    
    
    for j,i in enumerate(evals_proj.index):
        evals_proj[i]= Xfinal.transpose()[j]
    

    if calculate_ZCT:
        #alldf=alldf.dropna()
        for T in Trange:
            ZCT_tr,E0_true,VAG_true,SAG_true,V_aG_true,BmFs,Ks=SCT.sct(T,Vmep,svals,'calc','calc','calc',proj,Hessians,evals_proj,evecs_proj,if_ZCT=True)
            print('ZCT at',T,'K :',ZCT_tr)
            ZCT_true.append(ZCT_tr)
    
    if calculate_SCT:
        #alldf=alldf.dropna()
        for T in Trange:
            SCT_tr,E0_true,VAG_true,SAG_true,V_aG_true,BmFs,Ks=SCT.sct(T,Vmep,svals,'calc','calc','calc',proj,Hessians,evals_proj,evecs_proj)
            print('SCT at',T,'K :',SCT_tr)
            SCT_true.append(SCT_tr)
    for j,i in enumerate(evals_proj.index):
         evals_proj[i]= Xtrue.transpose()[j]   
    if calculate_ZCT:
        #alldf=alldf.dropna()
        for T in Trange:
            ZCT_true,E0_true,VAG_true,SAG_true,V_aG_true,BmFs,Ks=SCT.sct(T,Vmep,svals,'calc','calc','calc',proj,Hessians,evals_proj,evecs_proj,if_ZCT=True)
            print('True ZCT at',T,'K :',ZCT_true)
            #ZCT_true.append(ZCT_true)
    
    def get_eigenvalue_matrix(df,tag='All'):
        evals = []
        energies = []
        if tag in ['R','P','T']:
            for index,row in df.iterrows():
                if row.tag==tag:
                    evalrow = []
                    for x in row.Eigenvalues:
                        #if abs(x)>cutoff:
                        evalrow.append(x)
                    evals.append(evalrow)
                    energies.append(row.Energy)
        elif tag=='All':
            for index,row in df.iterrows():
                evalrow=[]
                for x in row.Eigenvalues:
                    #if abs(x)>cutoff:
                    evalrow.append(x)
                evals.append(evalrow)
                energies.append(row.Energy)
        elif tag=='Rhalf': 
            for index,row in df.iterrows():
                if index <=0:
                    evalrow=[]
                    for x in row.Eigenvalues:
                        #if abs(x)>cutoff:
                        evalrow.append(x)
                    evals.append(evalrow)
                    energies.append(row.Energy)
        elif tag=='Phalf': 
            for index,row in df.iterrows():
                if index >=0:
                    evalrow=[]
                    for x in row.Eigenvalues:
                        #if abs(x)>cutoff:
                        evalrow.append(x)
                    evals.append(evalrow)
                    energies.append(row.Energy)
        return evals, energies
    
    
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 15}
    
    matplotlib.rc('font', **font)
    
    indreact = -1
    
    #alldf = pd.read_pickle(pkl)
    #Xtrue, Energies = get_eigenvalue_matrix(alldf,'All')
    #Xtrue = np.array(Xtrue)
    Xtrue=Xfinal.transpose()
    Energies=np.array(Vmep)
    
    ntrials = len(alldf)
    
    def count_acc(dfcol,acc):
        count = 0
        for val in dfcol:
            if abs(val)<=acc:
                count+=1
        return count
    
    #######
    #THERMO
    #######
    '''
    #ZPE
    ZPE_true = []
    for column in Xtrue.T: 
        cm = np.array([au_to_wvno(x) for x in column])# if abs(x)>cutoff])
        zpe = ZPE_calc(cm)
        ZPE_true.append(zpe)
    ZPE_true = np.array(ZPE_true)
    
    #Free Energy
    G_true = []
    cm_true = []
    for index, column in enumerate(Xtrue): 
        cm = np.array([au_to_wvno(x) for x in column]) #if abs(x)>cutoff])
        
        freeEgyTemp = []
        for T in Trange: 
            H, S = enthalpy_entropy(cm,T)
            #print(index,T,Energies)
            #freeEgyTemp.append(free_energy(Energies[index],H,S,T))
            freeEgyTemp.append(free_energy_vtst(Energies[index],cm,T))
        G_true.append(freeEgyTemp)
        cm_true.append(cm)
    G_true = np.array(G_true).T 
    
    transpose = True
    thermodict = []
    error_init = []
    error_final = []
    all_colerr_init = []
    all_colerr_vmc = []
    ZPE_all=[]
    
    use_matlab_data=False
    labels=[]
    for Tindex, T in enumerate(Trange): 
        label = 'G, True, '+str(T)+'K'
        labels.append(label)
        
    
    G_true_dict={labels[0]:G_true[0,:],
                labels[1]:G_true[1,:],}
    G_true_df=pd.DataFrame(G_true_dict)
    
    
    rate_true=[]
    
    deltaG_allT=[]
    deltaG_true=[]
    
    for i,T in enumerate(Trange):
        deltaG_true.append(max(G_true[i])-G_true[i][0])
        rate_true.append(kb*T/h*np.exp(-deltaG_true[i]*kcaltokJ*1000/(R*T)))
    '''
    #print('Rate without tunneling:',Trange,rate_true)
    #print('Rate with ZCT:',Trange,np.multiply(rate_true,ZCT_true))
    #print('Rate with ZCT:',Trange,np.multiply(rate_true,SCT_true))
