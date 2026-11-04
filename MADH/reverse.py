#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:01:16 2021

@author: selinbac
"""

import pandas as pd
import numpy as np
import scipy.io as spio
#import pickle5 as pickle
import matplotlib.pyplot as plt

name='pkl/MADH-od2_dft2svp_step50_disp5000-extend-odd_num100_reversed.pkl'
alldf = pd.read_pickle(name)

newdf = alldf

#newdf.s=-1*alldf.s
#newdf.index=-1*alldf.index

newdf=newdf.sort_index()

new_name=name.split('.')[0]+'sorted'+'.pkl'
newdf.to_pickle(new_name)

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



alldf2 = newdf
#print(alldf.Eigenvalues)
##SJQ Plotting eigenvalues vs. step
#prefix='Sn2C3_ordered_dft2svp_step100_disp5000_num100_verify'
Xtrue, energies = get_eigenvalue_matrix(alldf2,'All')
Xtrue = np.array(Xtrue)


Xtrue = Xtrue.T
print(Xtrue.shape)
fig, ax = plt.subplots(figsize=(8,12),dpi=200)
#ax.set_yscale('log')
#ax.set_ylim([1.e-8,1.e+2])
#print(Xtrue.shape[0])
for i in range(Xtrue.shape[0]):
    ax.plot(alldf.s,Xtrue[i,:],linestyle='solid',label='Matrix Elements',lw=1.35,marker=r'$\odot$')
    if i==0:
        ax.legend()
        ax.set_xlabel('s')
        ax.set_ylabel('Frequency')
#        ax.set_title('C2H6_C6H5_noproj'+filename.split('_')[3])
        ax.grid()

