# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:58:43 2024

@author: Usama
"""
import numpy as np
import scipy
from matplotlib import pyplot as plt
frequency=False
mat_data = scipy.io.loadmat('results.mat')
Xrecon=mat_data['Xvmc2']
Xtrue=mat_data['Xtrue']
Xrecon=Xrecon
s=mat_data['s'][0]
fig, ax = plt.subplots(figsize=(8,6),dpi=1000)
for i in range(Xrecon.shape[0]):
#for i in range(3):
    #i=-20
    #[print(au_to_wvno(Xtrue[i,j])) for j in range(56)]
    ax.plot(s,Xrecon[i,:],linestyle='solid',label='Matrix Elements',lw=1.35,marker=r'$\odot$')
    #ax.plot(Snew,Xtrue[i,:],linestyle='solid',label='Matrix Elements',lw=1.35,marker=r'$\odot$')

    if i==0:
        #ax.legend(fontsize=20)
        ax.set_xlabel('s',fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15) #np.linspace(0,0.4,5),
        ax.set_ylabel('$\omega_i^2$',fontsize=20)
        if frequency:
            ax.set_ylabel('$\omega_i$',fontsize=20)
#        ax.set_title('C2H6_C6H5_noproj'+filename.split('_')[3])
        ax.grid()
        
        
fig, ax = plt.subplots(figsize=(8,6),dpi=1000)
for i in range(Xtrue.shape[0]):
#for i in range(3):
    #i=-20
    #[print(au_to_wvno(Xtrue[i,j])) for j in range(56)]
    ax.plot(s,Xtrue[i,:],linestyle='solid',label='Matrix Elements',lw=1.35,marker=r'$\odot$')
    #ax.plot(Snew,Xtrue[i,:],linestyle='solid',label='Matrix Elements',lw=1.35,marker=r'$\odot$')

    if i==0:
        #ax.legend(fontsize=20)
        ax.set_xlabel('s',fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15) #np.linspace(0,0.4,5),
        ax.set_ylabel('$\omega_i^2$',fontsize=20)
        if frequency:
            ax.set_ylabel('$\omega_i$',fontsize=20)
#        ax.set_title('C2H6_C6H5_noproj'+filename.split('_')[3])
        ax.grid()