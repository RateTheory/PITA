#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:49:37 2023
SCT with un-projected
@author: selinbac
"""

#mu is the Da equivalent of electron rest mass

def sct(T,Vmep,s,E0,VAG,SAG,grads,Hessians,evals,evecs,if_ZCT=False, print_bmf=False):
    import numpy as np
    from spline import VadiSpline
    from scipy.interpolate import UnivariateSpline as USpline
    from scipy.special import logsumexp
    import decimal
    #CONSTANTS
    h = 6.62617353e-34
    c = 2.99792458e10
    Na = 6.02204531e23
    kcaltokJ = 4.184
    kb = 1.3806244e-23
    htokcal = 627.5096  #for kcal/mol
    Jtoh=4.359197907585004e-18
    ratio=9.10938356E-031/1.66053904E-027
    inversecm_to_au=4.55634e-6 
    EPS_MEPE = 1e-8
    F=len(np.array(evals)[0])
    V_aG=[]
    
    s_list=np.array(s)
    index=s.index
    
    def BmF_func(s,F,grad,evecsall):
        norm = np.linalg.norm(grad)
        if s<0:
            v0   = [gi/norm for gi in grad]
        else:
            v0   = [-gi/norm for gi in grad]
        
        evecsall=np.array(evecsall)
        F=np.matrix(F)
        BmF= (evecsall)*F*np.matrix(v0).transpose()/norm
        # Page-McIver Method to compute BmF
        #BmF= np.transpose(evecsall)*F*np.matrix(v0).transpose()/norm
        
      
        BmF=-np.array(BmF).transpose()[0]
        return BmF
    
    def calc_mu(ratio):
        mu=1/ratio
        return mu
    
    def prob(E, VAG, E0, SAG):
        
        if E<E0:
            pE=0
            print('issues with E0!!!')
        elif E0<=E<=VAG:
            pE=1/(1+np.exp(2*get_theta(E,s_list,muSpl,VadiSpl,SAG,VAG)))
        elif 2*VAG-E0<E:
            pE=1
            print('issues with upper limit!!')
        else:
            pE=1-prob(2*VAG-E, VAG, E0, SAG)
        return pE
    
    def kappa_integrand(E,VAG,E0,SAG,beta):
        
        intgrd=float(decimal.Decimal(-beta*E).exp()/decimal.Decimal(-beta*VAG).exp())*prob(E,VAG,E0,SAG)
        return  intgrd
    
    def kappa_integral(E,svals,lmueff,VadiSpl,SAG,VAG,E0,beta,limit):
        args = (VAG,E0,SAG,beta)
        integral=intg_trap(kappa_integrand, E0, limit, n=80, args=args)
        return integral
    
    def kappa_integrand2(E,VAG,E0,SAG,beta):
        intgrd=np.sinh(-beta*abs(E-VAG))*prob(E,VAG,E0,SAG)
        return  intgrd
    
    def kappa_integral2(E,svals,lmueff,VadiSpl,SAG,VAG,E0,beta):
        args = (VAG,E0,SAG,beta)
        integral=intg_gau(kappa_integrand2, E0, VAG, n=10, args=args)
        return integral
    
    def intg_trap(function,x0,xf,args=None,n=100,dx=None):
        if dx is None:
           dx = 1.0 * (xf-x0)/(n-1)
        else:
           n = int(round((xf-x0)/dx))+1
        xvalues = [x0+ii*dx for ii in range(n)]
        if args is not None: integrand = [function(x,*args) for x in xvalues]
        else               : integrand = [function(x)       for x in xvalues]
        integral  = sum([y*dx for y in integrand])
        return integral
    
    def intg_gau(function,x0,xf,args=None,n=80):
        points, weights = np.polynomial.legendre.leggauss(n)
        suma     = (xf+x0)/2.0
        resta    = (xf-x0)/2.0
        # Points to evaluate and weights
        points   = [resta*xi+suma for xi in points]
        weights  = [wi*resta for wi in weights]
        # calculate integral
        if args is not None: integral = sum([w*function(x,*args) for x,w in zip(points,weights)])
        else               : integral = sum([w*function(x)       for x,w in zip(points,weights)])
        del points
        del weights
        return integral
    
    def theta_integrand(s_i,E,svals,lmueff,VadiSpl):
        mu_new=lmueff(s_i)
        
        if mu_new>1/ratio:
            mu_new=1/ratio
        #return  np.sqrt( 2*mu_new* abs(E-VadiSpl(s_i)) )
        
        if 2*mu_new* abs(E-VadiSpl(s_i))<1e-90:
            return 0
        
        else:
            return  np.sqrt( 2*mu_new* abs(E-VadiSpl(s_i)) )
        
    def get_theta(E,svals,lmueff,VadiSpl,SAG,VAG):
        if abs(E-VAG) < EPS_MEPE: return 0.0
        rpoints = VadiSpline.returnpoints(VadiSpl,E)
        theta=0
        for (si,sj) in rpoints:
            args = (E,svals,lmueff,VadiSpl)
            theta += intg_trap(theta_integrand, si, sj, n=160, args=args)
        return theta

    def au_to_wvno(eigval): 
        #converts from atomic units to wavenumbers
        tocm = 5.89141e-7
        tocm2 = 3.94562
        if eigval>0.:
            return np.sqrt(np.abs(eigval)/tocm)*tocm2
        else:
            return 0.
    
    summ=[]

    for i in s.index:
        for j in range(F):
            if evals[i][j]<0:
                evals[i][j]=0.00037843276850828  #replace evals with 100cm-1
                #evals[i][j]=0.00037843276850828/2  #replace evals with 50cm-1
                #evals[i][j]=abs(evals[i][j]) #replace with absolute value


                
    for i in index:
        harmonic_sum=0
        ref=np.array(Vmep)[0]
        for j in range(len(np.array(evals)[0])):

            harmonic_sum+=0.5*h*c*Na/(1e3*kcaltokJ)*au_to_wvno(evals[i][j])
          
        summ.append(harmonic_sum)
        V_aG.append((Vmep[i]+harmonic_sum/htokcal)-(ref))
    
    
    if VAG=='calc':
        VAG=max(V_aG)
    if SAG=='calc':
        SAG=s_list[V_aG.index(VAG)]
    if E0=='calc':
        if V_aG[-1]-V_aG[0]<0:
            E0=V_aG[0]
        else:
            E0=V_aG[-1]
    limit=2*VAG-E0
    
    VadiSpl=VadiSpline(s,V_aG)
    E1=np.linspace(E0,(limit-E0)*0.1+E0,25)
    E2=np.geomspace((limit-E0)*0.1+E0,limit,25)
    E=np.concatenate([E1,E2])    
    kappa_list=[]
    
    mu=[] 
    tbars=[]
    abars=[]
    BmFs=[]
    Ks=[]
    for i in s.index:
        if s[i]!=0:
            BmF=BmF_func(s[i],Hessians[i],grads[i],evecs[i])
            curv=[j*j for j in BmF]
            curvature=np.sqrt(np.sum(curv))
            BmFs.append(BmF)
            Ks.append(curvature)
            tm=[np.sqrt(ratio/(au_to_wvno(evals[i][j])*inversecm_to_au)) for j in np.arange(0,F,1)]
            tbar=np.power(np.sum([j*j/(curvature*curvature)*np.power(tm[k],-4) for k,j in enumerate(BmF)]),-1/4)
            abar=abs(curvature*tbar)
            tbars.append(tbar)
            abars.append(abar)

    if 0 in s.index:
        dtbarSpl=VadiSpline(np.array(s.drop(0)),tbars)
        dtbards=[dtbarSpl.derivative(i) for i in s.drop(0)]
    else:
        dtbarSpl=VadiSpline(np.array(s),tbars)
        dtbards=[dtbarSpl.derivative(i) for i in s]
           
    mueff=[min(1,np.exp(-2*abars[i]-abars[i]*abars[i]+dtbards[i]*dtbards[i])) for i in range(len(s)-1)]
    mu=[k/ratio for k in mueff]
    
    if if_ZCT:
        mu=np.ones(len(mu))*1822.8884853323707
        
    #print(mu)
    if 0 in s.index:
        muSpl=VadiSpline(np.array(s.drop(0)),mu)
    else:
        muSpl=VadiSpline(np.array(s),mu)
        
    #print('mu is:\n',mu) 

    #slist=np.array(s)
    #slist=np.delete(slist,0)
    #muSpl=USpline(slist,mu)


    beta=1.0 / (kb*T/Jtoh)
    kappa1=beta*kappa_integral(E, s_list, muSpl, VadiSpl, SAG, VAG, E0, beta,limit)
    kappa2=np.exp(-beta*(VAG-E0))
    kappa_list.append(kappa1+kappa2)
    return kappa1+kappa2,max(V_aG[0],V_aG[-1]),max(V_aG),s_list[V_aG.index(max(V_aG))],V_aG,BmFs,Ks
