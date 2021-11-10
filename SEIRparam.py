# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:16:50 2020

@author: arioldid
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import theano

def create_data (db):
    db1= db.shift()
    db_data= db.loc[1:60,:]
    db1_data= db1.loc[1:60,:]
    delta=db_data-db1_data
    delta['Sass']= -delta.S
    return db_data, db1_data, delta

def Num(BS_data):
    N=BS_data.S[1]+BS_data.I[1]
    return N


def estimate (name_model, GE_data, GE1_data,GE_delta):
    N=GE_data.S[1]+GE_data.I[1]   
    with pm.Model() as GE_model:
        # Priors for unknown model parameters
        #beta = pm.Normal('beta',mu=0, sd=2)
        ##Beta as gaussian random walk (time varying parameter)
        beta = GaussianRandomWalk('beta', tau=1., shape=len(GE1_data))
        sigma = pm.HalfCauchy('sigma', 1, shape=4)
        alpha = pm.Exponential('alpha', lam=0.2)
        rho = pm.Exponential('rho', lam=0.2)
     
        delta=1/4.25    
        gamma=1/5
        mu1= beta * GE1_data.S * (GE1_data.I / N)
        dSdt = pm.Normal('dSdt', mu=mu1, sd=sigma[0], observed=GE_delta.Sass)
    #    E_part=pm.AR('E_part', invdelta )
    #    E=pm.Deterministic('E', beta * ge_1_data.S *(ge_1_data.I / N) + E_part)
        e0=0
        e1=mu1[0]
        e2=(1-delta)*e1+mu1[1]
        e3= (1-delta)*e2+mu1[2]
        e4=(1-delta)*e3+mu1[3]
        e5=(1-delta)*e4+mu1[4]
        e6=(1-delta)*e5+mu1[5]
        e7=(1-delta)*e6+mu1[6]
        e8=(1-delta)*e7+mu1[7]
        e9=(1-delta)*e8+mu1[8]
        e10=(1-delta)*e9+mu1[9]
        e11=(1-delta)*e10+mu1[10]
        e12=(1-delta)*e11+mu1[11]
        e13=(1-delta)*e12+mu1[12]
        e14=(1-delta)*e13+mu1[13]
        e15=(1-delta)*e14+mu1[14]
        e16=(1-delta)*e15+mu1[15]
        e17=(1-delta)*e16+mu1[16]
        e18=(1-delta)*e17+mu1[17]
        e19=(1-delta)*e18+mu1[18]
        e20=(1-delta)*e19+mu1[19]
        e21=(1-delta)*e20+mu1[20]
        e22=(1-delta)*e21+mu1[21]
        e23=(1-delta)*e22+mu1[22]
        e24=(1-delta)*e23+mu1[23]
        e25=(1-delta)*e24+mu1[24]
        e26=(1-delta)*e25+mu1[25]
        e27=(1-delta)*e26+mu1[26]
        e28=(1-delta)*e27+mu1[27]
        e29=(1-delta)*e28+mu1[28]
        e30=(1-delta)*e29+mu1[29]
        e31=(1-delta)*e30+mu1[30]
        e32=(1-delta)*e31+mu1[31]
        e33=(1-delta)*e32+mu1[32]
        e34=(1-delta)*e33+mu1[33]
        e35=(1-delta)*e34+mu1[34]
        e36=(1-delta)*e35+mu1[35]
        e37=(1-delta)*e36+mu1[36]
        e38=(1-delta)*e37+mu1[37]
        e39=(1-delta)*e38+mu1[38]
        e40=(1-delta)*e39+mu1[39]
        e41=(1-delta)*e40+mu1[40]
        e42=(1-delta)*e41+mu1[41]
        e43=(1-delta)*e42+mu1[42]
        e44=(1-delta)*e43+mu1[43]
        e45=(1-delta)*e44+mu1[44]
        e46=(1-delta)*e45+mu1[45]
        e47=(1-delta)*e46+mu1[47]
        e48=(1-delta)*e47+mu1[48]
        e49=(1-delta)*e48+mu1[49]
        e50=(1-delta)*e49+mu1[50]
        e51=(1-delta)*e50+mu1[51]
        e52=(1-delta)*e51+mu1[52]
        e53=(1-delta)*e52+mu1[53]
        e54=(1-delta)*e53+mu1[54]
        e55=(1-delta)*e54+mu1[55]
        e56=(1-delta)*e55+mu1[56]
        e57=(1-delta)*e56+mu1[57]
        e58=(1-delta)*e57+mu1[58]
        e59=(1-delta)*e58+mu1[59]
    
    
        er_mat = tt.stack([e0, e1, e2, 	e3, e4, 	e5, 	e6, 	e7, 	e8, 	e9, 	e10, 	e11, 	e12, 	e13, 	e14, 	e15, 	e16, 	e17, 	e18, 	e19, 	e20, 	e21, 	e22, 	e23, 	e24,e25, 	e26, 	e27, 	e28, e29, 	e30, 	e31,   e32, 		e33, 	e34, 	e35, 	e36, 	e37, 	e38, 	e39, 	e40, 	e41, 	e42, 	e43, 	e44, 	e45, 	e46, 	e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, e59], axis= 1)
        
        mu3= delta * er_mat - (1-alpha) *gamma * GE1_data.I - alpha * rho * GE1_data.I 
        dIdt = pm.Normal('dIdt', mu=mu3, sd=sigma[1], observed=GE_delta.I)
        mu4=  (1-alpha) * gamma * GE1_data.I
        dRdt = pm.Normal('dRdt', mu=mu4, sd=sigma[2], observed=GE_delta.R)
        mu5=  alpha * rho * GE1_data.I 
        dDdt = pm.Normal('dDdt', mu=mu5, sd=sigma[3], observed=GE_delta.D)
    
        return name_model, N  
 
#    dSdt = -beta(t) * S * I / N
#    dEdt = beta(t) * S * I / N - delta * E
#    dIdt = delta * E - (1 - alpha(t)) * gamma * I - alpha(t) * rho * I
#    dRdt = (1 - alpha(t)) * gamma * I
#    dDdt = alpha(t) * rho * I
  
    
TI_data, TI1_data, TI_delta = create_data(TIcsv)
TI_model, N=estimate(model, TI_data, TI1_data, TI_delta  )   


GE_data, GE1_data, GE_delta = create_data(GEcsv)
   GE_model=estimate(model, GE_data, GE1_data, GE_delta  )   
  
    
with GE_model:
    v_params = pm.fit(n=200000,method='advi')



trace_advi = v_params.sample(10000)
pm.summary(trace_advi)
result= pm.summary(trace_advi, varnames=['beta'])
##
plt.plot(result['mean']/0.20)
rGE=result['mean']/0.20



rBS=pd.DataFrame(rBS).reset_index
rBS['t']=TI_data['t']
bBS=result['mean']
