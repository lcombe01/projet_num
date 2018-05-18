#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 22:46:41 2018

@author: sdiop
"""


L = 90
H = 80
e = 0
g = 1
y_min = -1

# paramètres des contacts
Kc = 1e4
eta = 0.8
Kd = eta**2 * Kc

Kc_p = Kc
eta_p = eta
Kd_p = Kd

Kt = 10
mu = 0.7
mu_p = mu
Kt_p = Kt


dt = 1e-3

paroi = True
gravite = True
remove = False


if __name__=='__main__':
    
    grains = []
    for i in range(0,42):
        for j in range(0,30):
            grains.append(grain(1+(i-21)*2.1+0.1*(rd.random()-0.5) ,1.4+j*2.6+(i%2)*R+0.2*(rd.random()-0.5),1*(rd.random()-0.5),1*(rd.random()-1)))
    mise_a_jour()

    T,X,Y,R = save_trajectories(grains,20,save=True,filename='sedim_1260')