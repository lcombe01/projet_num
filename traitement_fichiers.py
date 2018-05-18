#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:22:08 2018

@author: sdiop
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def save_trajectories(grains, t_final, t0 = 0, save = False, filename = None):
    '''
    @briefs: Renvoie les trajectoires [T,X,Y,R] qui sont tous des arrays d'un ensemble de grains initial 'grains', entre les instants 0 et t_final
    @params: 
        grains: liste initiale des grains
        t_final et t0: temps de début et de fin de simulation
        save: sauvegarde en .npz les trajectoires
        filename: nom du fichier a écrire
    '''
    t = t0 #initialisation
    X = [[] for k in range(len(grains))]
    Y = [[] for k in range(len(grains))]
    rayons = [grains[i].rayon for i in range(len(grains))]
    
    temps = []
    T0 = time.time()
    counter = 0          
    
    while t-t0 < t_final:
        for i in range(len(grains)) :
            X[i].append(grains[i].pos[0])
            Y[i].append(grains[i].pos[1])
        temps.append(t)
        if counter%3 == 0: detec = True # la détection des voisins se fait tous les 3*dt
        else: detec = False
        mise_a_jour(detec)
        t += dt #incrementation du temps
        if counter % 30 == 0:
            print('{:.2f} %'.format(100 * (t-t0)/t_final)) #donne 
            if t-t0>0 : # donne l'information temporelle sur la progression du calcul
                T1 = time.time() #mesure le temps actuel
                DeltaT = T1-T0
                Tf = t_final * DeltaT/(t-t0)
                T_rest = Tf - DeltaT # calcule le temps restant
                T_rest = time.gmtime(T_rest)
                print("Left: %i h %i min %i s" % ( T_rest.tm_hour, T_rest.tm_min, T_rest.tm_sec))
        counter += 1
        
    temps = np.array(temps)
    for k in range(len(X)): X[k] = np.array(X[k])
    for k in range(len(Y)): Y[k] = np.array(Y[k])
    rayons = np.array(rayons)
    
    if save == True :
        np.savez_compressed(filename,temps,X,Y,rayons) #sauvegarde un fichier .npz
    return(temps, X, Y, rayons)



def load_trajectories(filename):
    '''
    @brief
        Charge les trajectoires T,X,Y,R sous forme d'array numpy a partir d'un fichier .npz
    @params
        filename : chemin où chercher la simulation .npz
    '''
    T = np.load(filename)['arr_0']
    X = np.load(filename)['arr_1']
    Y = np.load(filename)['arr_2']
    R = np.load(filename)['arr_3']
    return(T,X,Y,R)
    
    
def load_state(T, X ,Y, rayons, counter = -1):
    '''
    @briefs,
        Créée une liste 'grains' a partir des trajectoires T,X,Y,R, à un instant précis
    @params
        T : tableau de temps 1D;
        X : tableau 2D des coordonnées x (chaque ligne est le x(t) d'une particule) ;
        Y : tableau 2D des coordonnées y (chaque ligne est le y(t) d'une particule) ;
        counter : instant observé (par défaut, -1 ce qui correspond au dernier)
    '''
    VX, VY = velocities(T,X,Y)
    X_0 = X.transpose()[counter]
    Y_0 = Y.transpose()[counter]
    VX_0 = VX.transpose()[counter]
    VY_0 = VY.transpose()[counter]
    grains = []
    for k in range(np.shape(X_0)[0]):
        grains.append(grain(X_0[k], Y_0[k], VX_0[k], VY_0[k] , rayons[k]))
    return(grains)
    
    
def velocities(T,X,Y):
    '''
    @briefs,
        Calcule les tabelaux de vitesses VX et VY a partir des trajectoires T,X,Y
    @params
        T : tableau de temps 1D;
        X : tableau 2D des coordonnées x (chaque ligne est le x(t) d'une particule) ;
        Y : tableau 2D des coordonnées y (chaque ligne est le y(t) d'une particule) ;
    '''
    dt = T[1]-T[0]
    VX = np.diff(X)/dt
    VY = np.diff(Y)/dt
    return(VX, VY)
    
    
    
def force_wall(grains, wall_id):
    '''
    @briefs,
        Calcule la force sur une paroi a partir d'une liste grains et de l'identifiant de la paroi
    @params
        grains: liste des grains
        wall_id: identifiant de la paroi (0: gauche; 1: bas; 2: droite)
    '''
    F = 0 
    if wall_id == 1: #force sur la paroi du bas
        for gr in grains:
            R = gr.rayon
            if gr.pos[1] < R : 
                F += Kd_p * (R-gr.pos[1])
    if wall_id == 0: #paroi de gauche
        for gr in grains:
            R = gr.rayon
            if gr.pos[0] < -L/2+R :
                F +=  - Kd_p * (gr.pos[0]-R+L/2)
    if wall_id == 2: #paroi de droite
         for gr in grains:
             R = gr.rayon
             if gr.pos[0] > L/2-R :
                F +=   Kd_p * (gr.pos[0]+R-L/2) 
    return F
    
    
def nb_out(T,X,Y):
    '''
    @briefs
        Calcule le nombre de grains sortis du récipient a chaque instant
    @params
        T : tableau de temps 1D;
        X : tableau 2D des coordonnées x (chaque ligne est le x(t) d'une particule) ;
        Y : tableau 2D des coordonnées y (chaque ligne est le y(t) d'une particule) ;
    '''
    nb = []
    for i in range(len(T)):
        Y_t = Y.transpose()[i]
        nb.append( (Y_t < 0).sum() )
    return np.array(nb)

        
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    