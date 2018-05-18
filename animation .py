#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:51:32 2018

@author: louis
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.cm as cm
import PIL.Image as Image
import imageio
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mpc



folder = '/home/sdiop/Documents/physique/sable/films2/'


def gif_from_files(filenames, framerate):
    '''@brief Crée un gif depuis un fichier d'images 
    @param filenames: une liste de noms de fichier '/path/filename' '''
    images = []
    N = len(filenames)
    for i,filename in enumerate(filenames):
        images.append(imageio.imread(filename)) #cree une liste d'image imageio 
        print('{:.1f}% de la creation du gif'.format((i/N)*100)) #progressbar
    imageio.mimsave(folder+'/movie.gif', images, duration = 1/framerate) #crée le gif a partir de la liste 



def traj2gif(T,X,Y,R, ratio = 100, gif = True):
    '''
    @brief 
        cree un gif depuis un fichier .npz de trajectoires, du me type que ceux renvoyés par 'trajectoires'
    @params 
        T : array 1D pour le temps ;
        X : array 2D pour les coordonées x (chaque ligne = x d'un grain en particulier) ;
        Y : array 2D pour les coordonées x (chaque ligne = x d'un grain en particulier) ;
        size : taille de la figure ;
        lim : limite en x et y des axes ;
        center : y du centre de la figure ;
        ratio: une image est sauvergardée tout les 'ratio' ; 
        gif: si True, crée un gif à partir des images enregistrées
    '''
    
    plt.ioff()  #éteint le mode interactif qui ralentirait la simulation 
    fig = plt.figure(1,figsize=(6,7)) #init de la figure
    ax = plt.axes(aspect='equal',xlim=(-L/2-5,L/2+5), ylim=(-5,H+5))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$',rotation=0)
    filenames = []
    t_final = T[-1]
    N = np.shape(X)[0] #nombre de grains
    
    for counter in range(np.shape(T)[0]):
        t = T[counter]
        if counter == 0 : T0 = time.time() #mesure le temps de depart
        
            
        if counter % ratio == 0: #la sauvegarde d'image dépend de ratio
            
            for artist in plt.gca().lines + plt.gca().collections: #nettoyage de la figure à chaque incrémentation 
                artist.remove()
            patches = []
            for i in range(N): #crée des patches circle autour de chaque position de chaque grain
                x = X[i][counter]
                y = Y[i][counter]
                ray = R[i]
                circle = pat.Circle((x, y), ray)
                patches.append(circle)
            p = PatchCollection(patches, alpha=0.5, color = 'r',linewidth =0.01) #cree une collection avec les patches 
            ax.add_collection(p) #ajout de la collection sur la figure précedemment nettoyée 
            plt.title('t = {:1f}'.format(t))
                
            
            if paroi == True: #dessine le récipient
                plt.plot([-L/2,-L/2], [0,H],'k')
                plt.plot([L/2,L/2], [0,H],'k')
                plt.plot([-L/2,-e], [0,0],'k')
                plt.plot([e, L/2], [0,0],'k')
            
            fig.savefig(folder+'fig{}.png'.format(counter))    
            filenames.append(folder+'fig{}.png'.format(counter))
            print('{:.2f} % of image creation'.format((t/t_final)* 100))
            
            if t>0 : # donne des infos sur le temps restant 
                T1 = time.time() #mesure le temps actuel 
                DeltaT = T1-T0
                Tf = t_final * DeltaT/t
                T_rest = Tf - DeltaT #calcul le temps qu'il reste
                T_rest = time.gmtime(T_rest)
                print("Left: %i h %i min %i s" % ( T_rest.tm_hour, T_rest.tm_min, T_rest.tm_sec))
            
    if gif == True : #appel gif_from_files et crée le gif 
        print('Creating gif')
        gif_from_files(filenames, ratio) 
            
    
    
    
    
def afficher(grains, disp = 'None'):
    '''@brief: affiche une image des grains avec une colormap telle que la couleur de chaque grain indique la valeur d'un param (vitesse, pression..) pour ce grain 
       @params: grains: liste d'élements de la classe "grain" disp: 'p', 'vy',.. détermine le param à représenter
    '''

    fig = plt.figure(1,figsize=(5,7)) #initialise la figure 
    ax = plt.axes(aspect='equal',xlim=(-L/2-5,L/2+5), ylim=(-5,H+5))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$',rotation=0)
    
    patches = []
    colors = []

    if disp == 'p': #on représente la pression 
        mise_a_jour() #actualisation des grains (permet d'obtenir plus facilement les valeurs de vitesses et forces)
        list_press = [i.pres for i in grains] #liste des presions 
        max_c = max(list_press) #max et min des pressions pour normaliser la cmap
        min_c = min(list_press)
        lab = r'$p$'
        cmap=cm.magma_r
        alpha = 0.7
    if disp == 'vy': #même chose que pour p 
        mise_a_jour()
        list_vit = []
        for i in grains:
            if i.pos[1]>-7: list_vit.append(np.log(np.sqrt(i.vit[1]**2)))
        max_c = max(list_vit)
        min_c = min(list_vit)
        lab = r'$-v_y$'
        cmap=cm.viridis
        alpha = 0.7
    if disp == 'None': #colormap uniforme 
        alpha = 0.5 
        
    for i in grains : #on attribue a chaque grain une couleur en fonction des valeurs calculées precedemment 
        if disp == 'p':
            col = i.pres+400
        if disp == 'vy' :
            col = np.log(np.sqrt(i.vit[1]**2))
            if i.pos[1]<-7: col=0
        if disp == 'None':
            col = 'r'
        circle = pat.Circle(i.pos, i.rayon) #creation des patches avec cette couleur 
        patches.append(circle)
        colors.append(col)
        
    if disp != 'None':
        pc = PatchCollection(patches,alpha=alpha, cmap=cmap, linestyle = 'solid')
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        
        if disp == 'vy': norm= mpc.LogNorm(vmin=np.exp(min_c), vmax=np.exp(max_c)) 
        else : norm = mpc.Normalize(vmin=min_c, vmax=max_c)
        sm = cm.ScalarMappable(norm=norm, cmap = pc.cmap)
        sm.set_array([])
    
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5, aspect=30)
        cb= fig.colorbar(sm, cax=cax, alpha = alpha)
        cax.set_title(lab)
    else:
        pc = PatchCollection(patches,alpha=alpha, color='r', linestyle = 'solid')
        ax.add_collection(pc)
        
    if paroi == True: #draws the recipient
        ax.plot([-L/2,-L/2], [0,H],'k')
        ax.plot([L/2,L/2], [0,H],'k')
        ax.plot([-L/2,-e], [0,0],'k')
        ax.plot([e, L/2], [0,0],'k')
        
    plt.show()


  

    
    
    
 

        