#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:45:34 2018
@author: sdiop
"""

import numpy as np
import random as rd


class grain : #Classe permettant de caractériser chaque grains, qui contient donc toutes les informations utiles (vitesse, position, force subie..)
    '''
    Definition des méthodes et attributs des grains
    '''
    
    def __init__(self, x0,y0,vx0,vy0, ray=0):
        
        if ray==0:
            self.rayon = np.random.normal(1,0.15) #Pour un écoulement polydisperse, permet d'associer un rayon aléatoire a chaque grain
        else:
            self.rayon = ray #Pour un cas monodisperse
        
        self.pos = np.array([x0,y0]) #position à t
        self.vit = np.array([vx0,vy0]) #vitesse à t
        self.acc = np.array([0.,0.], dtype = 'float64') #accélération à t
        
        self.pos_a = np.array([x0,y0]) - np.array([vx0,vy0])*dt  # position à t-dt
        self.pos_n = np.array([x0,y0]) #position à t+dt
        self.vit_n = np.array([vx0,vy0]) #position à t+dt
        
        self.pres = 0.
        self.force = np.array([0,0]) #force_totale(i) a t 
        
        self.voisins = [] #liste des voisins
        self.bords = [] #liste de booléens [gauche,bas,droite], True si contact, False sinon

        
    def detecter_voisins(self):
        '''
        Stock dans self.voisins la liste des grains j en contact avec le grain self
        On teste tous les grains (différents de i) pour voir si ils sont en contact
        '''
        self.voisins=[]
        for k in grains:
            if ((k.pos[0]-self.pos[0])**2+(k.pos[1]-self.pos[1])**2 < (k.rayon + self.rayon)**2) and k != self: #test si il y a contact
                self.voisins.append(k)
                
                
    def detecter_paroi(self):
	'''Stock dans self.bords une liste de booléen indiquant avec quelles parois self est en contact'''
        self.bords = []
        R = self.rayon
        if paroi == False : self.bords = [False, False, False] #paroi est un paramètre. Si paroi == False aucune paroi n'est prise en compte dans la simul. 
        else : #On teste le contact avec les parois dans l'ordre de la liste self.bords
            if self.pos[0] < -L/2 + R:
              self.bords.append(True)
            else:
               self.bords.append(False)
            if self.pos[1] < R and (self.pos[0] < -e or self.pos[0] > e):
               self.bords.append(True)
            else:
               self.bords.append(False)
            if self.pos[0] > L/2 - R:
               self.bords.append(True)
            else:
               self.bords.append(False)
            if self.pos[1]<y_min : 
                self.bords = [False, False, False]



def force_grain(i,j):
    '''
    @Brief: Fonction de calcul de la force due au grain i s'appliquant sur le grain j
    @Param: Deux grains i et j 
     '''
    #On commence par nommer les attributs des grains i et j 
    Ri = i.rayon
    Rj = j.rayon
    
    xi, yi = i.pos
    xj, yj = j.pos
    vxi, vyi = i.vit
    vxj, vyj = j.vit
    
    rij = np.array([xj - xi, yj - yi])
    x, y = rij
    vij = np.array([vxj - vxi, vyj - vyi])
    r = np.linalg.norm(rij)
    
    if Rj+Ri - r < 0: return np.array([0.,0.]) #Pas de contact -> pas de force 
    
    u_r = 1/r * np.array([x, y]) #vecteurs unitaires (u_r: normal, u_t: tangentiel)
    u_t = 1/r * np.array([-y, x])
    
    v_t = np.dot(vij, u_t) #vitesses normales et tangentielles 
    v_r = np.dot(vij, u_r)
    
    if np.dot(vij,u_r) < 0 : #Test pour savoir si on est dans le cas d'une charge ou d'une décharge (cf. rapport, modèle des contacts)
        K = Kc
    else :
        K = Kd
    F_r = K*(Rj+Ri - r) #Application du modèle de contact normal 
    
    if np.abs(Kt * v_t) < np.abs(mu * F_r) : #Test pour déterminer si les frottements sont dans le régime solide ou fluide puis application du modèle tangentiel
        F_t = - Kt * v_t
    else : 
        F_t = - mu * np.abs(F_r) * np.sign(v_t)
    
    j.pres += abs(F_r)/np.sqrt((Rj+Ri - r)*j.rayon) #Permet de stocker la pression subie par le grain
    
    return F_r * u_r + F_t * u_t #Renvoie la force selon u_r et u_t
       




def force_paroi(i):
    '''
    @brief: Fonction de calcul de la force s'appliquant sur le grain i à cause de la paroi
    @params: un grain i 
'''
    R = i.rayon #initialisation
    x,y = i.pos
    vx,vy = i.vit
    f = np.array([0.,0.])
    press = 0.
    
    if i.bords[0] == True : #Test des contacts avec self.bords, puis application du modèle de contact 
        alpha = -(x-R+L/2)
        if vx >= 0 : #La géométrie diffère mais le principe est identique à force_grains 
            fo = np.array([Kd_p*alpha,-min((mu_p*Kc_p*alpha, Kt_p*vy))])
        if vx < 0 :
            fo = np.array([Kc_p*alpha,-min((mu_p*Kd_p*alpha, Kt_p*vy))])
        press += np.linalg.norm(fo)/np.sqrt(alpha)
        f+=fo
    if i.bords[2] == True : #Même chose
        alpha = x+R-L/2
        if vx > 0 :
            fo = np.array([-Kc_p*alpha,-min((mu_p*Kc_p*alpha, Kt_p*vy))])
        if vx <= 0 :
            fo = np.array([-Kd_p*alpha,-min((mu_p*Kd_p*alpha, Kt_p*vy))])
        press += np.linalg.norm(fo)/np.sqrt(alpha)
        f+=fo
    if i.bords[1] == True : #Même chose
        alpha = R-y
        if vy > 0 :
            fo = np.array([-min((mu_p*Kc_p*alpha, Kt_p*vx)), Kd_p*alpha])
        if vy <= 0 :
            fo = np.array([-min((mu_p*Kc_p*alpha, Kt_p*vx)), Kc_p*alpha])
        press += np.linalg.norm(fo)/np.sqrt(alpha)
        f+=fo
    i.pres += press
    return f
    
    
    
    
    

def force_totale(i,detec):
    '''
    @brief Fonction renvoyant la force totale s'appliquant sur le grain i.
    gravite : détermine si le poids est pris en compte dans la force
    paroi : détermine si la force du récipient est prise en compte
    @param un grain i et un booléen 
'''
    force = np.array([0.,0.],dtype='float64') #initialisationde la force
    if detec == True: #Création de la liste des voisins et des parois 
        i.detecter_voisins()
    i.detecter_paroi()
    i.pres = 0.
    
    for j in i.voisins: #On additionne toutes les forces des voisins
        fg = force_grain(j,i) 
        force += fg
        
    if paroi == True : #Puis les parois si elles existent
        fp = force_paroi(i)  
        force += fp 
        
    if gravite == True : #Et la gravité 
        fgrav = np.array([0.,-i.rayon**2],dtype='float64')
        force += fgrav
        
    return force
    
  

def mise_a_jour(detec=True):
    '''
    @brief: Fonction de mise à jour de la liste des positions/vitesses des grains par l'algorithme de Verlet à deux pas.
    1 : calculer les nouvelles vitesses/positions de tous les grains
    2 : les "nouvelles positions/vitesses" deviennent les pos/vit actuelles
    '''
    for i in grains: #schéma de Verlet
        f = force_totale(i,detec)
        i.force = f 
        i.pos_n = 2 * i.pos - i.pos_a + 1/i.rayon**2 * f * dt**2
        i.vit_n = (i.pos_n - i.pos_a)/(2*dt)
    for i in grains: #actualisation 
        i.pos, i.pos_a = i.pos_n, i.pos
        i.vit = i.vit_n
                



