#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:45:34 2018
@author: sdiop
"""

import numpy as np
import random as rd


class grain :
    '''
    Definition des méthodes et attributs des grains
    '''
    
    def __init__(self, x0,y0,vx0,vy0, ray=0):
        
        if ray==0:
            self.rayon = np.random.normal(1,0.15)
        else:
            self.rayon = ray
        
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
        Renvoie la liste des grains j en contact avec le grain self
        On teste tous les grains (différents de i) pour voir si ils sont en contact
        '''
        self.voisins=[]
        for k in grains:
            if ((k.pos[0]-self.pos[0])**2+(k.pos[1]-self.pos[1])**2 < (k.rayon + self.rayon)**2) and k != self:
                self.voisins.append(k)
                
                
    def detecter_paroi(self):
        self.bords = []
        R = self.rayon
        if paroi == False : self.bords = [False, False, False]
        else :
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
    Fonction de calcul de la force due au grain i s'appliquant sur le grain j
    '''
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
    
    if Rj+Ri - r < 0: return np.array([0.,0.])
    
    u_r = 1/r * np.array([x, y])
    u_t = 1/r * np.array([-y, x])
    
    v_t = np.dot(vij, u_t)
    v_r = np.dot(vij, u_r)
    
    if np.dot(vij,u_r) < 0 :
        K = Kc
    else :
        K = Kd
    F_r = K*(Rj+Ri - r)
    
    if np.abs(Kt * v_t) < np.abs(mu * F_r) : 
        F_t = - Kt * v_t
    else :
        F_t = - mu * np.abs(F_r) * np.sign(v_t)
    
    j.pres += abs(F_r)/np.sqrt((Rj+Ri - r)*j.rayon)
    
    return F_r * u_r + F_t * u_t
       




def force_paroi(i):
    '''
    Fonction de calcul de la force s'appliquant sur le grain i à cause de la paroi
    '''
    R = i.rayon
    x,y = i.pos
    vx,vy = i.vit
    f = np.array([0.,0.])
    press = 0.
    
    if i.bords[0] == True :
        alpha = -(x-R+L/2)
        if vx >= 0 :
            fo = np.array([Kd_p*alpha,-min((mu_p*Kc_p*alpha, Kt_p*vy))])
        if vx < 0 :
            fo = np.array([Kc_p*alpha,-min((mu_p*Kd_p*alpha, Kt_p*vy))])
        press += np.linalg.norm(fo)/np.sqrt(alpha)
        f+=fo
    if i.bords[2] == True :
        alpha = x+R-L/2
        if vx > 0 :
            fo = np.array([-Kc_p*alpha,-min((mu_p*Kc_p*alpha, Kt_p*vy))])
        if vx <= 0 :
            fo = np.array([-Kd_p*alpha,-min((mu_p*Kd_p*alpha, Kt_p*vy))])
        press += np.linalg.norm(fo)/np.sqrt(alpha)
        f+=fo
    if i.bords[1] == True :
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
    Fonction renvoyant la force totale s'appliquant sur le grain i.
    gravite : détermine si le poids est pris en compte dans la force
    paroi : détermine si la force du récipient est prise en compte
    '''
    force = np.array([0.,0.],dtype='float64')
    if detec == True:
        i.detecter_voisins()
    i.detecter_paroi()
    i.pres = 0.
    
    for j in i.voisins:
        fg = force_grain(j,i) 
        force += fg
        
    if paroi == True :
        fp = force_paroi(i)  
        force += fp 
        
    if gravite == True :
        fgrav = np.array([0.,-i.rayon**2],dtype='float64')
        force += fgrav
        
    return force
    
  

def mise_a_jour(detec=True):
    '''
    Fonction de mise à jour de la liste des positions/vitesses des grains par l'algorithme de Verlet à deux pas.
    1 : calculer les nouvelles vitesses/positions de tous les grains
    2 : les "nouvelles positions/vitesses" deviennent les pos/vit actuelles
    '''
    for i in grains:
        f = force_totale(i,detec)
        i.force = f 
        i.pos_n = 2 * i.pos - i.pos_a + 1/i.rayon**2 * f * dt**2
        i.vit_n = (i.pos_n - i.pos_a)/(2*dt)
    for i in grains:
        i.pos, i.pos_a = i.pos_n, i.pos
        i.vit = i.vit_n
                



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
    '''@brief Creates gif frome a stock of files 
    @param filenames: a list of strings '/path/filename' '''
    images = []
    N = len(filenames)
    for i,filename in enumerate(filenames):
        images.append(imageio.imread(filename))
        print('{:.1f}% of gif creation'.format((i/N)*100))
    imageio.mimsave(folder+'/movie.gif', images, duration = 1/framerate)



def traj2gif(T,X,Y,R, ratio = 100, gif = True):
    '''
    @brief 
        creates a .gif animation of the simulated particles from input trajectories, such as the ones written by 'trajectoires'
    @params 
        T : 1D time array ;
        X : 2D x-coordinate array (each line = x of a specific particle) ;
        Y : 2D y-coordinate array (each line = y of a specific particle) ;
        size : size of the fig lim: x and y lim of the axis ;
        lim : x and y lim of the axis ;
        center : y of the center of the figure ;
        ratio: one image is saved every'ratio' ; 
        gif: if True, creates gif from the recorded pictures
    '''
    
    plt.ioff()  #turns off interactive mode that would slow down the simulation
    fig = plt.figure(1,figsize=(6,7))
    ax = plt.axes(aspect='equal',xlim=(-L/2-5,L/2+5), ylim=(-5,H+5))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$',rotation=0)
    filenames = []
    t_final = T[-1]
    N = np.shape(X)[0] #number of particles
    
    for counter in range(np.shape(T)[0]):
        t = T[counter]
        if counter == 0 : T0 = time.time() #measures start time
        
            
        if counter % ratio == 0: #saving image depends on ratio
            
            for artist in plt.gca().lines + plt.gca().collections: #clears figure at every incrementation
                artist.remove()
            patches = []
            for i in range(N):
                x = X[i][counter]
                y = Y[i][counter]
                ray = R[i]
                circle = pat.Circle((x, y), ray)
                patches.append(circle)
            p = PatchCollection(patches, alpha=0.5, color = 'r',linewidth =0.01)
            ax.add_collection(p) #actualision of the figure
            plt.title('t = {:1f}'.format(t))
                
            
            if paroi == True: #draws the recipient
                plt.plot([-L/2,-L/2], [0,H],'k')
                plt.plot([L/2,L/2], [0,H],'k')
                plt.plot([-L/2,-e], [0,0],'k')
                plt.plot([e, L/2], [0,0],'k')
            
            fig.savefig(folder+'fig{}.png'.format(counter))    
            filenames.append(folder+'fig{}.png'.format(counter))
            print('{:.2f} % of image creation'.format((t/t_final)* 100))
            
            if t>0 : # gives temporal information about the progression of the task
                T1 = time.time() #measures current time
                DeltaT = T1-T0
                Tf = t_final * DeltaT/t
                T_rest = Tf - DeltaT # calculates the time left
                T_rest = time.gmtime(T_rest)
                print("Left: %i h %i min %i s" % ( T_rest.tm_hour, T_rest.tm_min, T_rest.tm_sec))
            
    if gif == True : #call gif from files to create the final gif
        print('Creating gif')
        gif_from_files(filenames, ratio) 
            
    
    
    
    
def afficher(grains, disp = 'None'):

    fig = plt.figure(1,figsize=(5,7))
    ax = plt.axes(aspect='equal',xlim=(-L/2-5,L/2+5), ylim=(-5,H+5))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$',rotation=0)
    
    patches = []
    colors = []

    if disp == 'p':
        mise_a_jour()
        list_press = [i.pres for i in grains]
        max_c = max(list_press)
        min_c = min(list_press)
        lab = r'$p$'
        cmap=cm.magma_r
        alpha = 0.7
    if disp == 'vy':
        mise_a_jour()
        list_vit = []
        for i in grains:
            if i.pos[1]>-7: list_vit.append(np.log(np.sqrt(i.vit[1]**2)))
        max_c = max(list_vit)
        min_c = min(list_vit)
        lab = r'$-v_y$'
        cmap=cm.viridis
        alpha = 0.7
    if disp == 'None':
        alpha = 0.5 
        
    for i in grains :
        if disp == 'p':
            col = i.pres+400
        if disp == 'vy' :
            col = np.log(np.sqrt(i.vit[1]**2))
            if i.pos[1]<-7: col=0
        if disp == 'None':
            col = 'r'
        circle = pat.Circle(i.pos, i.rayon)
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
    Renvoie les trajectoires [T,X,Y,VX,VY] qui sont tous des arrays d'un ensemble de grains initial 'grains', entre les instants 0 et t_final
    '''
    t = t0
    X = [[] for k in range(len(grains))]
    Y = [[] for k in range(len(grains))]
    VX = [[] for k in range(len(grains))]
    VY = [[] for k in range(len(grains))]
    rayons = [grains[i].rayon for i in range(len(grains))]
    
    temps = []
    T0 = time.time()
    counter = 0          
    
    while t-t0 < t_final:
        for i in range(len(grains)) :
            X[i].append(grains[i].pos[0])
            Y[i].append(grains[i].pos[1])
            VX[i].append(grains[i].vit[0])
            VY[i].append(grains[i].vit[1])
        temps.append(t)
        if counter%3 == 0: detec = True
        else: detec = False
        mise_a_jour(detec)
        t += dt
        if counter % 30 == 0:
            print('{:.2f} %'.format(100 * (t-t0)/t_final))
            if t-t0>0 : # gives temporal information about the progression of the task
                T1 = time.time() #measures current time
                DeltaT = T1-T0
                Tf = t_final * DeltaT/(t-t0)
                T_rest = Tf - DeltaT # calculates the time left
                T_rest = time.gmtime(T_rest)
                print("Left: %i h %i min %i s" % ( T_rest.tm_hour, T_rest.tm_min, T_rest.tm_sec))
        counter += 1
        
    temps = np.array(temps)
    for k in range(len(X)): X[k] = np.array(X[k])
    for k in range(len(Y)): Y[k] = np.array(Y[k])
    for k in range(len(VX)): VX[k] = np.array(VX[k])
    for k in range(len(VY)): VY[k] = np.array(VY[k])
    rayons = np.array(rayons)
    
    if save == True :
        np.savez_compressed(filename,temps,X,Y,rayons) #saves .npz file with time array, X array and Y array (velocities can be retrospectively computed with T,X,Y)
        params = np.array([Kc, eta, Kt, mu, Kc_p, eta_p, Kt_p, mu_p, L, e , dt]) #saves a file with relevant simulation parameters
        np.savetxt(filename+'_params', params)
    return(temps, X, Y, rayons)



def load_trajectories(filename):
    '''
    @brief
        Loads T, X and Y arrays from a saved .npz file
    @params
        filename : path to find the .npz file containing the trajectories
    '''
    T = np.load(filename)['arr_0']
    X = np.load(filename)['arr_1']
    Y = np.load(filename)['arr_2']
    R = np.load(filename)['arr_3']
    return(T,X,Y,R)
    
    
def load_state(T, X ,Y, rayons, counter = -1):
    '''
    @briefs,
        Creates a 'grains' list at a specified moment of the input trajectories
    @params
        T : 1D time array ;
        X : 2D x-coordinate array (each line = x of a specific particle) ;
        Y : 2D y-coordinate array (each line = y of a specific particle) ;
        counter : number of the instant to capture (par défaut, -1 : ca correspond au dernier instant)
    '''
    VX, VY = velocities(T,X,Y)
    X_0 = X.transpose()[counter]
    Y_0 = Y.transpose()[counter]
    VX_0 = VX.transpose()[counter]
    VY_0 = VY.transpose()[counter]
    grains = []
    for k in range(np.shape(X_0)[0]):
        grains.append(grain(X_0[k], Y_0[k], VX_0[k], VY_0[k] , rayons[k] ))
    return(grains)
    
    
def velocities(T,X,Y):
    '''
    Computes velocities from trajectories
    '''
    dt = T[1]-T[0]
    VX = np.diff(X)/dt
    VY = np.diff(Y)/dt
    return(VX, VY)
    
    
    
def force_wall(grains, wall_id):
    F = 0
    if wall_id == 1:
        for gr in grains:
            R = gr.rayon
            if gr.pos[1] < R : 
                F += Kd_p * (R-gr.pos[1])
    if wall_id == 0:
        for gr in grains:
            R = gr.rayon
            if gr.pos[0] < -L/2+R :
                F +=  - Kd_p * (gr.pos[0]-R+L/2)
    if wall_id == 2:
         for gr in grains:
             R = gr.rayon
             if gr.pos[0] > L/2-R :
                F +=   Kd_p * (gr.pos[0]+R-L/2) 
    return F
    
    
def debit(T,X,Y):
    '''
    @brief: Calculates the flux of particles through the hole in the bottom wall, as a function of time
            Returns a 2D array [[time], [flux]]
    @params:         
        T : 1D time array ;
        X : 2D x-coordinate array (each line = x of a specific particle) ;
        Y : 2D y-coordinate array (each line = y of a specific particle)
    '''
    DT = 1000
    N = np.shape(X)[0]
    Xt, Yt = np.transpose(X), np.transpose(Y)
    liste = []
    for k in range(1,np.shape(T)[0]//DT):
        deb =0
        t = k*DT
        for gr in range(N):
            if Yt[t,gr] < y_min and Yt[t-DT,gr] > y_min : 
                deb +=1
        liste.append([t*dt, deb/(DT*dt)])
    liste = np.array(liste).transpose()
    return(liste)
    

def pressure_field(grains):
    x = []
    y = []
    p = []
    mise_a_jour()
    for i in grains:
        x.append(i.pos[0])
        y.append(i.pos[1])
        p.append(i.pres)
    x = np.array(x)
    y = np.array(y)
    p = np.array(p)
    return(x,y,p)
        
        
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:25:07 2018

@author: sdiop
"""


# exterieur
L=90
H = 80
e= 9
y_min = -1

# paramètres des contacts
Kc = 1e4
eta = 0.8
Kd = eta**2 * Kc

Kc_p = Kc
eta_p = eta
Kd_p = Kd

Kt = 100
mu = 0.7
mu_p = mu
Kt_p = Kt


dt = 1e-3


paroi = True
gravite = True
remove = False
 
   
T,X,Y,R = load_trajectories('ecoulement_1260.npz')

#%%
grains = load_state(T,X,Y,R,-1)
afficher(grains,'p')
plt.savefig('ecoul_1260_3.pdf',format='pdf')