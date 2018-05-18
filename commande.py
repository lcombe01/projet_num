#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:37:38 2018

@author: sdiop
"""


nom_simul = 'test_colonne_ecoulement.py'

main = open('main.py','r')
animation = open('animation.py','r')
traitement= open('traitement_fichiers.py','r')
simul = open(nom_simul,'r')
exe = open('execution.py','w')

for line in main:
    exe.write(line)

main.close()
    
for line in animation:
    exe.write(line)
    
animation.close()
    
for line in traitement:
    exe.write(line)
    
traitement.close()

for line in simul:
    exe.write(line)

exe.close()

exec(open("execution.py").read())