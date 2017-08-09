#!/usr/bin/python3

import numpy as np
import sys, os, fileinput
import time as t
import pandas as p


t1=t.clock()

archivo=sys.argv[1]
#print(t.clock()-t1)

#datos=p.read_table(archivo)
#-------------------------
f=open(archivo,'r')
datos=[]
for line in f:
    datos.append(line.split())

#-------------------------
#datos=np.loadtxt(archivo)
#-------------------------

size=len(datos)
pot=0
i=1
while size > pot:
    pot=2**i
    i+=1

datos_ar=np.asarray(datos, dtype=np.float32)
pad=np.zeros((pot-size,datos_ar.shape[1]-1))

energias_pad=np.vstack((datos_ar[:,1:],pad)) #datos2=np.append(np.asarray(datos),pad)
t_fft=t.clock()

freq = np.fft.rfftfreq(energias_pad.shape[0],d=1) #+str(freq[i])+' '
for col in range(energias_pad.shape[1]):
    fft = np.fft.rfft(energias_pad[:,col], pot)

    lista = [ x for x in range(10)]
    
    if not os.path.isfile("transformadas.dat"):
        with open("transformadas.dat", "w+") as w:
            for number in fft:
                w.write(str(number)+'\n')
    else:
        with fileinput.FileInput("transformadas.dat",inplace=1) as f:
            for line,elemento in zip(f,fft):
                line = line.strip('\n') +' '+ str(elemento)
                print(line)

#print(t.clock()-t_fft)

#-------------------------

with open("transformadas.dat", "w") as g:
    for i in range(len(freq)):
        g.write(freq[i], fft[i,:])

#for element in fft[20:]:
#    print(element**2)

