#!/usr/bin/env python3

import numpy as np
import glob as gl
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt

def ed2cen(ed):
    cen = .5*(ed[1:]+ed[:-1])
    return cen
filas=gl.glob('fi.row*cm*')
print(filas)
#filas = 'fi.row_0_cm.npy'
#fi=np.load(filas)
#w = np.diff(fi[0,:]); up=w<2; down=w>-2
#print(w[up&down].size)
#h, ed = np.histogram( w[up&down], bins=10, density=True )
#cen=.5*(ed[1:]+ed[:-1])
#plt.plot(cen , h, '*' )
##plt.plot(w[up&down], '*')
#plt.show()
binsize=200
H=np.zeros([6,binsize])
C=np.zeros([6,binsize])
maximos=[]
segundos=[]
for row in range(len(filas)):
    fi=np.load(filas[row])
    for chain in range(fi.shape[0]):
        w = np.diff(fi[chain,:]); up=w<4; down=w>-4
        #plt.plot(w[up&down])
        w_filt = w[up & down ]
        h, ed = np.histogram( w_filt, bins=binsize, normed=True ); cen=ed2cen(ed)
        i_max=np.argmax(h)
        i_sec=np.argmax( np.delete(h,i_max) )

        maximos.append([h[i_max], cen[i_max]])
        segundos.append([h[i_sec], cen[i_sec]])

        #print(h, ed)
        #plt.plot( fi[chain,:]  )
        #plt.plot( .5*(ed[1:]+ed[:-1]), h )
        H[row,:]+=h
        C[row,:]+=.5*(ed[1:]+ed[:-1])
    #plt.show()
    H[row,:]=H[row,:]/6
    C[row,:]=C[row,:]/6
#for row in range(len(filas)):
#    plt.plot(C[row,:],H[row,:], label=str(row))
#    plt.legend()
#plt.show()
maximos=np.asarray(maximos)
segundos=np.asarray(segundos)
#plt.plot(maximos[:,1],maximos[:,0],'*')
#plt.plot(segundos[:,1],segundos[:,0],'*')
#plt.plot( .5*(maximos[:,1]+segundos[:,1]) )
print(np.mean(.5*(maximos[:,1]+segundos[:,1])), np.std(.5*(maximos[:,1]+segundos[:,1])  ) )
#h, ed = np.histogram(.5*(maximos[:,1]+segundos[:,1]), bins=5) ; cen=ed2cen(ed)
h, ed = np.histogram(maximos[:,1], bins=5) ; cen=ed2cen(ed)
plt.plot(cen,h,'*')
plt.show()
