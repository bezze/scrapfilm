#!/usr/bin/env python3

import numpy as np
#import init_var as iv
import sys 
import os

mode = sys.argv[1]
path = sys.argv[2]

if not os.path.isdir('./movie'):
    os.mkdir('./movie')

for i in range(6): # 0 -> bottom, 60 -> top
    filename_k = path+'kura.row_'+str(i)+'_'+mode+'.npy'
    filename_fi = path+'fi.row_'+str(i)+'_'+mode+'.npy'
    filename_dfi = path+'dfi.row_'+str(i)+'_'+mode+'.npy'
    if i==0:
        dummy_k = np.load(filename_k) #all_kura[i,:]
        dummy_fi = np.load(filename_fi) #all_kura[i,:]
        dummy_dfi = np.load(filename_dfi) #all_kura[i,:]
        all_kura = np.empty([6,dummy_k.size], dtype=complex)
        matsize = [x for x in dummy_fi.shape]; fisize=[6]
        fisize.extend(matsize)
        all_fi = np.empty(fisize)
        all_dfi = np.empty(fisize)
        #print(all_kura.shape)
        #print(dummy[0:10])

    all_kura[i,:] = np.load(filename_k) 
    all_fi[i,:,:] = np.load(filename_fi) 
    all_dfi[i,:,:] = np.load(filename_dfi) 

order = np.absolute(all_kura)
psi = np.angle(all_kura)

import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
for chain in range(all_dfi.shape[1]):
    plt.plot(all_dfi[0,chain,:],color=)
plt.show()

#import matplotlib as mpl
#mpl.use('Agg')
#from matplotlib import pyplot as plt
#fase=0
#for time in range(all_fi.shape[2]):#range(1):# 
#    fig, ax = plt.subplots(2,3,subplot_kw=dict(projection='polar'))
#    for row in range(3):
#        for chain in range(all_fi.shape[1]):
#            #ax.plot( np.cos(s), np.sin(s),'b')
#            for ud in range(2):
#                place=row+3*ud
#                ax[ud,row].set_aspect('equal')
#                ax[ud,row].plot(all_fi[place,chain,time]+fase,1,
#                         marker='^', color='r', markersize=10 )
#
#    fig.suptitle('frame '+str(time), fontsize=14, fontweight='bold',
#                 color='red')
#    fig.tight_layout()
#    fig.savefig('movie/frame_'+str(time)+'.png',format='png')
#    plt.close(fig)

#for row in range(order.shape[0]):
#    plt.plot(order[row,:], label='row '+str(row))
#plt.legend()
#plt.show()

mean_rows = np.mean( order, axis=1)
std_rows = np.std( order, axis=1)
mean_kura = np.mean(mean_rows)
with open('./kura_results', 'w') as f:
    for l in range(mean_rows.size):    
        f.write(str(l)+':  '+str(mean_rows[l])+' '+str( std_rows[l])+'\n' )
    f.write('all: '+str(mean_kura)+' '+str(np.std(mean_rows))+'\n' )
