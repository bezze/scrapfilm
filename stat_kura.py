#!/usr/bin/env python3

#import init_var as iv
import numpy as np
import sys 
from scrapfilm import boundary
import os
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

mode = sys.argv[1]
path = './stat_kura/'
if not os.path.isdir(path):
    os.mkdir(path)

if "--frames" in sys.argv:
    frames_arg = sys.argv.index('--frames')
    try:
        start, end, jump = [ int(x) for x in sys.argv[frames_arg+1].split(':') ]
        if not os.path.isdir(path+'movie'):
            os.mkdir(path+'movie')
    except IndexError and ValueError:
        print("--frames needs integers of the form start:end:jump ")


for i in range(6): # 0 -> bottom, 60 -> top
    filename_k = 'kura.row_'+str(i)+'_'+mode+'.npy'
    filename_fi = 'fi.row_'+str(i)+'_'+mode+'.npy'
    filename_dfi = 'dfi.row_'+str(i)+'_'+mode+'.npy'
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

red = np.linspace(255,0,all_dfi.shape[1])/255
green = np.linspace(0,255,all_dfi.shape[1])/255
blue = np.linspace(0,0,all_dfi.shape[1])/255
#s1=np.linspace(0,1,10)
#s2=np.linspace(1,1,10)
#vfloor=np.vectorize(floor)
vboundary=np.vectorize(boundary)

for row in range(6):
    dms = np.mean( np.abs( vboundary(all_dfi[row,:,:], 2*np.pi) ),axis=0)**2
    dsm = np.mean( vboundary( all_dfi[row,:,:], 2*np.pi )**2,axis=0)
    sqrt_dsm = (dsm**.5)/np.pi
    
    #for chain in range(all_dfi.shape[1]):
    #    plt.plot(boundary(all_dfi[0,chain,:],2*np.pi)/np.pi,color=(red[chain],green[chain],blue[chain]))
        #plt.plot(all_dfi[0,chain,:],'b')#,color=(red[chain],green[chain],blue[chain]))
        #plt.plot(s1,chain*s2,linewidth=10,color=(red[chain],green[chain],blue[chain]))
    #plt.plot(dms , 'b', linewidth=3)
    #plt.plot(dsm , 'g', linewidth=3)
    #plt.plot(dsm - dms , 'k', linewidth=3)
    plt.plot(sqrt_dsm , linewidth=2)

plt.savefig(path+'root_mean_square.png',format='png')
#plt.show()

if "--frames" in sys.argv:
    fase=0
    for time in range(start,end,jump):#range(1):# 
        fig, ax = plt.subplots(2,3,subplot_kw=dict(projection='polar'))
        for row in range(3):
            for chain in range(all_fi.shape[1]):
                #ax.plot( np.cos(s), np.sin(s),'b')
                for ud in range(2):
                    place=row+3*ud
                    ax[ud,row].set_aspect('equal')
                    ax[ud,row].plot(all_fi[place,chain,time]+fase,1,
                             marker='^', color='r', markersize=10 )
    
        fig.suptitle('frame '+str(time), fontsize=14, fontweight='bold',
                     color='red')
        fig.tight_layout()
        fig.savefig(path+'movie/frame_'+str(time)+'.png',format='png')
        plt.close(fig)
    
    for row in range(order.shape[0]):
        plt.plot(order[row,:], label='row '+str(row))
    plt.legend()
    fig.savefig(path+'order_'+'.png',format='png')
    #plt.show()

mean_rows = np.mean( order, axis=1)
std_rows = np.std( order, axis=1)
mean_kura = np.mean(mean_rows)
with open(path+'kura_results', 'w') as f:
    for l in range(mean_rows.size):    
        f.write(str(l)+':  '+str(mean_rows[l])+' '+str( std_rows[l])+'\n' )
    f.write('all: '+str(mean_kura)+' '+str(np.std(mean_rows))+'\n' )
