#!/usr/bin/env python3

import init_var as iv
import numpy as np
from numpy.linalg import norm
import sys 
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from scrapfilm import analyze, analyze_v, boundary

def plot_hist(cini, cend, plot=False, save=False, fullpath='None'):

    for c in range(cini,cend):
        """ selecting c-th chain """
        rg=iv.all_root[:,:,c]
        r0=iv.all_9[:,:,c]
        v=iv.v_all_9[:,:,c]
    
        """ Centering """
        rel=r0-rg
        """ Boundary conditions """
        r_real_x= boundary(rel[0,:],iv.Lx)
        rb=np.copy(rel);rb[0,:]=r_real_x
       # if plot:
       #     plt.plot(rb[0,:],v[0,:],linestyle='None',markersize=1,marker='o')
    
        """ Stacking all the data into horizontal vectors """
        if c==cini:
            x_data = rb[0,:]
            v_data = v[0,:]
        else:
            x_data = np.hstack( (x_data,rb[0,:]) )
            v_data = np.hstack( (v_data,v[0,:]) )

    #if plot:
    #    plt.savefig('./plot.x_v_c'+str(cini)+'_c'+str(cend-1)+'.svg', format='svg')
    #    plt.clf()

    all_data = np.vstack(( x_data, v_data ))

    if plot:
        H, xedges, yedges = np.histogram2d(x_data, v_data, bins=100)
        plt.title('')
        X, Y = np.meshgrid(xedges, yedges)
        plt.pcolormesh(X, Y, H)
        plt.xlim( [np.min(x_data),np.max(x_data)] )
        plt.ylim( [np.min(v_data),np.max(v_data)] )
        plt.colorbar()
        #ax.set_aspect('equal')
        
        plt.title('plot.x_v_hist.c'+str(cini)+'_c'+str(cend-1))
        plt.savefig('./plot.x_v_hist.c'+str(cini)+'_c'+str(cend-1)+'.svg', format='svg')
        plt.clf()
    if save:
        #print(all_data.shape)
        np.save(fullpath,all_data)

#plot_hist(0,60)
#plot_hist(60,120)
#plot_hist(0,chains, bounds, save=True)

