#!/usr/bin/env python3

"""
Creates two png files showing the chain distribution of the top and bottom
brushes. Useful for mapping indexes.
"""


import matplotlib as mpl
import numpy as np
mpl.use('Agg')
from matplotlib import pyplot as plt

r_all = np.load('r_all.npy') # ALL POS (time, nchains, nbeads, xyz)
all_0 = r_all[:,:,0,:]  # 0th bead i.e. root
all_9 = r_all[:,:,9,:]  # 9th bead
def plot_chain_dist(cini,cend):
    for c in range(cini,cend):
        """ selecting c-th chain, 0 time """
        rg=all_0[0,c,:]
        plt.plot(rg[0],rg[1],'o')
        plt.annotate(str(c), xy=(rg[0],rg[1]), xytext=(rg[0]*1.01,rg[1]*1.01) )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./plot_chain_dist_c'+str(cini)+'_'+str(cend-1)+'.png')
    plt.cla()

#""" Bottom  """
plot_chain_dist(0,60)
#""" Top """
plot_chain_dist(60,120)
