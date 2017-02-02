#!/usr/bin/env python3

import init_var as iv
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

def plot_chain_dist(cini,cend):
    for c in range(cini,cend):
        """ selecting c-th chain, 0 time """
        rg=iv.all_root[:,0,c]
        plt.plot(rg[0],rg[1],'o')
        plt.annotate(str(c), xy=(rg[0],rg[1]), xytext=(rg[0]*1.01,rg[1]*1.01) )
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./plot_chain_dist_c'+str(cini)+'_'+str(cend-1)+'.png')
    plt.cla()

#""" Bottom  """
#plot_chain_dist(0,60)
#""" Top """
#plot_chain_dist(60,120)
