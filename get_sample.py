#!/usr/bin/env python3

import sys, os, numpy as np
import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.cbook as cbook
from matplotlib import pyplot as plt
from scipy.misc import imread

nbin = int(sys.argv[1])
path = sys.argv[2]
name = path.split('.')[0]

all_data = np.load(path)
x_data = all_data[0,:]
v_data = all_data[1,:]
 
H, xedges, yedges = np.histogram2d(x_data, v_data, bins=nbin)
X, Y = np.meshgrid(xedges, yedges)
fig_hist, ax_hist = plt.subplots(1,1)
ax_hist.pcolormesh(X, Y, H)
ax_hist.set_xlim( [np.min(x_data),np.max(x_data)] )
ax_hist.set_ylim( [np.min(v_data),np.max(v_data)] )
#fig_hist.colorbar(map, cax=ax_hist)
#ax.set_aspect('equal')
plt.title(name+'.bins_'+str(nbin))
#plt.savefig(name+'.bins_'+str(nbin)+'.svg', format='svg')
#plt.savefig(fig_hist,name+'.bins_'+str(nbin)+'.png', format='png')
#plt.clf()

if sys.argv[3] == "sample":
    if not os.path.isdir("sample"):
        os.mkdir("sample")
    samp_fiee = np.load("fi.row_0_ee.npy")
    samp_radee = np.load("rad.row_0_ee.npy")
    samp_ficm = np.load("fi.row_0_cm.npy")
    samp_radcm = np.load("rad.row_0_cm.npy")

    for t in range(19900,20000,1):

        x=samp_radee[0,t]*np.cos(samp_fiee[0,t]); v=samp_radee[0,t]*np.sin(samp_fiee[0,t])
        ax_ee = fig_hist.add_axes([0,0,1,1],  frameon=False)
        ax_ee.plot(x,v,'g^')
        ax_ee.set_aspect('equal')
        x=samp_radcm[0,t]*np.cos(samp_ficm[0,t]); v=samp_radcm[0,t]*np.sin(samp_ficm[0,t])
        ax_cm = fig_hist.add_axes([0,0,1,1],  frameon=False)
        ax_cm.plot(x,v,'bo')
        ax_cm.set_aspect('equal')

        fig_hist.suptitle('frame '+str(t), fontsize=14, fontweight='bold',
                     color='red')

        #del ax_cm; del ax_ee
        plt.show()
        #fig_hist.delaxes(ax_ee)
        #fig_hist.delaxes(ax_cm)
        #plt.savefig(fig_hist,"sample/sample_"+str(t)+".png", format="png")
        #plt.close(fig_hist)



