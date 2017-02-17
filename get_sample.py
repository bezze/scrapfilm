#!/usr/bin/env python3

import sys, os, numpy as np
import matplotlib as mpl
mpl.use('Agg')
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

if sys.argv[3] == "sample":
    if not os.path.isdir("sample"):
        os.mkdir("sample")
    samp_fiee = np.load("fi.row_0_ee.npy")
    samp_radee = np.load("rad.row_0_ee.npy")
    samp_ficm = np.load("fi.row_0_cm.npy")
    samp_radcm = np.load("rad.row_0_cm.npy")

    for t in range(19990,20000,1):
        #ax_back = fig_hist.add_axes([0,0,1,1])
        fig_hist, ax_hist = plt.subplots(1,1)
        ax_hist.pcolormesh(X, Y, H)
        ax_hist.set_xlim( [np.min(x_data),np.max(x_data)] )
        ax_hist.set_ylim( [np.min(v_data),np.max(v_data)] )

        x=samp_radcm[10,t]*np.cos(samp_ficm[10,t]);
        v=samp_radcm[10,t]*np.sin(samp_ficm[10,t])
        #ax_cm = fig_hist.add_axes(ax_hist.get_position(),  frameon=False) #[0,0,1,1]
        ax_cm = fig_hist.add_axes([0.,0.,1.,1.],  frameon=False) #[0,0,1,1]
        ax_cm.plot(x,v,'ro',ms=8)
        print(x,v)
        #x=samp_radee[0,t]*np.cos(samp_fiee[0,t]); v=samp_radee[0,t]*np.sin(samp_fiee[0,t])
        #ax_ee = fig_hist.add_axes([0,0,1,1],  frameon=False)
        #ax_ee.plot(x,v,'g^')

        fig_hist.suptitle('frame '+str(t), fontsize=14, fontweight='bold',
                     color='red')
        ax_cm.axis('off')
        #ax_cm; plt.cla()
        #ax_ee; plt.cla()
        #del ax_cm.lines[0]; del ax_ee.lines[0]
        #plt.show()
        plt.savefig("sample/sample_"+str(t)+".png", format="png")
        plt.close(fig_hist)
        #fig_hist.delaxes(ax_ee)
        #fig_hist.delaxes(ax_cm)
        #plt.close(fig_hist)



#ax_hist.pcolormesh(X, Y, H)
#ax_hist.set_xlim( [np.min(x_data),np.max(x_data)] )
#ax_hist.set_ylim( [np.min(v_data),np.max(v_data)] )
#fig_hist.colorbar(map, cax=ax_hist)
#ax.set_aspect('equal')
#plt.title(name+'.bins_'+str(nbin))
#plt.savefig(name+'.bins_'+str(nbin)+'.svg', format='svg')
#plt.savefig(fig_hist,name+'.bins_'+str(nbin)+'.png', format='png')
#plt.clf()
