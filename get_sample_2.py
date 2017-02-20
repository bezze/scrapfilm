#!/usr/bin/env python3

import sys, os, numpy as np
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
from glob import glob

nbin = int(sys.argv[1])

if "--frames" in sys.argv:
    frames_arg = sys.argv.index('--frames')
    if sys.argv[frames_arg+1]=='count':
        filename_fi = 'fi.row_0_cm.npy'
        print('Inspecting '+ filename_fi +' ...')
        print('(oscillators, timesteps)')
        print(np.load(filename_fi).shape)
        raise SystemExit
    try:
        start, end, jump = [ int(x) for x in sys.argv[frames_arg+1].split(':') ]
        if not os.path.isdir('sample'):
            os.mkdir('sample')
    except IndexError and ValueError:
        print("--frames needs integers of the form start:end:jump ")
        raise SystemExit

hd = { name : [] for name in glob('hist*')}

for entry in hd:
    data = np.load(entry)
    H, xedges, yedges = np.histogram2d(data[0,:], data[1,:], bins=nbin)
    X, Y = np.meshgrid(xedges, yedges)
    mode = entry.split('.')[0].split('_')[1]
    fi = np.load("fi.row_3_"+mode+".npy")
    rad = np.load("rad.row_3_"+mode+".npy")

    hd[entry] = {j : i for i,j in zip([data, H, xedges, yedges, X, Y, fi, rad],
                                            ['data', 'H', 'xedges', 'yedges',
                                             'X', 'Y', 'fi', 'rad'])}


#plt.plot(hd['hist_ee.npy']['rad'][0,:]/hd['hist_cm.npy']['rad'][0,:])
#plt.plot(hd['hist_ee.npy']['fi'][0,:]-hd['hist_cm.npy']['fi'][0,:])
##plt.plot(hd['hist_ee.npy']['rad'][0,:])
#plt.show()


for t in range(start,end,jump):
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    for col,e in enumerate(hd):
        X = hd[e]['X']; Y = hd[e]['Y']; H = hd[e]['H']
        sm = plt.cm.ScalarMappable(cmap='jet',
                                   norm=plt.Normalize(vmin=np.min(H),
                                                      vmax=np.max(H)))
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        ax[col].pcolormesh(X, Y, H, cmap='jet')
        ax[col].set_title(e.split('.')[0].split('_')[1])

    for col,e in enumerate(hd):
        rad = hd[e]['rad']; fi = hd[e]['fi']
        x=rad[10,t]*np.cos(fi[10,t]); v=rad[10,t]*np.sin(fi[10,t])
        ax_sub = ax[col].twinx() 
        ax_sub.plot(x,v,'ro',ms=8)
        ax_sub.set_axis_off()
        Xlimits = [np.min(hd[e]['xedges']),np.max(hd[e]['xedges'])]
        Ylimits = [np.min(hd[e]['yedges']),np.max(hd[e]['yedges'])]
        ax[col].set_xlim( Xlimits ); ax[col].set_ylim( Ylimits )
        ax_sub.set_xlim( Xlimits ); ax_sub.set_ylim( Ylimits )
    fig.subplots_adjust(right=0.8)
    axcb = fig.add_axes([0.18,0.15,.8,.7])
    plt.colorbar(sm, ax=axcb)
    axcb.set_axis_off()
    fig.suptitle('frame '+str(t), fontsize=14, fontweight='bold',
                 color='red')

    plt.savefig("sample/sample_"+str(t)+".png", format="png")
    plt.close(fig)
    

plt.show()

