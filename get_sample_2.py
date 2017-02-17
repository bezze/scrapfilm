#!/usr/bin/env python3

import sys, os, numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy.misc import imread
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

hist_files = glob('hist*')
data_list = []
for f in hist_files: data_list.append( np.load(f) )

for loaded_file, name_file in zip(data_list,hist_files):
    x_data = loaded_file[0,:]
    v_data = loaded_file[1,:]
    mode = name_file.split('.')[0].split('_')[1]

    H, xedges, yedges = np.histogram2d(x_data, v_data, bins=nbin)
    X, Y = np.meshgrid(xedges, yedges)

    samp_fi = np.load("fi.row_0_"+mode+".npy")
    samp_rad = np.load("rad.row_0_"+mode+".npy")

    for t in range(start,end,jump):
        fig_hist, ax_hist = plt.subplots(1,1)
        ax_hist.pcolormesh(X, Y, H)
        x=samp_rad[10,t]*np.cos(samp_fi[10,t]);
        v=samp_rad[10,t]*np.sin(samp_fi[10,t])
        ax = ax_hist.twinx() 
        ax.plot(x,v,'ro',ms=8)
        ax_hist.set_xlim( [np.min(x_data),np.max(x_data)] )
        ax_hist.set_ylim( [np.min(v_data),np.max(v_data)] )
        ax.set_xlim( [np.min(x_data),np.max(x_data)] )
        ax.set_ylim( [np.min(v_data),np.max(v_data)] )
        fig_hist.suptitle(mode+' frame '+str(t), fontsize=14, fontweight='bold',
                          color='red')
        plt.savefig("sample/sample_"+mode+"_"+str(t)+".png", format="png")
        plt.close(fig_hist)

