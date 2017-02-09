#!/usr/bin/env python3

import sys, os, numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

nbin = int(sys.argv[1])
path = sys.argv[2]
name = path.split('.')[0]

all_data = np.load(path)
x_data = all_data[0,:]
v_data = all_data[1,:]

H, xedges, yedges = np.histogram2d(x_data, v_data, bins=nbin)
X, Y = np.meshgrid(xedges, yedges)
plt.pcolormesh(X, Y, H)
plt.xlim( [np.min(x_data),np.max(x_data)] )
plt.ylim( [np.min(v_data),np.max(v_data)] )
plt.colorbar()
#ax.set_aspect('equal')
plt.title(name+'.bins_'+str(nbin))
plt.savefig(name+'.bins_'+str(nbin)+'.svg', format='svg')
plt.savefig(name+'.bins_'+str(nbin)+'.png', format='png')
plt.clf()
