#!/usr/bin/env python3

import sys, os, numpy as np
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt

nbin = int(sys.argv[1])
path = sys.argv[2]
name = path.split('.')[0]

all_data = np.load(path)
x_data = all_data[0,:]
v_data = all_data[1,:]

def ed2cen(ed):
    cen = .5*(ed[1:]+ed[:-1])
    return cen

Hx, xed = np.histogram(x_data, bins=nbin)
Hv, ved = np.histogram(v_data, bins=nbin)

fig, ax = plt.subplots(1,2)
fig.suptitle(name+'.x_and_v.bins_'+str(nbin))
ax[0].plot(ed2cen(xed),Hx, '-xr', label="x");ax[0].legend()
ax[1].plot(ed2cen(ved),Hv, '-sk', label="v");ax[1].legend()
plt.savefig(name+'.x_and_v.bins_'+str(nbin)+'.png', format='png')
plt.clf()

H, xedges, yedges = np.histogram2d(x_data, v_data, bins=nbin)
X, Y = np.meshgrid(xedges, yedges)
plt.pcolormesh(X, Y, H)
plt.xlim( [np.min(x_data),np.max(x_data)] )
plt.ylim( [np.min(v_data),np.max(v_data)] )
plt.colorbar()
#ax.set_aspect('equal')
plt.title(name+'.bins_'+str(nbin))
#plt.savefig(name+'.bins_'+str(nbin)+'.svg', format='svg')
plt.savefig(name+'.bins_'+str(nbin)+'.png', format='png')
plt.clf()
