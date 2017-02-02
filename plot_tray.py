#!/usr/bin/env python3

import init_var as iv
import numpy as np
from numpy.linalg import norm
import sys 
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
from scrapfilm import analyze, boundary, analyze_all

c=int(sys.argv[1]); b=9

""" selecting cth chain """
r0=iv.all_root[:,:,c]
r9=iv.all_9[:,:,c]
v=iv.v_all_9[:,:,c]

print(iv.v_all_9.shape)

""" (xyz,time,particle) """
rel=r9-r0
r_real_x= boundary(rel[0,:],iv.bounds)
rb=np.copy(rel);rb[0,:]=r_real_x

#print(rg[0,:])
# Centered
plt.plot(rb[0,:],label='x: c='+str(c)+', b='+str(b))
plt.plot(rb[1,:],label='y: c='+str(c)+', b='+str(b))
plt.plot(rb[2,:],label='z: c='+str(c)+', b='+str(b))

## Off centered (in box coord)
#plt.plot(r_real_x,label='xb: c='+str(c)+', b='+str(b))
#plt.plot(r9[:,0],label='x: c='+str(c)+', b='+str(b))
#plt.plot(r9[:,1],label='y: c='+str(c)+', b='+str(b))
#plt.plot(r9[:,2],label='z: c='+str(c)+', b='+str(b))
plt.legend()
plt.show()

plt.savefig('./plot_tray.png')

