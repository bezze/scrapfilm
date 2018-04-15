#!/usr/bin/env python3

# import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import init_var as iv
import sys

var = iv.init_var(3)

c=int(sys.argv[1]); b=int(sys.argv[2])

""" selecting cth chain """
r0 = var.r_all[:,c,0,:]
rb = var.r_all[:,c,b,:]
v = var.v_all[:,c,b,:]

""" (time,particle,xyz) """
rel = rb-r0
rel[:,0] = iv.vec_ub(rel[:,0], var.Lx)

#print(rg[0,:])
# Centered
plt.plot(rel[:,0],label='x: c='+str(c)+', b='+str(b))
plt.plot(rel[:,1],label='y: c='+str(c)+', b='+str(b))
plt.plot(rel[:,2],label='z: c='+str(c)+', b='+str(b))

## Off centered (in box coord)
#plt.plot(r_real_x,label='xb: c='+str(c)+', b='+str(b))
#plt.plot(r9[:,0],label='x: c='+str(c)+', b='+str(b))
#plt.plot(r9[:,1],label='y: c='+str(c)+', b='+str(b))
#plt.plot(r9[:,2],label='z: c='+str(c)+', b='+str(b))
plt.legend()
plt.show()

plt.savefig('./plot_tray.png')

