#!/usr/bin/env python3
import numpy as np
from scrapfilm import boundary

#all_0 = np.load('all_0.npy') 
#all_9 = np.load('all_9.npy') 

r_all = np.load('r_all.npy') # ALL POS (time, nchains, nbeads, xyz)
v_all = np.load('v_all.npy') # ALL VEL

print(r_all.shape, v_all.shape)

print("Loaded: all_0.npy, all_9.npy, v_all.npy")

all_0 = r_all[:,:,0,:]  # 0th bead i.e. root
all_9 = r_all[:,:,9,:]  # 9th bead

steps = r_all.shape[0]
chains = r_all.shape[1]
beads = r_all.shape[2]

a = r_all[0,3,0,0]-r_all[0,0,0,0]
b = r_all[0,1,0,1]-r_all[0,0,0,1]
Lx = r_all[0,int(chains*.5-3),0,0]+a/2
Ly = r_all[0,2,0,1]+b/2 
bounds = Lx

print("steps = ", steps)
print("chains = ", chains)
print("")
print("a = ",a )
print("b = ",b )
print("Lx = ", Lx )
print("Ly = ", Ly )
print("Using bounds = ",bounds)

r_cent = np.empty_like(r_all)

""" Centering """
for t in range(r_all.shape[0]):
    for c in range(r_all.shape[1]):
        for b in range(r_all.shape[2]):
            r_cent[t,c,b,:] = r_all[t,c,b,:] - r_all[t,c,0,:]

""" Boundary conditions """
r_aux = boundary(r_cent[:,:,:,0], bounds)
r_cent[:,:,:,0] = r_aux

rcm = np.mean(r_cent, axis=2)
vcm = np.mean(v_all, axis=2)

