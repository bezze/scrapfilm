#!/usr/bin/env python3

"""
INPUTS
    Implicit:
        r_all.npy   <-  savescrap.py
        v_all.npy   <-  savescrap.py

OUTPUTS
    Implicit:
        All local variables

    Explicit:
        None, its meant to be imported
"""

import numpy as np

def init_var():

    """
    Returns: r_cent, v_all

    r_cent is the particle positions, after being centered around the root of
    their respective chains and reversed the boundary conditions in x dir.

    v_all is just piped through
    """

    r_all = np.load('r_all.npy') # ALL POS (time, nchains, nbeads, xyz)
    v_all = np.load('v_all.npy') # ALL VEL

    print('r_all.shape =', r_all.shape, 'v_all.shape =',v_all.shape)

    all_0 = r_all[:,:,0,:]  # 0th bead i.e. root
    all_9 = r_all[:,:,9,:]  # 9th bead

    steps, chains, beads, ndim = r_all.shape

    """ This assumes 3 rows along the x direction """
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
    for b in range(10):
        r_cent[:,:,b,:] = r_all[:,:,b,:] - all_0

    """ UnBoundary conditions """
    def unbound (x):
        ratio = np.trunc(x/(Lx/2.))
        return x - Lx*ratio
    vec_ub = np.vectorize(unbound)

    r_aux = vec_ub(r_cent[:,:,:,0])
    r_cent[:,:,:,0] = r_aux

    return r_cent, v_all
