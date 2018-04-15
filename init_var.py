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

""" UnBoundary conditions """
def vec_ub(x,Lx):
    def unbound (y):
        ratio = np.trunc(y/(Lx/2.))
        return y - Lx*ratio
    return np.vectorize(unbound)(x)

class init_var ():

    def __init__(self,NROWS ):
        self.r_all = np.load('r_all.npy') # ALL POS (time, nchains, nbeads, xyz)
        self.v_all = np.load('v_all.npy') # ALL VEL (idem)

        # print('r_all.shape =', r_all.shape, 'v_all.shape =',v_all.shape)

        self.steps, self.chains, self.beads, self.ndim = self.r_all.shape

        """ This assumes NROWS rows along the x direction """
        self.a = self.r_all[0,NROWS,0,0]-self.r_all[0,0,0,0]
        self.b = self.r_all[0,1,0,1]-self.r_all[0,0,0,1]
        self.Lx = self.r_all[0,int(self.chains*.5-3),0,0]+self.a/2
        self.Ly = self.r_all[0,2,0,1]+self.b/2

    def print_info(self):
        print("steps = ", self.steps)
        print("chains = ", self.chains)
        print("")
        print("a = ",self.a )
        print("b = ",self.b )
        print("Lx = ", self.Lx )
        print("Ly = ", self.Ly )
        # print("Using bounds = ",bounds)

    def centered(self):

        """
        Returns: r_cent, v_all

        r_cent is the particle positions, after being centered around the root of
        their respective chains and reversed the boundary conditions in x dir.

        v_all is just piped through
        """
        r_all = self.r_all
        v_all = self.v_all

        r_cent = np.empty_like(r_all)

        """ Centering """
        for b in range(10):
            r_cent[:,:,b,:] = r_all[:,:,b,:] - r_all[:,:,0,:]

        r_aux = vec_ub(r_cent[:,:,:,0], self.Lx)
        r_cent[:,:,:,0] = r_aux

        return r_cent, v_all
