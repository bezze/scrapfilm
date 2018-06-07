#!/usr/bin/env python3

"""
INPUTS
    Explicit:
        film_xmol   ->  Particle positions
        vel_dat     ->  Particle velocities

OUTPUTS
    Implicit:

    Explicit:
        r_all.npy
        v_all.npy
"""

#!/usr/bin/env python3

from scrapfilm import scrapfilm
from numpy import save as npsave

sf = scrapfilm('film_xmol', 'vel.dat')

r_all, mask = sf.read_film()
v_all, mask = sf.read_vel(mask)

npsave('r_all',r_all)
npsave('v_all',v_all)
