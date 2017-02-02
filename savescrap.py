#!/usr/bin/env python3

import numpy as np
import sys 
from scrapfilm import analyze_bead, vel_all, vel_bead, pos_all

N = int(sys.argv[1])
archivos=sys.argv[2:]

v_all = vel_all(N,archivos[1])
np.save('v_all',v_all) #print(v_all.shape) 

r_all = pos_all(N,archivos[0])
np.save('r_all',r_all)

#b=9

#all_9 = analyze_bead(120,b,archivos[0])
#np.save('all_9',all_9)
#
#all_0 = analyze_bead(120,0,archivos[0])
#np.save('all_0',all_0)
#
##v_all_9 = vel_bead(120,b,archivos[1])
##np.save('v_all_9',v_all_9)
#

