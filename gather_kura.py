#!/usr/bin/env python3

import sys 

try:
    mode = sys.argv[1]

except IndexError: 
        
    print("Usage:")
    print("  gather_hist MODE ")
    print("")
    print("where mode is one of the following")
    print("  cm     Uses centre of mass x coord")
    print("  ee     Uses end to end vector's x coord")
    print("  gr     Uses gyration radius ")

    raise SystemExit

import init_var as iv
import numpy as np

path = '../'

import numpy as np
import init_var as iv
from calc_phase import phase
hfchain = int(iv.chains*.5)
for tb in [ 0, hfchain ]: # 0 -> bottom, 60 -> top
    for row in range(3):
        fi_matrix, kura = phase(mode, 0+tb, hfchain+tb, row)
        filename_k = path+'kura.row_'+str(int(row+3*tb/hfchain))+'_'+mode+'.npy'
        filename_fi = path+'fi.row_'+str(int(row+3*tb/hfchain))+'_'+mode+'.npy'
        try:
            old_data_k = np.load(filename_k)
            old_data_fi = np.load(filename_fi)
            joined_k = np.hstack( (old_data_k, kura) )
            joined_fi = np.hstack( (old_data_fi, fi_matrix) )
            np.save(filename_k, joined_k)
            np.save(filename_fi, joined_fi)
            print('Data appended to '+filename_k+' and '+filename_fi)
        except FileNotFoundError:
            print('File not found')
            np.save(filename_k, kura)
            np.save(filename_fi, fi_matrix)
            print('Starting new file at '+ filename_k+' and '+filename_fi)

