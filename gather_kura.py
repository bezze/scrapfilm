#!/usr/bin/env python3

import sys, os

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

path = '../'

import numpy as np
import init_var as iv
from calc_phase import phase, diff_phase

print(os.getcwd())

hfchain = int(iv.chains*.5)
for tb in [ 0, hfchain ]: # 0 -> bottom, 60 -> top
    for row in range(3):
        fi_matrix, rad_matrix, kura = phase(mode, 0+tb, hfchain+tb, row)
        dfi_matrix = diff_phase( fi_matrix )
        filename_k = path+'kura.row_'+str(int(row+3*tb/hfchain))+'_'+mode+'.npy'
        filename_fi = path+'fi.row_'+str(int(row+3*tb/hfchain))+'_'+mode+'.npy'
        filename_rad = path+'rad.row_'+str(int(row+3*tb/hfchain))+'_'+mode+'.npy'
        filename_dfi = path+'dfi.row_'+str(int(row+3*tb/hfchain))+'_'+mode+'.npy'
        try:
            old_data_k = np.load(filename_k)
            old_data_fi = np.load(filename_fi)
            old_data_rad = np.load(filename_rad)
            old_data_dfi = np.load(filename_dfi)
            joined_k = np.hstack( (old_data_k, kura) )
            joined_fi = np.hstack( (old_data_fi, fi_matrix) )
            joined_rad = np.hstack( (old_data_rad, rad_matrix) )
            joined_dfi = np.hstack( (old_data_dfi, dfi_matrix) )
            np.save(filename_k, joined_k)
            np.save(filename_fi, joined_fi)
            np.save(filename_rad, joined_rad)
            np.save(filename_dfi, joined_dfi)
            print('Data appended to '+filename_k+' and '+filename_fi)
        except FileNotFoundError:
            print('File not found')
            np.save(filename_k, kura)
            np.save(filename_fi, fi_matrix)
            np.save(filename_rad, rad_matrix)
            np.save(filename_dfi, dfi_matrix)
            print('Starting new file at '+ filename_k+' and '+filename_fi)

