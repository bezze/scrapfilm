#!/usr/bin/env python3

import sys , os

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
N = 400 # Runs number

for i in range(1,N+1): # 1_runs, 2_runs, ..., N_runs
    os.chdir(str(i)+'_run')

    if mode == "cm":
        """ Centre of Mass  """
        dummy_x = iv.rcm[:,:,0]
        dummy_vx = iv.vcm[:,:,0]
    
    elif mode == "ee":
        """ End to end  """
        dummy_x = iv.r_cent[:,:,9,0]
        dummy_vx = iv.vcm[:,:,0]
    elif mode == "gr":
        """ Gyration Radius """
        dummy_x = iv.r_cent[:,:,9,0]
        dummy_vx = iv.vcm[:,:,0]
    
    new_data = np.vstack(( dummy_x, dummy_vx ))
    
    filename = path+'stack_all_'+mode+'.npy'
    
    try:
        old_data = np.load(filename)
        joined_data = np.hstack( (old_data, new_data) )
        np.save(filename, joined_data)
        print('Data appended to '+ filename)
    except FileNotFoundError:
        print('File not found')
        print('Starting new file at '+filename)
        np.save(filename, new_data)

    os.chdir('..')
