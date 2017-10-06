#!/usr/bin/env python3

import sys , os
import numpy as np

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
N = 40 # Runs number

for i in range(1,N+1): # 1_runs, 2_runs, ..., N_runs
    os.chdir(str(i)+'_run')
    import init_var as iv

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
    
    new_data = np.stack(( dummy_x, dummy_vx ), axis=2)
    
    filename = path+'stack_all_'+mode+'.npy'
    
    try:
        old_data = np.load(filename)
        joined_data = np.concatenate( (old_data, new_data), axis=0 )
        np.save(filename, joined_data)
        print('Data appended to '+ filename)
    except FileNotFoundError:
        print('File not found')
        print('Starting new file at '+filename)
        np.save(filename, new_data)

    os.chdir('..')
