#!/usr/bin/env python3

import sys , os
import numpy as np
import init_var_func as iv

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

PWD=os.path.dirname((os.path.realpath(__file__)))
path = '../'
N = int(sys.argv[2]) # Runs number

for i in range(1,N+1): # 1_runs, 2_runs, ..., N_runs

    os.chdir(str(i)+'_run')

    rcm, vcm = iv.init_var()

    if mode == "cm":
        """ Centre of Mass  """
        dummy_x = rcm[:,:,0]
        dummy_vx = vcm[:,:,0]
    
    elif mode == "ee":
        """ End to end  """
        dummy_x = r_cent[:,:,9,0]
        dummy_vx = vcm[:,:,0]
    elif mode == "gr":
        """ Gyration Radius """
        dummy_x = r_cent[:,:,9,0]
        dummy_vx = vcm[:,:,0]
    
    new_data = np.stack(( dummy_x, dummy_vx ), axis=2)
    
    os.chdir('..')

    filename = 'stack_all_'+mode+'.npy'
    
    try:
        old_data = np.load(filename)
        joined_data = np.concatenate( (old_data, new_data), axis=0 )
        np.save(filename, joined_data)
        print('Data appended to '+ filename)
    except FileNotFoundError:
        print('File not found')
        print('Starting new file at '+filename)
        np.save(filename, new_data)

