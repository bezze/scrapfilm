#!/usr/bin/env python3

"""
Esto recorre las carpetas i_run, toma los archivos r_all.npy y v_all.npy, y
los junta en un gran archivo stack_all_{MODE}.npy

INPUTS
    Explicit:
        {MODE}  cm, ee, gr
        M       start run
        N       end run
    Implicit:
        r_all.npy   <-  savescrap.py
        v_all.npy   <-  savescrap.py

OUTPUTS
    Explicit:
        stack_all_{MODE}.npy  (frames, chains, beads, dims, r&vel)
    Implicit:
        None
"""

import sys , os
import numpy as np
import init_var_func as iv

try:
    mode = sys.argv[1]

except IndexError:

    print("Usage:")
    print("  stack_r_all.py MODE ")
    print("")
    print("where mode is one of the following")
    print("  cm     Uses centre of mass x coord")
    print("  ee     Uses end to end vector's x coord")
    print("  gr     Uses gyration radius ")

    raise SystemExit

PWD=os.path.dirname((os.path.realpath(__file__)))
path = '../'
M = int(sys.argv[2]) # Runs number
N = int(sys.argv[3]) # Runs number

for i in range(M,N+1): # M_runs, M+1_runs, ..., N_runs

    os.chdir(str(i)+'_run')

    r_cent, v_all = iv.init_var()

    vcm = np.mean(v_all, axis=2)

    if mode == "cm":
        """ Centre of Mass  """
        rcm = np.mean(r_cent, axis=2)
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

    print('RUN = ', i)
    try:
        old_data = np.load(filename)
        joined_data = np.concatenate( (old_data, new_data), axis=0 )
        np.save(filename, joined_data)
        print('Data appended to '+ filename)
    except FileNotFoundError:
        print('File not found')
        print('Starting new file at '+filename)
        np.save(filename, new_data)

