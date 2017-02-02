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


cini = 0; cend = iv.chains

for c in range(cini,cend):

    if mode == "cm":
        """ Centre of Mass  """
        dummy_x = iv.rcm[:,c,0]
        dummy_vx = iv.vcm[:,c,0]

    elif mode == "ee":
        """ End to end  """
        dummy_x = iv.r_cent[:,c,9,0]
        dummy_vx = iv.vcm[:,c,0]
    elif mode == "gr":
        """ Gyration Radius """
        dummy_x = iv.r_cent[:,c,9,0]
        dummy_vx = iv.vcm[:,c,0]

    """ Stacking all the rows data into horizontal vectors """
    if c==cini:
        x_data = dummy_x 
        vx_data = dummy_vx
    else:
        x_data = np.hstack( (x_data, dummy_x ) )
        vx_data = np.hstack( (vx_data, dummy_vx ) )

new_data = np.vstack(( x_data, vx_data ))

filename = path+'hist_'+mode+'.npy'

try:
    old_data = np.load(filename)
    joined_data = np.hstack( (old_data, new_data) )
    np.save(path, joined_data)
    print('Data appended to '+ filename)
except FileNotFoundError:
    print('File not found')
    print('Starting new file at '+filename)
    np.save(filename, new_data)
