#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import sys 
from matplotlib import pyplot as plt

archivos=sys.argv[1:]

def analyze(chain,mon,archivo, rel=False):
    import pandas as p
    nmon=10
    rawdat=p.read_table(archivo)
    datos=np.asarray(rawdat)
    N=int(list(rawdat)[0])
    step=N+1 #time step
    particle = mon+chain*nmon #particle indexing is 0-indexed
    root = chain*nmon
    
    #root = 0# For periodic CC
    root_pos = np.asarray(rawdat.iloc[root][0].split()[1:], dtype=float)# For periodic CC
    tray_list=[]
    index=0
    while (index<len(rawdat)): #
        tray_list.append( rawdat.iloc[index+particle][0].split()[1:] )
        index+=step
    
    tray = np.asarray(tray_list, dtype=np.float64)

    return tray

def analyze_v(N,chain,mon,archivo, rel=False):
    import pandas as p
    nmon=10
    rawdat=p.read_table(archivo)
    datos=np.asarray(rawdat)
#    N=int(list(rawdat)[0])
    step=N+1 #time step
    particle = mon+chain*nmon #particle indexing is 0-indexed
    root = chain*nmon
    
    #root = 0# For periodic CC
    root_pos = np.asarray(rawdat.iloc[root][0].split()[1:], dtype=float)# For periodic CC
    tray_list=[]
    index=0
    while (index<len(rawdat)): #
        tray_list.append( rawdat.iloc[index+particle][0].split()[0:] )
        index+=step
    
    tray = np.asarray(tray_list, dtype=np.float64)

    return tray

def analyze_bead(chains,mon,archivo):
    import pandas as p
    nmon=10
    rawdat=p.read_table(archivo)
    datos=np.asarray(rawdat)
    N=int(list(rawdat)[0])
    step=N+1 #time step
    tray_all_list=[]

    for ch in range(chains):
        particle = mon+ch*nmon #particle indexing is 0-indexed
        root = ch*nmon
        
        #root = 0# For periodic CC
        root_pos = np.asarray(rawdat.iloc[root][0].split()[1:], dtype=float)# For periodic CC
        tray_list=[]
        index=0
        while (index<len(rawdat)): #
            tray_list.append( rawdat.iloc[index+particle][0].split()[1:] )
            index+=step
        
        tray_all_list.append(tray_list)
        #tray = np.asarray(tray_list, dtype=np.float64)
        #np.append(tray_all,tray,axis=1)
    tray_all_aux = np.asarray(tray_all_list, dtype=np.float64)
    tray_all = np.swapaxes(tray_all_aux, 0, 2)
    #print(tray_all)

    return tray_all

def vel_bead(chains,mon,archivo):
    import pandas as p
    nmon=10
    N=nmon*chains
    rawdat=p.read_table(archivo)
    datos=np.asarray(rawdat)
    tray_all_list = []
    step=N+1 #time step
    for ch in range(chains):
        particle = mon+ch*nmon #particle indexing is 0-indexed
        root = ch*nmon
        
        tray_list=[]
        index=0
        while (index<len(rawdat)): #
            tray_list.append( rawdat.iloc[index+particle][0].split()[0:] )
            index+=step

        tray_all_list.append(tray_list)
    tray_all_aux = np.asarray(tray_all_list, dtype=np.float64)
    tray_all = np.swapaxes(tray_all_aux, 0, 2)

    return tray_all

def pos_all(chains,archivo):
    import pandas as p
    nmon=10
    N=nmon*chains
    rawdat=p.read_table(archivo)
    step=N+1 #time step
    pos_list_all=[]
    index=0;particle=0
    while (index+particle)<len(rawdat) :
        pos_list_chain = []
        for ch in range(chains):
            pos_list=[]
            for mon in range(nmon):
                """ Recorro la cadena ch"""
                particle = mon+ch*nmon #particle indexing is 0-indexed
                root = ch*nmon
                
                #print("ch",ch,"mon",mon,"index",index)
                #print(rawdat.iloc[index+particle][0].split()[0:])
                pos_list.append( [ float(x) for x in
                                  rawdat.iloc[index+particle][0].split()[1:] ] )
                """end mon"""
            #print(vel_list)
            pos_list_chain.append(pos_list)
            """end ch"""

        pos_list_all.append(pos_list_chain)
        index+=step
        """end index"""
    #print( len(pos_list_all) ) # -> tiempo
    #print( len(pos_list_all[0]) ) # -> cadenas
    #print( len(pos_list_all[0][0]) ) # -> beads
    #print( len(pos_list_all[0][0][0]) ) # -> xyz
    
    #for time in pos_list_all:
    #    print(time[0])
    #print( pos_list_all[0][0][0][:]  )

    pos_all = np.asarray(pos_list_all, dtype=np.float32)
    print(pos_all.shape)
    #vel_all = np.swapaxes(vel_all_aux, 0, 2)

    return pos_all

def vel_all(chains,archivo):
    import pandas as p
    nmon=10
    N=nmon*chains
    rawdat=p.read_table(archivo)
    step=N+1 #time step
    vel_list_all=[]
    index=0;particle=0
    while (index+particle)<len(rawdat) :
        vel_list_chain = []
        for ch in range(chains):
            vel_list=[]
            for mon in range(nmon):
                """ Recorro la cadena ch"""
                particle = mon+ch*nmon #particle indexing is 0-indexed
                root = ch*nmon
                
                #print("ch",ch,"mon",mon,"index",index)
                #print(rawdat.iloc[index+particle][0].split()[0:])
                vel_list.append( [ float(x) for x in
                                  rawdat.iloc[index+particle][0].split()[0:] ] )
                """end mon"""
            #print(vel_list)
            vel_list_chain.append(vel_list)
            """end ch"""

        vel_list_all.append(vel_list_chain)
        index+=step
        """end index"""
    #print( len(vel_list_all) ) # -> tiempo
    #print( len(vel_list_all[0]) ) # -> cadenas
    #print( len(vel_list_all[0][0]) ) # -> beads
    #print( len(vel_list_all[0][0][0]) ) # -> xyz
    
    #for time in vel_list_all:
    #    print(time[0])
    #print( vel_list_all[0][0][0][:]  )

    vel_all = np.asarray(vel_list_all, dtype=np.float32)
    print(vel_all.shape)
    #vel_all = np.swapaxes(vel_all_aux, 0, 2)

    return vel_all


#def boundary(r,xb):
#
##    out = np.copy(r)
##    x = np.trunc(2*r/xb)*xb #a*(4-0)
##    out = r - x #+xbound
#
#    return r - np.trunc(2*r/xb)*xb #out

boundary = lambda r,xb: r - np.trunc(2*r/xb)*xb

