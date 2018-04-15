#!/usr/bin/env python3
"""
INPUTS
    Implicit:
        None
    Explicit:
        None

OUTPUTS
    Implicit:
        None
    Explicit:
        None

This is essentially a proto-class. A collection of functions which are used in
other scripts.
"""

import numpy as np
import sys
import pandas as p

class scrapfilm ():

    def __init__(self, archivo_film, archivo_vel):
        with open(archivo_film,'r') as film:
            # Get particle number N from film_xmol header
            self.N = int(film.readline().strip())
        self.NMON = 10 # Number of beads per chain
        self._film = archivo_film
        self._vel = archivo_vel

    def read_ (self,archivo):
        """ Esta es una rutina magica para levantar rapido los datos y meterlos
        en un array de python, el resultado tiene la forma:
            ( #FRAMES, #CHAINS, #MON, #DIM )
            """
        NCH = int(self.N/self.NMON) # Number of chains

        # This is used to filter one out of every 1202 rows
        trueIf1200Line = lambda row: True if row % (self.N+2) == 0 else False

        # This two lines guarantee compatibility of the method with film_xmol
        # and vel.dat
        cols = [1,2,3] if archivo == self._film else [0,1,2]
        colname = list('AXYZ') if archivo == self._film else list('XYZ')

        rawdat=p.read_table(archivo, header=None, usecols=cols,
                            skiprows=trueIf1200Line, names=colname,
                            delim_whitespace=True)

        NFR = int(rawdat.shape[0]/self.N) # Number of frames

        # A = NFR*NCH*(['Cl']+(self.NMON-1)*['O'])
        A = NFR*NCH*([i for i in range(self.NMON)])
        CH = NFR*sorted([i for i in range(NCH)]*self.NMON)
        FRAMES = sorted([i for i in range(NFR)]*self.N)
        # print(len(A));print(len(CH));print(len(FRAMES))

        index_array = [FRAMES,CH,A]
        tuples = list(zip(*index_array))
        index = p.MultiIndex.from_tuples(tuples, names=['frame', 'chain', 'bead'])
        chunk = p.DataFrame(data=rawdat.values, index=index, columns=list('XYZ') )

        shape = list(map(len, chunk.index.levels)) + [3] # [ frames, chains, beads, dims ]
        # print( chunk.values.reshape(shape) )
        return chunk.values.reshape(shape)

    def read_film (self):
        return self.read_(self._film)

    def read_vel (self):
        return self.read_(self._vel)

def analyze(chain,mon,archivo, rel=False):
    import pandas as p
    rawdat=p.read_table(archivo)
    datos=np.asarray(rawdat)
    N=int(list(rawdat)[0])
    step=N+1 #time step
    particle = mon+chain*NMON #particle indexing is 0-indexed
    root = chain*NMON

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
    rawdat=p.read_table(archivo)
    datos=np.asarray(rawdat)
#    N=int(list(rawdat)[0])
    step=N+1 #time step
    particle = mon+chain*NMON #particle indexing is 0-indexed
    root = chain*NMON

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
    rawdat=p.read_table(archivo)
    datos=np.asarray(rawdat)
    N=int(list(rawdat)[0])
    step=N+1 #time step
    tray_all_list=[]

    for ch in range(chains):
        particle = mon+ch*NMON #particle indexing is 0-indexed
        root = ch*NMON

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
    N=NMON*chains
    rawdat=p.read_table(archivo)
    datos=np.asarray(rawdat)
    tray_all_list = []
    step=N+1 #time step
    for ch in range(chains):
        particle = mon+ch*NMON #particle indexing is 0-indexed
        root = ch*NMON

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
    """ This function reads the film_xmol file and returns a NxT matrix, where N is
    the particle number and T is the length of the time vector. This matrix is sorted
    as it was print by the mfa program, so it needs to be further processed to be useful."""

    import pandas as p
    import xarray as xr
    with open(archivo,'r') as film:
        # Get particle number N from film_xmol header
        N = int(film.readline().strip())

    # rawdat=p.read_table(archivo)
    # rawdat=p.read_table(archivo, header=None, names=['A','X','Y','Z'],
    #                  delim_whitespace=True, chunksize=N+1)
    rawdat=p.read_table(archivo, header=None, names=['A','X','Y','Z'],
                     delim_whitespace=True, iterator=True)
    step=N+1 #time step
    pos_list_all=[]
    index=0;particle=0
    frame = 0
    chunk = rawdat.get_chunk(1201)
    print(chunk)
    # for chunk in rawdat:
    #     # xData = xr.DataArray(chunk, coords=[('particle', range(N)), ('space', locs)])
    #     xData = xr.Dataset(chunk)
    #     xData = xData[:1] # drop the first row
    #     frame +=1
    #     print(xData)
        #pos_list_chain = []
        #for ch in range(chains):
        #    pos_list=[]
        #    for mon in range(NMON):
        #        """ Recorro la cadena ch"""
        #        particle = mon+ch*NMON #particle indexing is 0-indexed
        #        root = ch*NMON


        #        #print("ch",ch,"mon",mon,"index",index)
        #        #print(rawdat.iloc[index+particle][0].split()[0:])
        #        # pos_list.append( [ float(x) for x in
        #        #                   rawdat.iloc[index+particle][0].split()[1:] ] )
        #        """end mon"""
        #    #print(vel_list)
        #    pos_list_chain.append(pos_list)
        #    """end ch"""

        #pos_list_all.append(pos_list_chain)
        #index+=step
        #"""end index"""
    #while (index+particle)<len(rawdat) :
    #    pos_list_chain = []
    #    for ch in range(chains):
    #        pos_list=[]
    #        for mon in range(NMON):
    #            """ Recorro la cadena ch"""
    #            particle = mon+ch*NMON #particle indexing is 0-indexed
    #            root = ch*NMON

    #            #print("ch",ch,"mon",mon,"index",index)
    #            #print(rawdat.iloc[index+particle][0].split()[0:])
    #            pos_list.append( [ float(x) for x in
    #                              rawdat.iloc[index+particle][0].split()[1:] ] )
    #            """end mon"""
    #        #print(vel_list)
    #        pos_list_chain.append(pos_list)
    #        """end ch"""

    #    pos_list_all.append(pos_list_chain)
    #    index+=step
    #    """end index"""
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
    N=NMON*chains
    rawdat=p.read_table(archivo)
    step=N+1 #time step
    vel_list_all=[]
    index=0;particle=0
    while (index+particle)<len(rawdat) :
        vel_list_chain = []
        for ch in range(chains):
            vel_list=[]
            for mon in range(NMON):
                """ Recorro la cadena ch"""
                particle = mon+ch*NMON #particle indexing is 0-indexed
                root = ch*NMON

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

unbound = lambda r,xb: r - np.trunc(2*r/xb)*xb

