#!/usr/bin/env python3

import sys 

try:
    mode = sys.argv[1]

except IndexError: 
        
    print("Usage:")
    print("  calc_phase MODE ")
    print("")
    print("where mode is one of the following")
    print("  cm     Uses centre of mass x coord")
    print("  ee     Uses end to end vector's x coord")
    print("  gr     Uses gyration radius ")

    raise SystemExit

import init_var as iv
import numpy as np
import os

def phase(mode,cini,cend,fila, nrow=3, plot=False ):
    """ Calcula la fase de cada oscilador de la fila,
    cuidado, no separa los bloques, hay que hacer
    cada fila por separado."""


    if plot:
        import matplotlib as mpl
        #mpl.use('Agg')
        from matplotlib import pyplot as plt

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)

    for c in range(cini,cend,nrow):
        c = c + fila

        cnxt = c + nrow #neighbour
        """ Last one sees the first, because of periodicty """
        if cnxt>=cend:
            cnxt=cini+fila

        """ MODE selection """
        if mode == "cm":
            """ Centre of Mass  """
            x = iv.rcm[:,c,0]
            v = iv.vcm[:,c,0]
        elif mode == "ee":
            """ End to end  """
            x = iv.r_cent[:,c,9,0]
            v = iv.vcm[:,c,0]
        elif mode == "gr":
            """ Gyration Radius """
            print("Gyration not finished!")
            x = iv.r_cent[:,c,9,0]
            v = iv.vcm[:,c,0]

        fiaux = np.arctan2( v, x )
        fi= np.array([ f + np.pi*( 1-np.sign(f) ) for f in fiaux  ])
        rad = np.array([ (xi**2+vi**2)**.5  for xi, vi in zip(x,v) ])

        """ Stacking all the data into to horizontal vectors """
        if c==(cini+fila):
            fi_data = fi
            rad_data = rad
        else:
            fi_data = np.vstack( (fi_data,fi) )
            rad_data = np.vstack( (rad_data,rad) )


        if plot:
            ax1.plot( np.cos(fi) )# ,linestyle='None', markersize=1,marker='o' )
            ax2.plot( fi )# ,linestyle='None', markersize=1,marker='o' )    
        # end c loop
    
    if plot:
        fig1.title('plot.cosfi.c'+str(cini)+'_c'+str(cend-1))
        fig2.title('plot.fi.c'+str(cini)+'_c'+str(cend-1))
        fig1.savefig('./plot.cosfi.c'+str(cini)+'_c'+str(cend-1)+'.svg', format='svg')
        fig2.savefig('./plot.fi.c'+str(cini)+'_c'+str(cend-1)+'.svg', format='svg')
        plt.show()

    order = ( 1/int(iv.chains/6) )*np.sum(np.exp(1j*fi_data), axis=0)

    return fi_data, rad_data, order

def diff_phase( fi_matrix ):
    """ Calcula diff fase entre cada oscilador y su vecino"""

    diff_fi = np.empty_like(fi_matrix)
    chains_per_row = diff_fi.shape[0]

    for c in range(chains_per_row):

        cnxt = c + 1 #neighbour
        """ Last one sees the first, because of periodicty """
        if cnxt>=chains_per_row:
            cnxt = 0
        diff_fi[c,:] = fi_matrix[cnxt,:] - fi_matrix[c,:] #

    return diff_fi
