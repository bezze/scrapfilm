#!/usr/bin/env python3

import init_var as iv
import numpy as np
from numpy.linalg import norm
import os
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scrapfilm import boundary

run = os.path.relpath(".","..").split('_')[0]

#vmean = np.mean( iv.v_all, axis=2 )# (time, chains, beads, xyz)

def plot_phase(cini,cend):
    
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    
    vcm = np.mean(iv.v_all, axis=2)

    for c in range(cini,cend):
        c=c+3
        """ selecting c-th chain """
        rg=iv.all_0[:,:,c]
        r0=iv.all_9[:,:,c]
        v = vcm[:,c,:] #iv.v_all[:,c,9,:]
    
        """ Centering """
        rel=r0-rg
        """ Boundary conditions """
        r_real_x= boundary(rel[0,:],iv.bounds)
        rb=np.copy(rel);rb[0,:]=r_real_x
        fiaux = np.arctan2( v[:,0],rb[0,:] )
        fi= np.array([ f + np.pi*( 1-np.sign(f) ) for f in fiaux  ])
        ax1.plot( np.cos(fi) )# ,linestyle='None', markersize=1,marker='o' )
        ax2.plot( fi )# ,linestyle='None', markersize=1,marker='o' )
    
        """ Stacking all the data into to horizontal vectors """
        #if c==cini:
        #    x_data = rb[0,:]
        #    v_data = v[0,:]
        #else:
        #    x_data = np.hstack( (x_data,rb[0,:]) )
        #    v_data = np.hstack( (v_data,v[0,:]) )
    
    fig1.savefig('./plot_cosfi_c'+str(cini)+'_'+str(cend-1)+'.svg', format='svg')
    fig2.savefig('./plot_fi_c'+str(cini)+'_'+str(cend-1)+'.svg', format='svg')
    #plt.show()
    #fig1.show(); fig2.show()
    #fig1.clf(); fig2.clf()

def phase(cini,cend,fila, step=3, plot=False ):
    """ Calcula la fase de cada oscilador de la fila,
    cuidado, no separa los bloques, hay que hacer
    cada fila por separado."""

    vcm = np.mean(iv.v_all, axis=2)

    if plot:
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)

    for c in range(cini,cend,step):
        c = c + fila
        cnxt = c + step #neighbour
        """ Last one sees the first, because of periodicty """
        if cnxt>=cend:
            cnxt=cini+fila

        """ selecting c-th chain """
        rg=iv.all_0[:,:,c]
        r0=iv.all_9[:,:,c]
        v = vcm[:,c,:] #iv.v_all[:,c,9,:]
    
        """ Centering """
        rel=r0-rg

        """ Boundary conditions """
        rb = np.copy(rel);
        rb[0,:] = boundary(rel[0,:],iv.bounds)
        #print(v[:,0])
        fiaux = np.arctan2( v[:,0],rb[0,:] )
        fi= np.array([ f + np.pi*( 1-np.sign(f) ) for f in fiaux  ])

        """ Stacking all the data into to horizontal vectors """
        if c==(cini+fila):
            fi_data = fi
        else:
            fi_data = np.vstack( (fi_data,fi) )


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

    order = (1/20.)*np.sum(np.exp(1j*fi_data), axis=0)

    return fi_data, order

def plot_synchro(matrix, order, cini, cend, row, run):

    #plt.plot(np.angle(order), linestyle='None', markersize=3, marker='^' ,label='phase')
    #plt.plot(np.absolute(order), linestyle='None', markersize=3, marker='o', label='abs')
    fig, ax1 = plt.subplots() 
    ax1.plot(np.angle(order), linestyle='solid', markersize=4,color='r', marker='^' ,label='phase') #linestyle='solid'
    y_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{2}$" ,  r"$2\pi$"]
    y_tick = np.arange(-1, 1+0.5, 0.5)
    ax1.set_yticks(y_tick*np.pi)
    ax1.set_yticklabels(y_label, fontsize=20)
    ax1.set_ylabel('Fase [rad]', color='r')
    ax1.set_ylim([-np.pi, np.pi])
    ax2 = ax1.twinx()
    ax2.plot(np.absolute(order), linestyle='None', markersize=4,color='b', marker='o', label='abs')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Magnitud', color='b')
    fig.tight_layout()
    plt.legend()
    plt.title('plot.kura.row_'+str(row)+'.c'+str(cini)+'_c'+str(cend-1)+'.run_'+str(run))
    plt.savefig('./plot.kura.row_'+str(row)+'.c'+str(cini)+'_c'+str(cend-1)+'.run_'+str(run)+'.svg', format='svg')
    #plt.show()
    plt.close()

    fig = plt.figure(1)
    plt.imshow(matrix+np.pi,interpolation='nearest', aspect='auto', cmap="gray") #'auto'
    plt.title('plot.fi_matrix.row_'+str(row)+'.c'+str(cini)+'_c'+str(cend-1)+'.run_'+str(run))
    plt.savefig('./plot.fi_matrix.row_'+str(row)+'.c'+str(cini)+'_c'+str(cend-1)+'.run_'+str(run)+'.svg', format='svg')
    plt.close()



def threeD_kura(order, cini, cend, row, run):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(np.real(order), np.imag(order), range(order.size) )
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.title('plot.kura_3d.row_'+str(row)+'.c'+str(cini)+'_c'+str(cend-1)+'.run_'+str(run))
    plt.savefig('./plot.kura_3d.row_'+str(row)+'.c'+str(cini)+'_c'+str(cend-1)+'.run_'+str(run)+'.svg', format='svg')
    #plt.show()
    plt.close()

def threeD_all(matrix, order, cini, cend, row, run, frame):
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)#, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    #ax.plot(np.real(order), np.imag(order), range(order.size), color='g' , linestyle='None', marker='o', markersize=2 )
    print(order.shape, np.real(order[frame]))
    ax.plot(np.real(order[frame]), np.imag(order[frame]), color='g' , linestyle='None', marker='o', markersize=2 )
    for c in range(matrix.shape[0]):
        #ax.plot( np.cos(matrix[c,:]), np.sin(matrix[c,:]), range(order.size), color='b', linestyle='None', marker='o', markersize=2 )
        ax.plot( np.cos(matrix[c,frame]), np.sin(matrix[c,frame]), color='b', linestyle='None', marker='o', markersize=2 )

    plt.title('plot.all_3d.row_'+str(row)+'.c'+str(cini)+'_c'+str(cend-1)+'.run_'+str(run))
    #plt.savefig('./plot.all_3d.row_'+str(row)+'.c'+str(cini)+'_c'+str(cend-1)+'.run_'+str(run)+'.svg', format='svg')
    plt.savefig('./plot.all_3d.row_'+str(row)+'.c'+str(cini)+'_c'+str(cend-1)+'.run_'+str(run)+'.fr_'+str(frame)+'.svg', format='svg')
    #plt.show()
    plt.close()

#""" Bottom  """
#for row in range(3):
#    fi_matrix, kura=phase(0,60,row)
#    plot_synchro(fi_matrix, kura, 0, 60, row, run)
#    threeD_kura(kura, 0, 60, row, run)
#   # threeD_all(fi_matrix, kura, 0, 60, row, run, 0)
#   # threeD_all(fi_matrix, kura, 0, 60, row, run, 100)
#   # threeD_all(fi_matrix, kura, 0, 60, row, run, 200)
#   # threeD_all(fi_matrix, kura, 0, 60, row, run, 300)
#   # threeD_all(fi_matrix, kura, 0, 60, row, run, 499)
#
#""" Top """
#for row in range(3):
#    fi_matrix, kura=phase(60,120,row)
#    #plot_synchro(fi_matrix, kura, 60, 120, row, run)
#    #threeD_kura(kura)
#
#print('shape = ',fi_matrix.shape)
