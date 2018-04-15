#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.colors as mcs
import matplotlib.pyplot as plt

import numpy as np
from numpy import conjugate as cj
import sys, pickle, os
import scipy.signal

FILEPATH='/'.join(os.path.realpath(__file__).split('/')[:-1])+'/'
# Import custom module 'metadata' to add metada to png files
import importlib.util
spec = importlib.util.spec_from_file_location("metadata", FILEPATH+"metadata.py")
md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md)

meta = md.produce_meta() # Produce metadata
print(">>>>>>>>>METADATA>>>>>>>>>")
for entry in meta: print(entry,'=', meta[entry])
print("<<<<<<<<<<<<<<<<<<<<<<<<<<")
print()
#-------------------------------------------------------------------------------
def loadargs(args):

    RUNS, ini, end, rows, Name = args

    RUNS = int(RUNS)
    ini = float(ini)
    end = float(end)#+1
    rows = int(rows)

    return RUNS, ini, end, rows, Name

def fake_loadargs():

    try:
        with open('loadargs.dat') as f:
            RUNS, ini, end, rows, Name = f.readline().split()
    except FileNotFoundError:
        print("ERROR: Supply SETTINGS or create a loadargs.dat")
        sys.exit()

    """ loadargs.dat is a textfile that contains RUNS ini end rows """
    RUNS = int(RUNS)
    ini = float(ini)
    end = float(end)#+1
    rows = int(rows)

    return RUNS, ini, end, rows, Name

def slicer(SETTINGS):

    try:
        RUNS, ini, end, rows, Name = loadargs(SETTINGS)
    except ValueError:
        RUNS, ini, end, rows, Name = fake_loadargs()

    path = '.'
    all_matrix = np.load(path+'/stack_all_cm.npy') # t, chains, x|v
    N, Nch,_, = all_matrix.shape

    TRUN=N/(RUNS+1.)
    nini=int(np.floor(TRUN*ini)-1)
    nend=int(np.ceil(TRUN*end)+1)
    # zini = ini-1
    # zend = end-1
    # TRUN = N/RUNS
    # nini = int( TRUN*(ini-1) )
    # nend = int( TRUN*(end-1) )


    xcm = all_matrix[nini:nend,:,0] # t, chains
    vcm = all_matrix[nini:nend,:,1]

    l,m = xcm.shape # time, N
    Npr = int(m/rows)
    fi_rows = np.empty((l,Npr,rows))
    X = np.empty((l,Npr,rows))
    V = np.empty((l,Npr,rows))
    imap={}
    rmap={}
    for half in range(2): # 0 - down, 1 - top
        for fila in range(int(rows/2)):
            for chain in range(Npr):
                # fi_rows[:,chain,int(fila+half*rows/2)]= fi[:, int(chain*rows/2 + fila + half*m/2)]
                X[:,chain,int(fila+half*rows/2)]= xcm[:, int(chain*rows/2 + fila + half*m/2)]
                V[:,chain,int(fila+half*rows/2)]= vcm[:, int(chain*rows/2 + fila + half*m/2)]
                # print(int(chain*rows/2 + fila + half*m/2))
                a = (chain, int( fila+half*rows/2 ) )
                b = int(chain*rows/2 + fila + half*m/2)
                imap[b] = a  #  xcm[i] --> (chain, fila)
                rmap[a] = b  # (chain, fila) --> xcm[i]

    # fiaux = np.arctan2( V, X )
    fiaux = - np.angle( scipy.signal.hilbert(X) )
    fi_rows= np.array([ f + np.pi*( 1-np.sign(f) ) for f in fiaux  ])

    R = np.zeros((l,rows), dtype='complex128')
    for fila in range(rows):
        R[:,fila] = np.sum(np.exp(1j*fi_rows[:,:,fila]), axis=1)/Npr
    t=np.linspace(0,RUNS+1,N)[nini:nend]

    return X, V, fi_rows, R, t, imap, rmap

def pleg_col(Npr):
    if Npr % 2 == 0:
        pleg_col = int(Npr/2.-1)
    else:
        pleg_col = int((Npr+1)/2.)
    return pleg_col

def plegado(A,j):
    # Pliega la matriz de correlacion sobre la
    # col j
    # A[i, j+1] -> (A[i,j+1] + A[i,j-1])/2
    col = (A.shape[0],1)
    flip = np.copy(A[:,:j])
    for i in range(j):
        flip[:,i] = A[:,j-1-i]
    if A.shape[1] % 2 == 0:
        B=np.concatenate( ( (A[:,j].reshape(col)), ((A[:,j+1:-1] + flip)*.5), (A[:,-1].reshape(col)) ) , axis=1)
    else:
        B=np.concatenate( ( (A[:,j].reshape(col)), ((A[:,j+1:] + flip)*.5) ) , axis=1)
    return B

cm2in = lambda w,h: (w/2.54,h/2.54)
ed2cen = lambda ed:  .5*(ed[1:]+ed[:-1])

def create_custom_cmap ():
    colors = ["black", "white", "black"]
    cmap = mcs.LinearSegmentedColormap.from_list("", colors)
    return cmap

#-------------------------------------------------------------------------------

def peak_finder(Fmatrix, thresh, neighborhood, iterations=1):
    # Peak finder

    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, iterate_structure

    neighborhood = neighborhood.astype(bool)
    local_max = maximum_filter(Fmatrix, footprint=neighborhood)==Fmatrix
    background = (Fmatrix <= thresh)
    eroded_background = binary_erosion(background, structure=neighborhood, iterations=iterations, border_value=1)
    detected_peaks = local_max ^ eroded_background
    peaks_index = np.transpose( np.nonzero( detected_peaks ) )
    return peaks_index, detected_peaks


def plot_r2(SETTINGS,ROWLIST):
    xcm, vcm, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = SETTINGS
    fig, axROWS = plt.subplots(len(ROWLIST), 1)
    cmap = plt.cm.tab10(range(len(ROWLIST)))
    r2 = R*cj(R)
    if type(axROWS) == np.ndarray:
        for i,ROW in enumerate(ROWLIST):
            axROWS[i].plot(t,r2[:,ROW].real, color = cmap[i] )
        axROWS[int(len(axROWS)*.5)].set_ylabel(r'$\left\Vert R\right\Vert^{2}$', fontsize=13)
        axROWS[-1].set_xlabel(r'$Nº\;Corrida\;[10^6 \Delta t]$', fontsize=13)
    else:
        for ROW in ROWLIST:
            axROWS.plot(t,r2[:,ROW].real)
        axROWS.set_ylabel(r'$\left\Vert R\right\Vert^{2}$', fontsize=13)
        axROWS.set_xlabel(r'$Nº\;Corrida\;[10^6 \Delta t]$', fontsize=13)
    # ax.plot(t,r2[:,0].real)
    print(np.mean(r2.real,axis=0))
    return fig, axROWS


def plot_x(SETTINGS, fila=0, chain=0):
    xcm, vcm, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = SETTINGS

    #fila = 0; chain = 0
    x = xcm[:,chain,fila ]

    fig, ax = plt.subplots(1,1)
    ax.plot(t,x)

    return fig, ax

def plot_xvfi(SETTINGS, fila=0, chain=0):
    xcm, vcm, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = SETTINGS

    #fila = 0; chain = 0
    x = xcm[:,chain,fila ]
    v = vcm[:,chain,fila ]
    fi = fi_rows[:,chain,fila ]

    fig, ax = plt.subplots(3,1)
    size_ = (16,8)
    fig.set_size_inches(cm2in(*size_),forward=True)

    ax[0].plot(t,x, c='tab:blue')
    ax[0].set_ylabel(r'$x\;[\sigma]$', fontsize=11)
    ax[1].plot(t,v, c='tab:orange')
    ax[1].plot(t,v_hil, c='tab:red')
    ax[1].set_ylabel(r'$v\;[\sigma/\tau]$', fontsize=11)
    ax[2].plot(t,fi, c='tab:green')
    ax[2].plot(t,fi_hil, c='tab:red')
    ax[2].set_ylabel(r'$\varphi\;[rad]$', fontsize=11)
    ax[2].set_xlabel(r'$Nº\;Corrida\;[10^6 \Delta t]$', fontsize=11)
    fig.tight_layout()

    return fig, ax

def plot_fft_x(sample, SETTINGS):

    xcm, vcm, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = SETTINGS #loadargs( sys.argv[2:] )
    T, Npr, rows = fi_rows.shape

    r2 = (R*cj(R)).real
    label_bp=r'$\left\langle r^{{2}}\right\rangle _{{{0}}}={1:.2f}$'
    r2mean=np.mean(r2,axis=0)

    fig, ax = plt.subplots(1,1)
    # size_zoom = (12.3, 10) #(11,10)
    # size_ = size_zoom
    size_ = (16,8)
    fig.set_size_inches(cm2in(*size_),forward=True)

    for ROW in range(rows):
        x = xcm[:, :, ROW ]
        v = vcm[:, :, ROW ]
        F = np.fft.rfft(x, axis = 0)
        Fv = np.fft.rfft(v, axis = 0)
        E = (F*np.conjugate(F)).real
        Ev = (Fv*np.conjugate(Fv)).real
        f = np.fft.rfftfreq(x.shape[0], d=1./sample)
        # print( x.shape[0], f.shape )
        meanE = np.mean(E, axis=1)
        meanEv = np.mean(Ev, axis=1)
        # ax.plot(f, meanE, label = "{0}".format(ROW))
        # ax.plot(f, meanE, label = label_bp.format(ROW, r2mean[ROW]))

        # ax.set_xlim([27.25,30])
        # ax.set_ylim([10**5, 10**9.5])
        ax.set_xlabel(r'$f\;[\frac{10^{-6}}{\Delta t}]$', size = 15)
        ax.set_ylabel(r'$\mathscr{F}\cdot\mathscr{F}^{\mathscr{*}}$', size = 15)

        # print(ROW, f[ np.argmax(meanE) ] )
        try:
            meanRow += meanE
            meanRowv += meanEv
        except:
            meanRow = meanE
            meanRowv = meanEv

    ax.plot(f,meanRow/rows, label = r"$\left\langle \left\Vert \mathscr{F}[X_{CM}]\right\Vert ^{2}\right\rangle$", zorder=1)
    ax.plot(f,meanRowv/rows, label = r"$\left\langle \left\Vert \mathscr{F}[V_{CM}]\right\Vert ^{2}\right\rangle$", zorder=0)
    ax.legend(fontsize=11)

    ax.set_yscale('log')
    fig.tight_layout()
    dat = [f, meanRow/ROW]
    return ax, dat

def plot_phase_dyn():

    from scipy import signal
    from scipy import ndimage
    import mahota as mh

    xcm, vcm, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = loadargs( sys.argv[2:] )

    T, Npr, rows = fi_rows.shape

    # print(fi_rows.shape)
    matrix = fi_rows[:,:,2].T
    # neigh = np.zeros((2,6)); neigh[:,:4]=1
    neigh = np.ones((1,6)); neigh[:,:5]=0
    thresh = 2*np.pi*0.3 #np.max(matrix)*.5
    matrix =ndimage.gaussian_filter( matrix,.4 )
     #, structure=neigh) #, structure=np.array([0,1,0]))
    peak_index, detected_peaks = peak_finder(matrix, thresh, neigh, 1)
    struct1 = np.zeros((3,3)); struct1[:,1] = 1
    struct2 = np.ones((3,3)); struct2[:,1] = 0
    propag_img = detected_peaks
    for i in range(20):
        eroded_img = ndimage.binary_erosion(propag_img,structure=struct1, iterations=2)
        propag_img = ndimage.binary_propagation(eroded_img, mask=detected_peaks, structure=struct2)
    reconstruct_img = propag_img
    tmp = np.logical_not(reconstruct_img)
    eroded_tmp = ndimage.binary_erosion(tmp)
    reconstruct_final = np.logical_not(ndimage.binary_propagation(eroded_tmp, mask=tmp))
    # reconstruct_final= ndimage.binary_closing(reconstruct_final)

    # sox = ndimage.sobel( detected_peaks.astype(int), axis=0, mode='constant' )
    # soy = ndimage.sobel( detected_peaks.astype(int), axis=1, mode='constant' )
    # sob = np.hypot( sox, soy)


    fig, ax = plt.subplots(1,2,gridspec_kw = {'width_ratios':[9, 1]} )
    fig2, ax2 = plt.subplots(1,2)
    plot = ax[0].imshow(matrix,interpolation='none', aspect='auto', cmap="gray",vmin=0, vmax=2*np.pi, extent=[t[0],t[-1],19.5,-0.5])
    plot2 = ax2[0].imshow(detected_peaks,interpolation='none', aspect='auto',cmap="gray", vmin=0, vmax=1, extent=[t[0],t[-1],19.5,-0.5])
    plot2 = ax2[1].imshow(reconstruct_final,interpolation='none', aspect='auto',cmap="gray", vmin=0, vmax=1) #, extent=[t[0],t[-1],19.5,-0.5])
    # plot2 = ax2[1].contour(reconstruct_final, [0.5],lw=2,c='r')

    # plot2 = ax2.imshow(sob,interpolation='none', aspect='auto',cmap="gray", vmin=0, vmax=1, extent=[t[0],t[-1],19.5,-0.5])
    # peakind = signal.find_peaks_cwt(matrix[0,:], np.arange(1,15))
    # for p in peakind:
    #     ax[0].scatter( t[p], matrix[0,p] )
    ax[0].set_xlabel(r'$Corrida \: [10^6 \: \Delta t ] $')
    ax[0].set_ylabel('Oscilador')
    ax[1].set_axis_off()
    cbar = fig.colorbar(plot,ax=ax[1] , fraction=1)
    ticks = (np.arange(0,2.5,.5))*np.pi
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    # fig.savefig('phase_dynamics_'+Name+'.png')
    # for i in range(0,len(t),18):
    #     ax[0].axvline(t[i],color='r')
    plt.show()


def find_stripes(matrix):

    import mahotas as mh
    import scipy.ndimage as nd

    m2gr = lambda A: np.array( A/np.max(A) * 255, dtype = np.uint8)

    im = m2gr( matrix )
    im = nd.gaussian_filter( im, [0.4, .2] )
    im_t = im>200; im_o = im_t
    # -------------
    so = np.ones((1,3)); so[0,1]=0; so=so.astype(bool)#
    sc = ~ so
    for i in range(1):
        im_o = nd.binary_opening( im_o, so, 1)
        im_o = nd.binary_closing(im_o, sc, 1)
    # -------------

    im_o = nd.binary_dilation(im_o, np.ones((4,1)), 1)

    s=np.ones((3,3)); s[1,1]=0
    im_o = nd.binary_closing(im_o, s, 1)

    s=np.ones((2,2));
    im_o = nd.binary_erosion(im_o,s, 1)

    labeled,nr_objects = mh.label(im_o)

    return labeled,nr_objects


def stripes_ROW(ROWLIST, SETTINGS):

    xcm, vcm, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = SETTINGS

    r2 = R*cj(R)

    T, Npr, rows = fi_rows.shape
    DAT = [RUNS, ini, end, rows, Name, T, Npr ]

    for ROW in ROWLIST:

        matrix = fi_rows[:,:,ROW].T # (osc, time)

        labeled, n_obj = find_stripes(matrix)
        pendientes=[]
        tiempos=[]

        # Taking means
        centers = np.empty( ( labeled.shape[0], n_obj ) )
        centers[:,:] = np.nan
        for obj in range(1,n_obj+1): # objects start at 1
            filt = labeled == obj
            obj_i= np.where( filt )
            length = len( set(np.transpose(obj_i)[:,0]) )
            if length > int(Npr*.6): # stripe-length % of system
                for fila in range( centers.shape[0] ):
                    centers[fila, obj-1] = np.mean( t [ filt[fila,:] ] )
                center_obj = centers[:, obj-1]
                nanfilt = ~np.isnan(center_obj)
                if any(nanfilt):
                    cent = center_obj[nanfilt]
                    vec = (np.arange(0,Npr)+.5)[nanfilt]
                    P = np.polyfit( vec, cent, 1 )
                    pendientes.append(P[0])
                    tiempos.append( np.mean( cent ) )

        block_name = "block_"+str(ROW)+"_" + Name

        np.savez( block_name, pendientes=pendientes,
                tiempos=tiempos, DAT = DAT, centers = centers, r2=r2)

def plot_stripes_ROW(ROWLIST,SETTINGS):

    xcm, vcm, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = SETTINGS

    r2 = R*cj(R)
    T, Npr, rows = fi_rows.shape
    DAT = [RUNS, ini, end, rows, Name, T, Npr ]

    cm2in = lambda w,h: (w/2.54,h/2.54)
    print(len(ROWLIST))
    fig_pend, ax_pend = plt.subplots(len(ROWLIST),1)
    cmap = create_custom_cmap ()
    # np.save("matrix", matrix )
    for index, ROW in enumerate(ROWLIST):

        matrix = fi_rows[:,:,ROW].T # (osc, time)
        # matrix[0,:] = 0*matrix[0,:]
        labeled, n_obj = find_stripes(matrix)
        pendientes=[]
        tiempos=[]
        color_bar_TRUE = True
        if color_bar_TRUE:
            fig, ax = plt.subplots(1,2,gridspec_kw = {'width_ratios':[9, 1]} )
            axim = ax[0]
            axcb = ax[1]
        else:
            fig, ax = plt.subplots(1,1)
            axim = ax
            p=1.; fig.set_size_inches(cm2in(p*12,p*12),forward=True)

        extent = [t[0],t[-1],0,Npr]

        implot = axim.imshow(matrix,interpolation='none', cmap=cmap, # cmap="gray",
                             vmin=0,
                             vmax=2*np.pi, extent=extent, aspect='auto',
                             origin='lower')

        ytick_labels = np.arange(0,Npr,10)
        ytick_pos = np.arange(0,Npr,10)+.5
        axim.set_xlabel(r'$Nº\;Corrida \: [10^6 \: \Delta t ] $', fontsize=13)
        axim.set_ylabel('Oscilador', fontsize=13)
        axim.set_ylim([ 0, Npr])
        axim.set_xlim([t[0],t[-1]])
        axim.set_yticks(ytick_pos)
        axim.set_yticklabels(ytick_labels)

        # Colorbar
        if color_bar_TRUE:
            axcb.set_axis_off()
            cbar = fig.colorbar(implot,ax=axcb , fraction=1)
            ticks = (np.arange(0,2.5,.5))*np.pi
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])

        fig.tight_layout()

    # Taking means
        centers = np.empty( ( labeled.shape[0], n_obj ) )
        centers[:,:] = np.nan
        for obj in range(1,n_obj+1): # objects start at 1
            filt = labeled == obj
            obj_i= np.where( filt )
            length = len( set(np.transpose(obj_i)[:,0]) )
            if length > int(Npr*.6): # stripe-length % of system
                for fila in range( centers.shape[0] ):
                    centers[fila, obj-1] = np.mean( t [ filt[fila,:] ] )
                center_obj = centers[:, obj-1]
                nanfilt = ~np.isnan(center_obj)
                if any(nanfilt):
                    cent = center_obj[nanfilt]
                    vec = (np.arange(0,Npr)+.5)[nanfilt]
                    P = np.polyfit( vec, cent, 1 )
                    pendientes.append(P[0]*10**3)
                    tiempos.append( np.mean( cent ) )
                    # axim.plot(  cent, vec, color='xkcd:blue', marker='o', ms=3, ls='', lw=3, zorder=2 )
                    # axim.plot( np.polyval(P, np.arange(0,Npr+1) ), np.arange(0,Npr+1), color='xkcd:orange', lw=4, zorder=1 )
        # for m in range(11):
        #     axim.axhline( 18.2*m, color='xkcd:green' )
        # axim.axhline( 100, color='xkcd:green' )

        size_ = (11.2,4.8)
        fig_pend.set_size_inches(cm2in(*size_),forward=True)

        if type(ax_pend) == np.ndarray:
            ax_pend_current_plot = ax_pend[index]
            ax_pend[int(len(ROWLIST)*.5)].set_ylabel(r'$a/v\;[10^6 \Delta t]$')
            ax_pend[-1].set_xlabel(r'$Nº\;Corrida \: [10^6 \: \Delta t ] $')
        else:
            ax_pend_current_plot = ax_pend
            ax_pend.set_ylabel(r'$a/v\;[10^6 \Delta t]$')
            ax_pend.set_xlabel(r'$Nº\;Corrida \: [10^6 \: \Delta t ] $')

        ax_pend_current_plot.plot(tiempos, pendientes,'o', ms=1)
        ax_pend_current_plot.set_xlim([t[0],t[-1]])
        ax_pend_current_plot.text( -0.10,0.99, "$\\times 10^{-"+str(3)+"}$", fontsize=9, transform = plt.gca().transAxes)


        # ax_pend[1].plot(t,r2[:,ROW].real, lw=.5)
        # ax_pend[1].set_xlabel(r'$Corrida \: [10^6 \: \Delta t ] $')
        # ax_pend[1].set_ylabel(r'$r^2$')
        # ax_pend[1].set_xlim([t[0],t[-1]])
        fig_pend.tight_layout()

        if not color_bar_TRUE:
            p=1;fig.set_size_inches(cm2in(p*16,p*11) )
            fig.tight_layout()
    plots = ([fig, ax], [fig_pend,ax_pend])
    # fig_pend.savefig('pendientes.png')
    # md.add_meta('pendientes.png', meta)

    return plots

def plot_fft2(samp_t, ROWLIST, SETTINGS):
    from scipy import signal
    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, iterate_structure

    xcm, vcm, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name =  SETTINGS

    mag = lambda F: (F*cj(F)).real
    T, Npr, rows = fi_rows.shape

    for ROW in ROWLIST:
        fig, ax = plt.subplots(1,1)
        print(ROW)
        matrix = fi_rows[:,:,ROW].T
        Fmatrix = np.fft.fftshift( mag( np.fft.fftn(matrix) ) )
        # Fmatrix =  mag( np.fft.fftn(matrix) )
        shape = Fmatrix.shape
        F = Fmatrix.flatten()
        F[np.argmax(F)] = 0
        newmax = F[np.argmax(F)]
        Fmatrix = F.reshape(shape)
        thresh = Fmatrix>0.5*newmax
        samp_e = 1. # vecinos
        t_freq = np.fft.fftshift( np.fft.fftfreq(len(t), d=1./samp_t) )
        e_freq = np.fft.fftshift( np.fft.fftfreq(Npr, d=1./samp_e) )
        # print(e_freq)
        defreq=e_freq[1]-e_freq[0]
        dtfreq=t_freq[1]-t_freq[0]
        logF = np.log( Fmatrix )
        filt = logF>np.max(logF)*.7
        # print( logF.shape, filt.shape )
        # # Peak finder
        # neighborhood = np.zeros((5,5)); neighborhood[2,2]=1
        # neighborhood = neighborhood.astype(bool)
        # local_max = maximum_filter(Fmatrix, footprint=neighborhood)==Fmatrix
        # background = (Fmatrix <= np.max(Fmatrix)*.9)
        # eroded_background = binary_erosion(background, structure=neighborhood, iterations=1, border_value=1)
        # detected_peaks = local_max ^ eroded_background
        # peaks_index = np.transpose( np.nonzero( detected_peaks ) )

        # for ipeak in peaks_index:
        #     espacial = "i={0}, k={1:f} 1/vec, la={2:f} vec".format(ipeak[0],e_freq[ipeak[0]],1/e_freq[ipeak[0]])
        #     temporal = "i={0}, f={1:f} 1/, T={2:f} corrida".format(ipeak[1],t_freq[ipeak[1]],1/t_freq[ipeak[1]])
        #     print( espacial )
        #     print( temporal )

        # ax.imshow( np.log( Fmatrix*thresh +1 )  ,aspect='auto', extent=[ t_freq[0], t_freq[-1], e_freq[0]+defreq, e_freq[-1]+defreq  ] )
        ax.imshow( logF*filt  ,aspect='auto', extent=[ t_freq[0], t_freq[-1], e_freq[0]+defreq, e_freq[-1]+defreq  ] )
        ax.axhline(0,c='r')
        ax.axvline(0,c='r')
        # plt.show()
    return fig, ax

def corr(ROWLIST, SETTINGS):

    xcm, vcm, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name =  SETTINGS

    r2 = (R*cj(R)).real

    mag = lambda F: (F*cj(F)).real
    T, Npr, rows = fi_rows.shape

    def rot(v,j):
        # posiciona el indice j primero, preservando orden ciclico
        v2 = np.copy(v); n = len(v)
        i = int(n/2 + 1 + j)
        if i > n-1:
            i = int(i - n)
        v2[:n-i] = v[i:]
        v2[n-i:] = v[:i]
        return v2

    def plegado(A,j):
        # Pliega la matriz de correlacion sobre la col j
        # A[i, j+1] -> (A[i,j+1] + A[i,j-1])/2
        col = (A.shape[0],1)
        flip = np.copy(A[:,:j])
        for i in range(j):
            flip[:,i] = A[:,j-1-i]
        if A.shape[1] % 2 == 0:
            B=np.concatenate( ( (A[:,j].reshape(col)), ((A[:,j+1:-1] + flip)*.5), (A[:,-1].reshape(col)) ) , axis=1)
        else:
            B=np.concatenate( ( (A[:,j].reshape(col)), ((A[:,j+1:] + flip)*.5) ) , axis=1)
        return B

    def CrossCorr(ROW):
        # fi = fi_rows[:,:,ROW] # time, chain
        c = xcm[:,:,ROW] # time, chain
        C = 0*np.empty((Npr,Npr))
        for i in range(Npr):
            sig_i = np.std(c[:,i])
            c_i = c[:,i] - np.mean(c[:,i])
            for j in range(i+1):
                sig_j = np.std(c[:,j])
                c_j = c[:,j] - np.mean(c[:,j])
                C[i,j] = np.mean(c_i*c_j)/(sig_i*sig_j)
                C[j,i] = C[i,j]

        C_cent = np.empty((Npr,Npr))
        for ch in range(Npr):
            C_cent[ch,:] = rot(C[ch,:],ch)
        return C_cent

    if Npr % 2 == 0:
        pleg_col = int(Npr/2.-1)
    else:
        pleg_col = int((Npr+1)/2.)

    Cross = 0*np.empty((Npr,Npr,rows))

    r2mean = np.mean(r2, axis = 0)
    text_start = r"$\left\langle r^{{2}}\right\rangle_"

    fig, ax = plt.subplots(1,1)
    for ROW in ROWLIST:
        Cross_ROW = CrossCorr(ROW)
        CrossPleg = plegado(Cross_ROW, pleg_col)
        CrossMean = np.mean( CrossPleg, axis=0 )
        CrossErr = np.std( CrossPleg, axis = 0)/(CrossMean.shape[0])**.5
        vecinos = range(1, CrossMean[1:].size+1)
        text = text_start+str(ROW) + r"={0:.2f}$".format(r2mean[ROW])
        # ax.plot(vecinos, CrossMean[1:],'-o', ms=3, label=text)
        ax.errorbar(vecinos, CrossMean[1:], yerr=CrossErr[1:],capsize=5, fmt='-o', ms=3, label=text)
        ax.axhline(0, color='k')
        Cross[:,:,ROW] = Cross_ROW
    ax.legend()
    return ax, Cross



def plot_polar(ROWLIST, SETTINGS):

    X, V, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = SETTINGS

    r2 = (R*cj(R)).real

    Tsize, Npr, rows = fi_rows.shape
    DAT = [RUNS, ini, end, rows, Name, Tsize, Npr ]
    # fig_R, ax_R = plt.subplots(1,1,figsize=fsize)
    cm2in = lambda w,h: (w/2.54,h/2.54)
    fig, ax = plt.subplots(2,3,subplot_kw=dict(polar=True))
    rowi = np.asarray([ [0,0],[0,1],[0,2], [1,0],[1,1],[1,2] ])
    p=1.8; fig.set_size_inches(cm2in(p*12,p*8),forward=True)

    def bound2(A):
        if A < 0:
            A += 2*np.pi
        return A

    T = 0000
    for ROW in ROWLIST:
        for c in range(Npr):
            i, j = rowi[ROW]
            # fi_rel = bound2( fi_rows[T,c,ROW]-np.angle(R[T,ROW]) )
            fi_rel = fi_rows[T,c,ROW]- ( np.angle(R[T,ROW]) )
            ax[i,j].scatter(0,np.absolute(R[T,ROW]), 50, color='xkcd:red')
            ax[i,j].scatter(fi_rel,1, 30, color='xkcd:blue')

        # ax_R.plot(t,np.absolute(R),zorder=-1)
        # ax_R.set_xlabel(r'$Corrida \: [10^6 \: \Delta t ] $')
        # fig_R.tight_layout()
    fig.tight_layout()
    plt.show()
    # ax.plot(fi[0,:],r_0[0,:])

    return [fig,fig_R], [ax, ax_R]

def plot_spectrogram(fs, SETTINGS, nperseg=500):

    from scipy import signal
    X, V, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = SETTINGS #loadargs( sys.argv[2:] )
    T, Npr, rows = fi_rows.shape

    fig, ax = plt.subplots(1,1)
    for chain in range(Npr):
        f, t, Sxx = signal.spectrogram(X[:,chain,0], fs,
                                       # window='boxcar',
                                       nperseg=nperseg )
        try:
            Sxxmean += Sxx
        except:
            Sxxmean = Sxx
    ax.pcolormesh(t, f, np.log(Sxxmean/Npr))
    # ax.pcolormesh(t, f, Sxxmean/Npr)

    return ax

def angle_dist(ROWLIST):

    X, V, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = loadargs( sys.argv[2:] )
    r2 = (R*cj(R)).real
    Rang = np.angle(R) #+ np.pi
    print( Rang.min(), Rang.max())
    print( fi_rows.min(), fi_rows.max())

    Tsize, Npr, rows = fi_rows.shape
    DAT = [RUNS, ini, end, rows, Name, Tsize, Npr ]

    ed2cen = lambda ed:  .5*(ed[1:]+ed[:-1])

    def bound(A):
        for i in range(len(A)):
            if A[i] > np.pi:
                A[i] = - A[i] + 2*np.pi
        return A

    def bound2(A):
        for i in range(len(A)):
            if A[i] < 0:
                A[i] =  A[i] + 2*np.pi
        return A

    def bound3(A):
        for i in range(len(A)):
            if A[i] > np.pi*.5:
                A[i] =  A[i] - np.pi
        return A

    fi_rel = np.empty_like(fi_rows)
    for c in range(Npr): fi_rel[:,c,:] = fi_rows[:,c,:] - Rang

    fig, ax = plt.subplots(1,1)
    for ROW in ROWLIST:
        fi_rel_flat = fi_rel[:,:,ROW].flatten()

        fi_rel_flat_bound = bound(fi_rel_flat)
        # fi_rel_flat_bound = fi_rel_flat

        # ax.plot( t, np.angle(R[:,ROW]), 'k' )

        H, ed = np.histogram( abs(fi_rel_flat_bound), bins=100, density=True)
        bins = ed2cen(ed)
        ax.plot( bins, H, 'o-' )
    plt.show()

def plot_hist(SETTINGS,nbin=100):

    import matplotlib.ticker as mtick

    nbin = 100 if nbin == None else nbin

    xcm, vcm, fi_rows, R, t, imap, rmap = slicer(SETTINGS)
    RUNS, ini, end, rows, Name = SETTINGS
    x_data = xcm.flatten()
    v_data = vcm.flatten()

    Act1 = 4.461*np.cos(80*np.pi/180) # CM
    Act2 = 4.461*np.cos(88*np.pi/180) # CM
    # Act1 = 8.9*np.cos(80*np.pi/180) # EE
    # Act2 = 8.9*np.cos(88*np.pi/180) # EE

    Hx, xed = np.histogram(x_data, bins=nbin)#, density=True)
    Hv, ved = np.histogram(v_data, bins=nbin)#, density=True)
    H, xedges, vedges = np.histogram2d(x_data, v_data, bins=(nbin, nbin))
    """
    ATENCION
    H tiene el shape (xedges,vedges), lo que quiere decir, que la distro en X
    esta en Hx=np.sum(H, axis=1) y en v es Hv=np.sum(H, axis=0)
    """
    print(H.shape, xedges.shape, vedges.shape)
    print(np.sum(H,axis=1).shape)
    X, Y = np.meshgrid(xedges, vedges)
    xcn = ed2cen(xed)
    vcn = ed2cen(ved)
    xcenter = ed2cen(xedges)
    vcenter = ed2cen(vedges)
    index=50
    cmap='BrBG' #'binary'
    fig, ax = plt.subplots(2,3, gridspec_kw = {'width_ratios':[2, 8, 1], 'height_ratios':[4, 1]})
    ax_Hv = ax[0,0]; ax_2d = ax[0,1]; ax10 = ax[1,0]
    ax_Hx = ax[1,1]; ax_bar = ax[0,2]; ax12 = ax[1,2]

    ax[1,0].set_axis_off()
    for axi in ax[:,2]:
        axi.set_axis_off()
    H[0,0]= 3000
    H[-1,-1]= 3000
    ''' Figura central '''
    # plot = ax[0,1].pcolormesh(X, Y, H)
    # plot = ax[0,1].pcolormesh(X, Y, H, cmap=cmap)
    plot = ax[0,1].imshow(np.rot90(H,k=1,axes=(0,1)), extent=[xedges[0], xedges[-1], vedges[0], vedges[-1]], aspect='auto', origin='lower', cmap=cmap)
    ax_2d.vlines(Act1, 0.75,3, linestyle='--',color='xkcd:red') # CM
    ax_2d.vlines(Act2, 0.75,3, linestyle='--',color='xkcd:red') # CM
    ax_2d.axhline(vcn[index], linestyle='--',color='black') # pruebas

    ''' Colorbar '''
    fig.colorbar(plot,ax=ax_bar, fraction=1)

    ''' Histograma en X '''
    ax_Hx.axhline(vcn[index], linestyle='--',color='black') # pruebas
    ax_Hx.axvline(Act1, ls='--',color='xkcd:red')
    ax_Hx.axvline(Act2,ls='--',color='xkcd:red')
    ax_Hx.plot(xcn,Hx, '-o', ms=3, label="x");#ax[0].legend()
    # ax_Hx.plot(xcenter, np.sum(H, axis=1), '-o', ms=3,c='xkcd:green', label="v");#ax[1].legend()

    ''' Histograma en V '''
    ax_Hv.plot(Hv,vcn, '-o', ms=3,c='xkcd:orange', label="v");#ax[1].legend()
    # ax_Hv.plot(np.sum(H, axis=0), vcenter, '-o', ms=3,c='xkcd:green', label="v");#ax[1].legend()

    """ Ticks and misc"""
    # ax[1,1].yaxis.tick_right()
    ax[0,0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.E'))
    ax[1,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.E'))
    ax[0,0].set_ylabel(r'$v\: [\sigma/\tau]$')
    ax[1,1].set_xlabel(r'$x\: [\sigma]$')

    # xlim=[xcn[0],xcn[-1]]; ylim=[vcn[0],vcn[-1]]
    # xlim=[xcenter[0],xcenter[-1]]; ylim=[vcenter[0],vcenter[-1]]
    xlim=[-4,4]; ylim=[-3,3]
    ax[0,1].set_xlim( xlim )
    ax[0,1].set_ylim( ylim )
    ax[1,1].set_xlim( xlim )
    ax[0,0].set_ylim( ylim )

    # plt.savefig(name+'_all.bins_'+str(nbin)+'.png', format='png')
    # plt.clf()
    return fig, ax

def init_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('RUNS', type=int , help='Total number of runs in stack')
    parser.add_argument('INI' , type=float , help='Inital run to process (takes floats)')
    parser.add_argument('END' , type=float, help='Stop at END+1 run (take floats)')
    parser.add_argument('ROWS', type=int, help='Number of ROWS in system (UP+DOWN rows)')
    parser.add_argument('NAME', help='Optional name of file')
    parser.add_argument('--r2',  nargs='+', type=int,  metavar='row' ,default=False, help='Plots r2')
    parser.add_argument('--ph-dyn', action='store_true', default=False, help='')
    parser.add_argument('--str', action='store_true', default=False, help='Gets and writes block data')
    parser.add_argument('--plot-str', nargs='+', type=int, default=False, metavar='row', help='Plots stripes')
    parser.add_argument('--plot-polar',nargs='+', type=int, default=False, metavar='row', help='Plots polar')
    parser.add_argument('--ang-dist', nargs='+', type=int, default=False, metavar='row', help='Plots angular distro')
    parser.add_argument('--peak', action='store_true', default=False, help='Finds peaks')
    parser.add_argument('--x', action='store_true', default=False, help='Plots x')
    parser.add_argument('--xvfi', action='store_true', default=False, help='Plots x, v and phi')
    parser.add_argument('--fft-x', action='store_true', default=False, help='Plots fft-x')
    parser.add_argument('--spectrogram', nargs='?',type=int, default=False, metavar='nperseg', help='Plots spectrogram')
    parser.add_argument('--fft2', nargs='+', type=int, metavar='row', default=False, help='Plots fft2')
    parser.add_argument('--corr', action='store_true', default=False, help='Plots corr')
    parser.add_argument('--hist', nargs='?', type=int, default=False, metavar='BINS', help='Plots 2d hist')
    parser.add_argument('--save', action='store_true', default=False, help='saves plot')
    # print(parser.print_help())

    # parser.add_argument('--hist', action='store_true', default=False, help='Plots 2d hist')
    # parser.add_argument('SETTINGS', nargs=argparse.REMAINDER, help='RUNS INI END ROWS Name')

    # Execute parsing of inputs
    return parser

def main():
    parser = init_parser()
    args = parser.parse_args()
    # print(args)
    # print(args.plot_str)

    try:
        SETTINGS = loadargs([args.RUNS,args.INI,args.END,args.ROWS,args.NAME])
    except ValueError:
        SETTINGS = fake_loadargs()
    RUNS, ini, end, rows, Name = SETTINGS


    if args.r2:
        fig, ax = plot_r2(SETTINGS, args.r2)
        if args.save:
            fig.savefig('r2.png')
            md.add_meta('r2.png', meta)
        plt.show()
    elif args.ph_dyn:
        plot_phase_dyn()
    elif args.str:
        stripes_ROW(range(rows), SETTINGS )
    elif args.plot_str:
        figax, figax_pend = plot_stripes_ROW(args.plot_str, SETTINGS ) #range(rows)
        if args.save:
            figax[0].savefig('stripe.png')
            md.add_meta('stripe.png', meta)
            figax_pend[0].savefig('pend.png')
            md.add_meta('pend.png', meta)
        plt.show()
    elif args.plot_polar:
        figs, axs = plot_polar(args.plot_polar, SETTINGS)
    elif args.ang_dist:
        angle_dist(args.ang_dist)
    elif args.peak:
        plot_peak()
    elif args.x:
        fig, ax = plot_x(SETTINGS)
        plt.show()
    elif args.xvfi:
        fig, ax = plot_xvfi(SETTINGS)
        plt.show()
    elif args.fft_x:
        # sample_rate = 2000 # elastico
        sample_rate = 500 # mecanico/free/long
        ax, dat = plot_fft_x(sample_rate, SETTINGS)
        plt.show()
        if args.save:
            with open('fourier-plot.pkl','wb') as fid: pickle.dump(ax, fid)
            with open('fourier-dat.pkl','wb') as fid: pickle.dump(dat, fid)
    elif args.spectrogram or args.spectrogram == None:
        # sample_rate = 2000 # elastico
        sample_rate = 500 # mecanico/free/long

        ax = plot_spectrogram(sample_rate, SETTINGS, args.spectrogram)
        plt.show()
        if args.save:
            with open('fourier-plot.pkl','wb') as fid: pickle.dump(ax, fid)
            with open('fourier-dat.pkl','wb') as fid: pickle.dump(dat, fid)
    elif args.fft2:
        # sample_rate = 2000 # elastico
        sample_rate = 500 # mecanico/free/long
        # fig, ax = plot_fft2(sample_rate, range(6))
        fig, ax = plot_fft2(sample_rate, args.fft2, SETTINGS)
        plt.show()
    elif args.corr:
        ax, dat = corr(range(rows), SETTINGS) #range(rows)
        if args.save:
            with open('corr-dat-'+str(ini)+'-'+str(end)+'.pkl','wb') as fid: pickle.dump(dat, fid)
            with open('corr-plot-'+str(ini)+'-'+str(end)+'.pkl','wb') as fid: pickle.dump(ax, fid)
        else:
            plt.show()
    elif args.hist or args.hist == None:
        fig, ax = plot_hist(SETTINGS,args.hist)
        plt.show()

if __name__ == "__main__":
    main()
