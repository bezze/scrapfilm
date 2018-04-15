#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import conjugate as cj
import sys, argparse, pickle

padlength = lambda size: int(2**np.ceil(np.log2(size))-size)
Mag = lambda f: (f*cj(f)).real
cm2in = lambda w,h: (w/2.54,h/2.54)
get_par = lambda d: "{0:.2f}".format(float(d.split('=')[1].strip('$')))

def plot_par(sim,par,ROWS):
    par = int(par)
    datos = np.load('corr_err_rows_'+sim+'.npz')
    corr, cerr, labels = datos['CORR'],datos['CERR'],datos['labels']
    ch=np.linspace(0,10,11); N=ch.shape[0]
    if sim == 'mec' : sim_label = r'$\rho ='; sim_unit = r'\;1/\sigma$'
    if sim == 'ela' : sim_label = r'$ k ='; sim_unit = r'\;\epsilon/\sigma^2$'

    fig, ax = plt.subplots(1,1)

    # colormap = plt.cm.nipy_spectral #I suggest to use nipy_spectral, Set1,Paired
    # ax.set_color_cycle([colormap(i) for i in np.linspace(0., 0.95,4 )])
    for ROW in ROWS:
    # for d, label in enumerate( labels ):
    #     label = labels[n]
    #     corr = C_ce[n,:,0]
    #     err = C_ce[n,:,1]
        ax.errorbar(ch[1:], corr[par,ROW,1:], yerr=cerr[par,ROW,1:],
                    fmt='-o', ms=3, capsize=5, lw=1.,
                    label=str(ROW))

    # ax.set_title( sim_label + get_par(labels[par]) +r'$' )
    ax.set_title( sim_label + get_par(labels[par]) + sim_unit )
    ax.set_xlabel(r'$Vecino$')
    ax.set_ylabel(r'$Correlación$')
    ax.set_ylim([-.4,.75])
    plt.legend(ncol=2)
    p=1.0; fig.set_size_inches(cm2in(p*9,p*9),forward=True)
    fig.tight_layout()

    return fig, ax

def plot_separados(sim):
    datos = np.load('corr_err_'+sim+'.npz')
    C_ce, labels = datos['C'],datos['labels']
    ch=np.linspace(0,10,11); N=ch.shape[0]
    colormap = plt.cm.nipy_spectral #I suggest to use nipy_spectral, Set1,Paired

    reformat_label = lambda l: r"$\rho = {:.2f}$".format(float(l.split('=')[1].strip('$')))

    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)

    for ax in ax1,ax2:
        ax.set_color_cycle([colormap(i) for i in np.linspace(0., 0.95,int(N) )])
        # ax.set_title(r'$Correlación\ espacial$')
        ax.set_xlabel(r'$Vecino$')

    for n in range(0,10):
        label = reformat_label(labels[n])
        corr = C_ce[n,:,0]
        err = C_ce[n,:,1]
        ax1.errorbar(ch, corr, yerr=err, fmt='-o', capsize=7, label=label)
    fig1.legend(ncol=2,bbox_to_anchor=(0.8,0.8))

    for n in range(11,len(labels)-4):
        label = reformat_label(labels[n])
        corr = C_ce[n,:,0]
        err = C_ce[n,:,1]
        ax2.errorbar(ch, corr, yerr=err, fmt='-o', capsize=7, label=label)
    fig2.legend(ncol=2,bbox_to_anchor=(0.8,0.8))

    fig1.tight_layout()
    fig2.tight_layout()

    return [fig1, fig2], [ax1, ax2]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('sim', help='Either mec or ela')
    parser.add_argument('--Npar', default=0, help='Plots parameter N\'s rows only')
    parser.add_argument('--row', default=False, help='Plots only row')
    parser.add_argument('--save', action='store_true', default=False, help='Saves plot in svg')
    parser.add_argument('--plot', action='store_true', default=False, help='Shows plot')
    parser.add_argument('--load', default=False, help='Loads pickle plot')

    # Execute parsing of inputs
    args = parser.parse_args()
    if args.row:
        ROWS = [ int(args.row) ]
    else:
        ROWS = range(6)

    fig, ax = plot_par(args.sim, args.Npar, ROWS)
    if args.save:
        plt.savefig('corr_'+args.sim+'_par_'+args.Npar+'.svg')
    if args.plot:
        plt.show()
    # if args.fourier:
    #     ax = cycle_through_rows(range(6))
    #     if args.save:
    #         with open('myplot.pkl','wb') as fid: pickle.dump(ax, fid)
    #     plt.show()

    if args.load:
        with open(args.load, 'rb') as fid:
                ax = pickle.load(fid)
        plt.show()

if __name__ == "__main__":
    main()
