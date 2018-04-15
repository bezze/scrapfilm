#!/usr/bin/env python3

"""
Plots energies spectrum density.

Takes transformada.dat which is a file with 1+3 columns of spectral data, the
first for frequency and the rest for total energy, cinetic and potential.
"""

import numpy as np
import matplotlib as mpl
import pandas as p
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
from matplotlib import pyplot as plt

def plot_transf(n):
    """n:
            0   Energia
            1   Cinetica
            2   Potencial
        """

    leyenda = ['Total','Cinetica','Potencial']

    rawdat=p.read_table('transformadas.dat', header=None, delim_whitespace=True)#, dtype=np.float32)
    array=np.asarray(rawdat)
    ini=0; end=11184811
    freq = array[ini:end,0]
    espectro = array[ini:end,3*n+1:3*n+4]
    labeled = ['Real','Imaginaria','Absoluto']
    fig, ax = plt.subplots(1,1)
    for i in range(3):
        ax.plot(freq[ini:end], espectro[ini:end,i], label = labeled[i])
    ax.set_ylim([-20, 10000])
    #ax.set_xlim([0, 10000])
    plt.legend()

    plt.xlabel('Freq')
    plt.ylabel('')
    plt.savefig('./transformada_E_'+ leyenda[n] +'.png')
    plt.cla()

plot_transf(0)
plot_transf(1)
plot_transf(2)
