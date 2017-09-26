#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import pandas as p
mpl.use('Agg')
from matplotlib import pyplot as plt

def plot_transf(n):
    """n: 
            0   Energia
            1   Cinetica
            2   Potencial
        """

    leyenda = ['Total','Cinética','Potencial']

    rawdat=p.read_table('transformadas.dat', header=None, delim_whitespace=True)#, dtype=np.float32)
    array=np.asarray(rawdat)
    freq = array[:,0]
    espectro = array[:,3*n+1:3*n+4]
    labeled = ['Real','Imaginaria','Absoluto']
    for i in range(3):
        plt.plot(freq, espectro[:,i], label = labeled[i])
    plt.legend()

    plt.xlabel('Freq')
    plt.ylabel('')
    plt.savefig('./transformada_E_'+ leyenda[n] +'.png')
    plt.cla()

plot_transf(0)
plot_transf(1)
plot_transf(2)
