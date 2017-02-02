#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import sys 
from matplotlib import pyplot as plt
from scrapfilm import analyze, boundary

archivos=sys.argv[1:]

print(archivos)
binning=20
for c in range(30):
    rg=analyze(c,0,archivos[0])
    rb=analyze(c,1,archivos[0])
    r0=rb-rg
    hist, bins=np.histogram(r0[:,0], bins=binning)
    centers = np.linspace(bins[0],bins[-1],binning)
    plt.plot(centers,hist,'o')
plt.show()
