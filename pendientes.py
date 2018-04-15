#!/usr/bin/env python3
import numpy as np
import scipy as sc
import scipy.signal as sg
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import glob, os

RUN_DIR = os.getcwd().split('/')[-1]

if "graft" in RUN_DIR:
    title = r'$\rho = '+ RUN_DIR.split('_')[-1] + '$'
elif "k" in RUN_DIR:
    title = r'$ k = '+ RUN_DIR.split('k')[-1] + '$'
else:
    title = ''


files = glob.glob('block_*.npz')
get_row = lambda d: int(d.split('_')[1])
files = sorted(files, key = get_row)

figlist = []
axlist = []
pendientes_all = []
expo = 3
cm2in = lambda w,h: (w/2.54,h/2.54)
ed2cen = lambda ed: .5*(ed[1:]+ed[:-1])
def sci_notation(x, pos):
        return "${:.1f}$".format(x * 10**int(expo))
MyFormatter = FuncFormatter(sci_notation)
# plt.rc('text', usetex=True)
fig, ax = plt.subplots(1,1)
p=0.8; fig.set_size_inches(cm2in(p*12,p*12),forward=True)


fig.suptitle(title)
for i, block_i in enumerate(files):

    block = np.load(block_i)

    pendientes = block["pendientes"]
    tiempos = block["tiempos"]
    DAT = block["DAT"]
    centers = block["centers"]
    r2 = block["r2"][:,i].real
    ini = int(r2.size*.05)
    r2mean = np.mean(r2[ini:])

    pendientes_all.extend(pendientes)

    H, ed = np.histogram(pendientes, bins=20)
    bincen = ed2cen(ed)
    # ax.plot(bincen, H, label=r"Fila$={0}$, $\left\langle r^{{2}}\right\rangle={1:.2f}$".format(i,r2mean), lw=1.5)
    ax.plot(bincen, H, label=r"$\left\langle r^{{2}}\right\rangle_{1}={0:.2f}$".format(r2mean,i), lw=1.5)


H, ed = np.histogram(pendientes_all, bins=40)
bincen = ed2cen(ed)
ax.plot(bincen, H, label='Todas', color='k', lw=2)
ax.set_xlabel(r'$a/v\;[10^6\Delta t]$', fontsize=11)
ax.set_ylabel(r'Nº', fontsize=11)
fig.tight_layout()
# fig.subplots_adjust(
#     top=0.88,
#     bottom=0.12,
#     left=0.12,
#     right=0.9,
#     hspace=0.2,
#     wspace=0.2)

ax.xaxis.set_major_formatter(MyFormatter)
ax.text(0.92, -0.12, "$\\times 10^{-"+str(expo)+"}$", fontsize=9, transform = plt.gca().transAxes)
ax.legend(fontsize=9)
# plt.savefig("pendientes_"+RUN_DIR+".svg")
plt.show()


