import numpy as np
import quasar
import qfd

import matplotlib.pyplot as plt

import matplotlib.style
matplotlib.style.use('classic')
    
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

def plot_gorshgorin_disks(
    filename,
    aiem_solver,
    ):

    evals = aiem_solver.hamiltonian_evals
    diags = aiem_solver.hamiltonian_diags
    disks = aiem_solver.hamiltonian_gorshgorin_disks

    plt.clf()
    plt.figure(figsize=(6,8))
    plt.subplot(aspect='equal')
    for diag, disk in zip(diags, disks): 
        circle = plt.Circle((diag, 0.0), disk, fill=True, facecolor='skyblue', edgecolor='k')
        plt.gcf().gca().add_artist(circle)
    circle = plt.Circle((diags[0], 0.0), disks[0], facecolor='r', fill=True, edgecolor='k')
    plt.gcf().gca().add_artist(circle)
    circle = plt.Circle((diags[-1], 0.0), disks[-1], facecolor='r', fill=True, edgecolor='k')
    plt.gcf().gca().add_artist(circle)
    plt.plot(evals, np.zeros_like(evals), 'kx', markeredgewidth=1.2, markersize=7)
    dd = 0.02
    plt.axis([min(diags) - max(disks) - dd, max(diags) + max(disks) + dd, -max(disks) - dd, +max(disks) +dd])
    plt.xlabel('Energy [a.u.]')
    plt.ylabel('Gorshgorin Disk Radius [a.u.]')
    plt.savefig(filename, bbox_inches='tight')
    

if __name__ == '__main__':

    backend = quasar.QuasarSimulatorBackend()

    datapath = '../../data/aiem/bchl-a-8-stack/tc'
    filenames = ['%s/%d/exciton.dat' % (datapath, _) for _ in range(1, 8+1)]
    N = 8
    nstate = N+1
    connectivity = 'linear'

    aiem_monomer = qfd.AIEMMonomer.from_tc_exciton_files(
        filenames=filenames,
        N=N,
        connectivity=connectivity,
        )

    aiem_solver = qfd.QFD.from_options(
        backend=backend,
        nstate=nstate,
        aiem_monomer=aiem_monomer,
        )
    aiem_solver.compute_energy()

    plot_gorshgorin_disks(
        filename='disks-%d.pdf' % N,
        aiem_solver=aiem_solver,
        )
