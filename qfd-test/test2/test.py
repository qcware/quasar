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

if __name__ == '__main__':

    backend = quasar.QuasarSimulatorBackend()

    datapath = '../../data/aiem/bchl-a-8-stack/tc'
    filenames = ['%s/%d/exciton.dat' % (datapath, _) for _ in range(1, 8+1)]
    N = 8
    nstate = 9
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
        qfd_kmax=2,
        qfd_cutoff=1.0E-2,
        # qfd_kappa_method='explicit',
        # qfd_kappa_explicit=0.1,
        # qfd_kappa_explicit=4.3,
        # qfd_kappa_explicit=1.0,
        )
    aiem_solver.compute_energy()

    # basis = aiem_solver.qfd_subspace_basis(nref=9, kmax=3)
    # evecs = np.einsum('Ivk,vkT->IT', basis, aiem_solver.qfd_C)
