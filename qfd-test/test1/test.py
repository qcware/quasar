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

# => PRL Style Plots <= #

eVperH = 27.2114

def lorentzian(
    x,
    x0,
    delta,
    ):

    return 0.5 * delta / np.pi * 1.0 / ((x - x0)**2 + (0.5 * delta)**2)

def make_plot(
    aiem_solver,
    outfile,
    cutoff=1.0E-7,
    Emin=0.0,
    Emax=2.0,
    Omax=2.0,
    delta=0.05,
    scale=0.1,
    ):

    keys = ['fci', 'cis', 'qfd-1', 'qfd-2', 'qfd-3']
    Es = {
        'fci' : aiem_solver.fci_E,
        'cis' : aiem_solver.cis_E,
        }
    Os = {
        'fci' : aiem_solver.fci_O,
        'cis' : aiem_solver.cis_O,
        }
    Es['qfd-1'], Cs, Os['qfd-1'], ss = aiem_solver.qfd_diagonalize(kmax=1, nref=aiem_solver.nstate, cutoff=cutoff)
    Es['qfd-2'], Cs, Os['qfd-2'], ss = aiem_solver.qfd_diagonalize(kmax=2, nref=aiem_solver.nstate, cutoff=cutoff)
    Es['qfd-3'], Cs, Os['qfd-3'], ss = aiem_solver.qfd_diagonalize(kmax=3, nref=aiem_solver.nstate, cutoff=cutoff)

    Es = { key : value[:Es['fci'].shape[0]] for key, value in Es.items() }
    Os = { key : value[:Os['fci'].shape[0]] for key, value in Os.items() }

    dEs = { key : eVperH * (E[1:] - E[0]) for key, E in Es.items() }

    # Colors
    cmap = matplotlib.cm.get_cmap('plasma')
    colors = [cmap(float(x) / (1.55 * (len(keys) - 1))) for x in reversed(list(range(len(keys))))]
    colors = list(reversed(colors))
    colors = { k : v for k, v in zip(keys, colors) }

    symbols = {
        'fci' : '-s',
        'cis' : '-o',
        'qfd-1' : '-v',
        'qfd-2' : '-^',
        'qfd-3' : '-p',
        }

    labels = {
        'fci' : 'FCI (Classically Exponential)',
        'cis' : 'CIS (Classically Polynomial)',
        'qfd-1' : 'QFD-1 (This Work)',
        'qfd-2' : 'QFD-2 (This Work)',
        'qfd-3' : 'QFD-3 (This Work)',
        }

    plt.clf()

    plt.figure(figsize=(6,4))

    f, (a0, a1, a2) = plt.subplots(3,1, gridspec_kw = {'height_ratios' : [3.5, 1, 1]}, figsize=(6,8))
    
    dE = np.linspace(Emin, Emax, 2000)
    for key in keys:
        for I in range(len(dEs[key])):
            a0.plot([dEs[key][I]]*2, [-0.5, Os[key][I]], 
                symbols[key], 
                color=colors[key], 
                label=labels[key] if I==0 else None,
                linewidth=2.0,
                markersize=8.0,
                )
        O = np.zeros_like(dE)
        for I in range(len(dEs[key])):
            O += Os[key][I] * lorentzian(dE, dEs[key][I], delta=delta)
        O *= scale
        a0.plot(dE, O, color=colors[key], linewidth=3.0 if key=='fci' else 1.0)
    # a0.xlabel(r'$\Delta E$ [eV]')
    # a0.ylabel('Oscillator Strength [-]')
    a0.legend(loc=2)
    a0.axis([Emin, Emax, 0.0, Omax])

    for I in range(len(dEs[key])):
        for key in ['qfd-3', 'qfd-2', 'qfd-1', 'cis']:    
            a1.semilogy([dEs[key][I]]*2, [1.0E+2, 1.0E-7],
                '-',
                color=colors[key],
                linewidth=0.5,
                )
    for I in range(len(dEs[key])):
        a1.semilogy(
            [dEs['qfd-1'][I], dEs['cis'][I]],
            [np.abs(dEs['qfd-1'][I] - dEs['fci'][I]),  np.abs(dEs['cis'][I] - dEs['fci'][I])],
            '-k',
            linewidth=0.5,
            )
    for I in range(len(dEs[key])):
        for key in ['qfd-1', 'cis']:    
            a1.semilogy(dEs[key][I], np.abs(dEs[key][I] - dEs['fci'][I]),
                symbols[key],
                color=colors[key],
                linewidth=2.0,
                markersize=8.0,
                )
    for I in range(len(dEs[key])):
        a1.semilogy(
            [dEs['qfd-2'][I], dEs['qfd-1'][I]],
            [np.abs(dEs['qfd-2'][I] - dEs['fci'][I]),  np.abs(dEs['qfd-1'][I] - dEs['fci'][I])],
            '-k',
            linewidth=0.5,
            )
    for I in range(len(dEs[key])):
        for key in ['qfd-2', 'qfd-1']:    
            a1.semilogy(dEs[key][I], np.abs(dEs[key][I] - dEs['fci'][I]),
                symbols[key],
                color=colors[key],
                linewidth=2.0,
                markersize=8.0,
                )
    for I in range(len(dEs[key])):
        a1.semilogy(
            [dEs['qfd-3'][I], dEs['qfd-2'][I]],
            [np.abs(dEs['qfd-3'][I] - dEs['fci'][I]),  np.abs(dEs['qfd-2'][I] - dEs['fci'][I])],
            '-k',
            linewidth=0.5,
            )
    for I in range(len(dEs[key])):
        for key in ['qfd-3', 'qfd-2']:    
            a1.semilogy(dEs[key][I], np.abs(dEs[key][I] - dEs['fci'][I]),
                symbols[key],
                color=colors[key],
                linewidth=2.0,
                markersize=8.0,
                )
    a1.axis([Emin, Emax, 1.0E-4, 2.0E+0])

    for I in range(len(dEs[key])):
        for key in ['qfd-3', 'qfd-2', 'qfd-1', 'cis']:    
            a2.semilogy([dEs[key][I]]*2, [1.0E+2, 1.0E-6],
                '-',
                color=colors[key],
                linewidth=0.5,
                )
    for I in range(len(dEs[key])):
        a2.semilogy(
            [dEs['qfd-1'][I], dEs['cis'][I]],
            [np.abs(Os['qfd-1'][I] - Os['fci'][I]),  np.abs(Os['cis'][I] - Os['fci'][I])],
            '-k',
            linewidth=0.5,
            )
    for I in range(len(dEs[key])):
        for key in ['qfd-1', 'cis']:    
            a2.semilogy(dEs[key][I], np.abs(Os[key][I] - Os['fci'][I]),
                symbols[key],
                color=colors[key],
                linewidth=2.0,
                markersize=8.0,
                )
    for I in range(len(dEs[key])):
        a2.semilogy(
            [dEs['qfd-2'][I], dEs['qfd-1'][I]],
            [np.abs(Os['qfd-2'][I] - Os['fci'][I]),  np.abs(Os['qfd-1'][I] - Os['fci'][I])],
            '-k',
            linewidth=0.5,
            )
    for I in range(len(dEs[key])):
        for key in ['qfd-2', 'qfd-1']:    
            a2.semilogy(dEs[key][I], np.abs(Os[key][I] - Os['fci'][I]),
                symbols[key],
                color=colors[key],
                linewidth=2.0,
                markersize=8.0,
                )
    for I in range(len(dEs[key])):
        a2.semilogy(
            [dEs['qfd-3'][I], dEs['qfd-2'][I]],
            [np.abs(Os['qfd-3'][I] - Os['fci'][I]),  np.abs(Os['qfd-2'][I] - Os['fci'][I])],
            '-k',
            linewidth=0.5,
            )
    for I in range(len(dEs[key])):
        for key in ['qfd-3', 'qfd-2']:    
            a2.semilogy(dEs[key][I], np.abs(Os[key][I] - Os['fci'][I]),
                symbols[key],
                color=colors[key],
                linewidth=2.0,
                markersize=8.0,
                )
    a2.axis([Emin, Emax, 1.0E-5, 0.9E+1])


    a0.set_ylabel(r'Osc. Str. [-]')
    a1.set_ylabel(r'Err. $\Delta E$ [eV]')
    a2.set_ylabel(r'Err. Osc. Str. [-]')
    a2.set_xlabel(r'$\Delta E$ [eV]')
    a0.set_xticklabels([])
    a1.set_xticklabels([])

    plt.subplots_adjust(hspace=0.02)
    # f.tight_layout()
    f.savefig(outfile, bbox_inches='tight')

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
        qfd_kmax=1,
        qfd_cutoff=1.0E-7,
        # qfd_kappa_method='explicit',
        # qfd_kappa_explicit=4.3,
        qfd_matrix_method='exact',
        # qfd_matrix_method='trotter_toeplitz',
        # qfd_matrix_method='exact',
        # qfd_matrix_method='trotter',
        qfd_trotters_per_k=16000,
        )
    aiem_solver.compute_energy()

    aiem_solver2 = qfd.QFD.from_options(
        backend=backend,
        nstate=nstate,
        aiem_monomer=aiem_monomer,
        qfd_kmax=1,
        qfd_cutoff=1.0E-7,
        # qfd_kappa_method='explicit',
        # qfd_kappa_explicit=4.3,
        # qfd_matrix_method='exact_toeplitz',
        qfd_matrix_method='trotter',
        # qfd_matrix_method='exact',
        # qfd_matrix_method='trotter',
        qfd_trotters_per_k=1,
        )
    aiem_solver2.compute_energy()

    print(np.max(np.abs(aiem_solver.qfd_H - aiem_solver2.qfd_H)))
    print(np.max(np.abs(aiem_solver.qfd_S - aiem_solver2.qfd_S)))

    # > Gorshgorin Disk Plots < #

    # plot_gorshgorin_disks(
    #     filename='disks-%d.pdf' % N,
    #     aiem_solver=aiem_solver,
    #     )

    # > Spectral Plots < #

    make_plot(
        aiem_solver=aiem_solver,
        cutoff=1.0E-7,
        outfile='data-8-9.pdf',
        Emin=1.3,
        Emax=3.7,
        Omax=4.15,
        delta=0.15,
        scale=0.280,
        )
