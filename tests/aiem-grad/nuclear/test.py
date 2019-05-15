import numpy as np
import quasar

def verify_grad_fd(
    aiem,
    datapath,
    connectivity,
    monomer,
    atom,
    coordinate,
    delta,
    include_vqe_response=True,
    include_cis_response=True,
    ):

    N = aiem.N
    d = { 'x' : 0, 'y' : 1, 'z' : 2 }[coordinate]

    filenames0 = ['%s/%d/exciton.dat' % (
        datapath,
        A,    
        ) for A in range(1,N+1)]

    filenamesp = ['%s/%d-%d-%s-p%s/exciton.dat' % (
        datapath,
        A,    
        atom,
        coordinate,
        delta,
        ) for A in range(1, N+1)]

    filenamesm = ['%s/%d-%d-%s-m%s/exciton.dat' % (
        datapath,
        A,    
        atom,
        coordinate,
        delta,
        ) for A in range(1, N+1)]

    # Positive displacement    
    aiem_monomerp = quasar.AIEMMonomer.from_tc_exciton_files(
        filenames=[filenamesp[A] if A == monomer else filenames0[A] for A in range(N)],
        N=N,
        connectivity=connectivity,
        # zero_gauge=True,
        )

    aiemp = quasar.AIEM(aiem.options.copy().set_values({
        'aiem_monomer' : aiem_monomerp,
        'aiem_monomer_grad' : None,
        'print_level' : 0,
        })) 
    aiemp.compute_energy(
        param_values_ref=aiem.vqe_circuit.param_values,
        cis_C_ref=aiem.cis_C,
        vqe_C_ref=aiem.vqe_C,
        )

    # Negative displacement    
    aiem_monomerm = quasar.AIEMMonomer.from_tc_exciton_files(
        filenames=[filenamesm[A] if A == monomer else filenames0[A] for A in range(N)],
        N=N,
        connectivity=connectivity,
        # zero_gauge=True,
        )

    aiemm = quasar.AIEM(aiem.options.copy().set_values({
        'aiem_monomer' : aiem_monomerm,
        'aiem_monomer_grad' : None,
        'print_level' : 0,
        })) 
    aiemm.compute_energy(
        param_values_ref=aiem.vqe_circuit.param_values,
        cis_C_ref=aiem.cis_C,
        vqe_C_ref=aiem.vqe_C,
        )

    # Finite difference
    delta_au = delta / 0.52917724924 # TODO
    G_fci_fd = np.array([(aiemp.fci_tot_E[I] - aiemm.fci_tot_E[I]) / (2 * delta_au) for I in range(aiem.nstate)])
    G_vqe_fd = np.array([(aiemp.vqe_tot_E[I] - aiemm.vqe_tot_E[I]) / (2 * delta_au) for I in range(aiem.nstate)])
    G_cis_fd = np.array([(aiemp.cis_tot_E[I] - aiemm.cis_tot_E[I]) / (2 * delta_au) for I in range(aiem.nstate)])

    # Analytical
    G_fci = np.zeros((aiem.nstate,))
    G_vqe = np.zeros((aiem.nstate,))
    G_cis = np.zeros((aiem.nstate,))
    for I in range(aiem.nstate):
        G_fci[I] = aiem.compute_fci_gradient(I=I)[monomer][atom][d]
        G_cis[I] = aiem.compute_cis_gradient(I=I)[monomer][atom][d]
        G_vqe[I] = aiem.compute_vqe_gradient(
            I=I,
            include_vqe_response=include_vqe_response,
            include_cis_response=include_cis_response,
            )[monomer][atom][d]

    print('==> Nuclear Gradient Check <==\n')

    print('Analytical vs. Finite Difference:\n')
    print('%s = %d' % ('monomer', monomer))
    print('%s = %d' % ('atom', atom))
    print('%s = %s' % ('coordinate', coordinate))
    print('%s = %s' % ('delta', delta))
    print('')
    for label, G, G_fd in zip(['FCI', 'VQE', 'CIS'], [G_fci, G_vqe, G_cis], [G_fci_fd, G_vqe_fd, G_cis_fd]):
    
        print('Method: %s\n' % label)
        print('%2s: %11s' % (
            'I', 'E'))
        for I in range(len(G)):
            G1 = G[I]
            G2 = G_fd[I]
            print('%2d: %11.3E' % (
                I,
                np.max(np.abs(G1 - G2)),
                ))
        print('')

    print('Method to Method (Analytical):\n')
    for label, Gs1, Gs2 in zip(['FCI-VQE', 'FCI-CIS', 'VQE-CIS'], [G_fci, G_fci, G_vqe], [G_vqe, G_cis, G_cis]):
    
        print('Method: %s\n' % label)
        print('%2s: %11s' % (
            'I', 'E'))
        for I in range(len(G)):
            G1 = Gs1[I]
            G2 = Gs2[I]
            print('%2d: %11.3E' % (
                I,
                np.max(np.abs(G1 - G2)),
                ))
        print('')

    print('==> End Nuclear Gradient Check <==\n')

    return G_fci, G_vqe, G_cis, G_fci_fd, G_vqe_fd, G_cis_fd

if __name__ == '__main__':

    import sys
    include_vqe_response = True if sys.argv[1] == 'True' else False
    include_cis_response = True if sys.argv[2] == 'True' else False
    npzfile = sys.argv[3]

    backend = quasar.QuasarSimulatorBackend()

    datapath = '../../../data/aiem/bchl-a-2-stack-fd/tc'
    filenames = ['%s/%d/exciton.dat' % (datapath, _) for _ in range(1, 2+1)]
    N = 2
    nstate = 2
    connectivity = 'linear'

    aiem_monomer = quasar.AIEMMonomer.from_tc_exciton_files(
        filenames=filenames,
        N=N,
        connectivity=connectivity,
        # zero_gauge=True,
        )

    aiem_monomer_grad = quasar.AIEMMonomerGrad.from_tc_exciton_files(
        filenames=filenames,
        N=N,
        connectivity=connectivity,
        )

    optimizer = quasar.JacobiOptimizer.from_options(
        g_convergence=1.0E-16,
        jacobi_level=2,
        )

    vqe_circuit = quasar.Circuit(N=N)
    for A in range(N):
        vqe_circuit.add_gate(T=0, key=A, gate=quasar.Gate.Ry(theta=0.0))

    aiem = quasar.AIEM.from_options(
        optimizer=optimizer,
        backend=backend,
        nstate=nstate,
        aiem_monomer=aiem_monomer,
        aiem_monomer_grad=aiem_monomer_grad,
        vqe_circuit=vqe_circuit,
        )
    aiem.compute_energy()

    # Mg O, N, C, H
    monomers = [0, 0, 0, 1, 1]
    atoms = [20, 41, 16, 22, 31]
    
    G_fci = []
    G_vqe = []
    G_cis = []
    G_fci_fd = []
    G_vqe_fd = []
    G_cis_fd = []

    for monomer, atom in zip(monomers, atoms):
        for coordinate in ['x', 'y', 'z']:

            G_fci2, G_vqe2, G_cis2, G_fci_fd2, G_vqe_fd2, G_cis_fd2 = verify_grad_fd(
                aiem=aiem,
                connectivity=connectivity,
                datapath=datapath,
                monomer=monomer,
                atom=atom,
                coordinate=coordinate,
                delta=0.002,
                include_vqe_response=include_vqe_response,
                include_cis_response=include_cis_response,
                )
            G_fci.append(G_fci2)
            G_vqe.append(G_vqe2)
            G_cis.append(G_cis2)
            G_fci_fd.append(G_fci_fd2)
            G_vqe_fd.append(G_vqe_fd2)
            G_cis_fd.append(G_cis_fd2)

    G_fci = np.array(G_fci)
    G_vqe = np.array(G_vqe)
    G_cis = np.array(G_cis)
    G_fci_fd = np.array(G_fci_fd)
    G_vqe_fd = np.array(G_vqe_fd)
    G_cis_fd = np.array(G_cis_fd)

    np.savez(npzfile,
        G_fci=G_fci,
        G_vqe=G_vqe,
        G_cis=G_cis,
        G_fci_fd=G_fci_fd,
        G_vqe_fd=G_vqe_fd,
        G_cis_fd=G_cis_fd,
        )
