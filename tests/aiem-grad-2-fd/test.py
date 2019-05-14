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
        monomer,    
        ) for monomer in range(1,N+1)]

    filenamesp = ['%s/%d-%d-%s-p%s/exciton.dat' % (
        datapath,
        monomer,    
        atom,
        coordinate,
        delta,
        ) for monomer in range(1, N+1)]

    filenamesm = ['%s/%d-%d-%s-m%s/exciton.dat' % (
        datapath,
        monomer,    
        atom,
        coordinate,
        delta,
        ) for monomer in range(1, N+1)]

    # Positive displacement    
    aiem_monomerp = quasar.AIEMMonomer.from_tc_exciton_files(
        filenames=[filenamesp[A] if A+1 == monomer else filenames0[A] for A in range(N)],
        N=N,
        connectivity=connectivity,
        # zero_gauge=True,
        )

    aiemp = quasar.AIEM(aiem.options.copy().set_values({
        'aiem_monomer' : aiem_monomerp,
        'aiem_monomer_grad' : None,
        'print_level' : 0,
        })) 
    aiemp.compute_energy()

    # Negative displacement    
    aiem_monomerm = quasar.AIEMMonomer.from_tc_exciton_files(
        filenames=[filenamesm[A] if A+1 == monomer else filenames0[A] for A in range(N)],
        N=N,
        connectivity=connectivity,
        # zero_gauge=True,
        )

    aiemm = quasar.AIEM(aiem.options.copy().set_values({
        'aiem_monomer' : aiem_monomerm,
        'aiem_monomer_grad' : None,
        'print_level' : 0,
        })) 
    aiemm.compute_energy()

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

    print(G_fci)
    print(G_fci_fd)
    print(G_vqe)
    print(G_vqe_fd)
    print(G_cis)
    print(G_cis_fd)

if __name__ == '__main__':

    backend = quasar.QuasarSimulatorBackend()

    datapath = '../../data/aiem/bchl-a-2-stack-fd/tc'
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

    include_vqe_response = True
    include_cis_response = True

    verify_grad_fd(
        aiem=aiem,
        connectivity=connectivity,
        datapath=datapath,
        monomer=1,
        atom=20,
        coordinate='x',
        delta=0.002,
        include_vqe_response=include_vqe_response,
        include_cis_response=include_cis_response,
        )
