import numpy as np
import quasar

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
        zero_gauge=True,
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

    G_fci, G_vqe, G_cis, G_fci_fd, G_vqe_fd, G_cis_fd = quasar.AIEMGradCheck.test_fd_gradient_monomer(aiem=aiem, include_vqe_response=include_vqe_response, include_cis_response=include_cis_response)

    np.savez(npzfile,
        G_fci=G_fci,
        G_vqe=G_vqe,
        G_cis=G_cis,
        G_fci_fd=G_fci_fd,
        G_vqe_fd=G_vqe_fd,
        G_cis_fd=G_cis_fd,
        )
