import numpy as np
import quasar

if __name__ == '__main__':

    backend = quasar.QuasarSimulatorBackend()

    datapath = '../../data/aiem/bchl-a-8-stack/tc'
    filenames = ['%s/%d/exciton.dat' % (datapath, _) for _ in range(1, 8+1)]
    N = 4
    nstate = 3
    # connectivity = 'linear'
    connectivity = 'cyclic'
    cis_circuit_type = 'mark2'
    vqe_circuit_type = 'mark1'

    aiem_monomer = quasar.AIEMMonomer.from_tc_exciton_files(
        filenames=filenames,
        N=N,
        connectivity=connectivity,
        )

    aiem_monomer_grad = quasar.AIEMMonomerGrad.from_tc_exciton_files(
        filenames=filenames,
        N=N,
        connectivity=connectivity,
        )

    optimizer = quasar.BFGSOptimizer.from_options(
        g_convergence=1.0E-4,
        maxiter=0,
        )

    aiem = quasar.AIEM.from_options(
        optimizer=optimizer,
        backend=backend,
        nstate=nstate,
        aiem_monomer=aiem_monomer,
        aiem_monomer_grad=aiem_monomer_grad,
        cis_circuit_type=cis_circuit_type,
        vqe_circuit_type=vqe_circuit_type,
        )
    aiem.compute_energy()
    
    print(aiem.compute_fci_gradient(I=0))
    print(aiem.compute_fci_coupling(I=0, J=1))
    print(aiem.compute_cis_gradient(I=0))
    print(aiem.compute_cis_coupling(I=0, J=1))
