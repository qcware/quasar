import numpy as np
import quasar

if __name__ == '__main__':

    backend = quasar.QuasarSimulatorBackend()

    datapath = '../../data/aiem/bchl-a-8-stack/tc'
    filenames = ['%s/%d/exciton.dat' % (datapath, _) for _ in range(1, 8+1)]
    N = 2
    nstate = 2
    # connectivity = 'linear'
    connectivity = 'cyclic'
    vqe_circuit_type = 'mark2x'

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

    optimizer = quasar.BFGSOptimizer.from_options(
        g_convergence=1.0E-18,
        maxiter=100,
        )

    # optimizer = quasar.PowellOptimizer.from_options(
    #     ftol=1.0E-8,
    #     maxiter=100,
    #     )

    vqe_circuit = quasar.Circuit(N=N)
    for A in range(N):
        vqe_circuit.add_gate(T=0, key=A, gate=quasar.Gate.Ry(theta=0.0))

    aiem = quasar.AIEM.from_options(
        optimizer=optimizer,
        backend=backend,
        nstate=nstate,
        aiem_monomer=aiem_monomer,
        aiem_monomer_grad=aiem_monomer_grad,
        # vqe_circuit_type=vqe_circuit_type,
        vqe_circuit=vqe_circuit,
        )
    aiem.compute_energy()

    include_vqe_response = True
    include_cis_response = True
    
    quasar.AIEMGradCheck.test_fd_gradient_pauli(aiem=aiem, include_vqe_response=include_vqe_response, include_cis_response=include_cis_response)
    quasar.AIEMGradCheck.test_fd_gradient_monomer(aiem=aiem, include_vqe_response=include_vqe_response, include_cis_response=include_cis_response)
