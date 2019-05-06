import numpy as np
import quasar

if __name__ == '__main__':

    backend = quasar.QuasarSimulatorBackend()
    # backend = quasar.QiskitSimulatorBackend()

    datapath = '../../data/aiem/bchl-a-8-stack/tc'
    filenames = ['%s/%d/exciton.dat' % (datapath, _) for _ in range(1, 8+1)]
    # charges = [0.0]*8
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

    aiem = quasar.AIEM.from_options(
        backend=backend,
        nstate=nstate,
        aiem_monomer=aiem_monomer,
        aiem_monomer_grad=aiem_monomer_grad,
        cis_circuit_type=cis_circuit_type,
        vqe_circuit_type=vqe_circuit_type,
        )
    aiem.compute_energy()
    
    print(aiem.cis_circuits[0])
    print(aiem.vqe_circuit)
    print(aiem.fci_cis_overlaps)
    print(aiem.fci_vqe_overlaps)
    print(aiem.vqe_cis_overlaps)
    
        
    print(aiem.vqe_V)
        
    H = np.zeros((aiem.nstate,)*2)
    for I in range(aiem.nstate):
        for J in range(aiem.nstate):
            H[I,J] = aiem.vqe_D[I,J].dot(aiem.hamiltonian_pauli)
    print(np.max(np.abs(H - aiem.vqe_H)))
        
    H2 = np.zeros((aiem.nstate,)*2)
    for I in range(aiem.nstate):
        for J in range(aiem.nstate):
            H2[I,J] = aiem.vqe_D2[I,J].dot(aiem.hamiltonian_pauli)
    print(np.max(np.abs(H2 - np.diag(aiem.vqe_E))))
        
    backend = quasar.QiskitSimulatorBackend()
    pauli_dm = backend.compute_pauli_dm(circuit=aiem.cis_circuits[0], pauli=aiem.hamiltonian_pauli, nmeasurement=1025)
    
    print(pauli_dm.content_str)
