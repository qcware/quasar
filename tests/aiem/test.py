import tomcat

if __name__ == '__main__':

    backend = tomcat.QuasarSimulatorBackend()
    # backend = tomcat.QiskitSimulatorBackend()

    datapath = '../../data/aiem/bchl-a-8-stack/tc'
    filenames = ['%s/%d/exciton.dat' % (datapath, _) for _ in range(1, 8+1)]
    charges = [0.0]*8
    N = 2
    nstate = 3
    connectivity = 'linear'
    cis_circuit_type = 'mark2'

    aiem_monomer = tomcat.AIEMMonomer.from_tc_exciton_files(
        filenames=filenames,
        charges=charges,
        N=N,
        connectivity=connectivity,
        )

    aiem_monomer_grad = tomcat.AIEMMonomerGrad.from_tc_exciton_files(
        filenames=filenames,
        charges=charges,
        N=N,
        connectivity=connectivity,
        )

    aiem = tomcat.AIEM.from_options(
        backend=backend,
        nstate=nstate,
        aiem_monomer=aiem_monomer,
        aiem_monomer_grad=aiem_monomer_grad,
        cis_circuit_type=cis_circuit_type,
        )
    aiem.compute_energy()
    
    print(aiem.cis_circuits[0])
    print(aiem.vqe_circuit)
    print(aiem.fci_cis_overlaps)
    print(aiem.fci_vqe_overlaps)
    print(aiem.vqe_cis_overlaps)
    
        
        
