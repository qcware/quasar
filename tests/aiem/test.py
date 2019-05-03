import hammer

if __name__ == '__main__':

    backend = hammer.QuasarSimulatorBackend()

    datapath = '../../data/aiem/bchl-a-8-stack/tc'
    filenames = ['%s/%d/exciton.dat' % (datapath, _) for _ in range(1, 8+1)]
    charges = [0.0]*8
    N = 6
    nstate = 3
    connectivity = 'linear'
    # cis_circuit_type = 'mark2'
    cis_circuit_type = 'mark1'

    aiem_monomer = hammer.AIEMMonomer.from_tc_exciton_files(
        filenames=filenames,
        charges=charges,
        N=N,
        connectivity=connectivity,
        )

    aiem_monomer_grad = hammer.AIEMMonomerGrad.from_tc_exciton_files(
        filenames=filenames,
        charges=charges,
        N=N,
        connectivity=connectivity,
        )

    aiem = hammer.AIEM.from_options(
        backend=backend,
        nstate=nstate,
        aiem_monomer=aiem_monomer,
        aiem_monomer_grad=aiem_monomer_grad,
        cis_circuit_type=cis_circuit_type,
        )
    aiem.compute_energy()

    
        
        
