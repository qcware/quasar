import numpy as np
import quasar

def write_nmm_file(
    filename,
    data,
    scale=1.0,
    offset=1,
    ):

    if not isinstance(data, np.ndarray): raise RuntimeError('data is not np.ndarray')
    if data.ndim != 2: raise RuntimeError('data.ndim != 2')
    if data.shape[1] != 3: raise RuntimeError('data.shape[1] != 3')

    fh = open(filename, 'w')
    for A in range(data.shape[0]):
        fh.write('ATOM   %5d  %8.3f %8.3f %8.3f\n' % (
            A + offset, 
            scale*data[A,0],
            scale*data[A,1],
            scale*data[A,2],
            ))

def print_grad(
    grad,
    ):

    for A, grad2 in enumerate(grad):

        print('Monomer %d:\n' % (A))
        for A2 in range(grad2.shape[0]):
            print('%24.16E %24.16E %24.16E' % (grad2[A2,0], grad2[A2,1], grad2[A2,2]))
        print('')
        
if __name__ == '__main__':

    import sys
    include_vqe_response = True if sys.argv[1] == 'True' else False
    include_cis_response = True if sys.argv[2] == 'True' else False
    I = int(sys.argv[3])

    backend = quasar.QuasarSimulatorBackend()

    datapath = '../../../data/aiem/bchl-a-2-stack/tc'
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

    grad = aiem.compute_vqe_gradient(I=I, include_vqe_response=include_vqe_response, include_cis_response=include_cis_response)
    print('VQE Gradient I = %d:\n' % I)
    print_grad(grad)
    grad = np.vstack(grad)
    write_nmm_file(
        'grad-%d-vqe%r-cis%r.nmm' % (I, include_vqe_response, include_cis_response),
        grad,
        scale=100.0,
        )
    
    grad = aiem.compute_cis_gradient(I=I)
    print('CIS Gradient I = %d:\n' % I)
    print_grad(grad)
    grad = np.vstack(grad)
    write_nmm_file(
        'grad-%d-cis.nmm' % (I),
        grad,
        scale=100.0,
        )

    grad = aiem.compute_fci_gradient(I=I)
    print('FCI Gradient I = %d:\n' % I)
    print_grad(grad)
    grad = np.vstack(grad)
    write_nmm_file(
        'grad-%d-fci.nmm' % (I),
        grad,
        scale=100.0,
        )
