import quasar
from .mcvqe import MCVQE
from .aiem_data import AIEMMonomer

_default_datapath = '../../data/aiem/bchl-a-8-stack/tc'
_default_filenames = ['%s/%d/exciton.dat' % (_default_datapath, _) for _ in range(1, 8+1)]

def run_mcvqe(
    filenames=_default_filenames,
    N=2,
    connectivity='linear',
    backend_name='quasar',
    nmeasurement=None,
    nmeasurement_subspace=None,
    nstate=3,
    vqe_circuit_type='mark1x',
    ):

    """ Run MC-VQE+AIEM and return the computational results as a dict

    Params:
        filenames (list of str) - list of filenames of TeraChem exciton files
            (classical electronic structure computation output defining monomer
            characteristics for ab initio exciton model).
        N (int) - number of monomers to include (the first N filenames are
            used).
        backend_name (str) - 'quasar' or 'qiskit' or 'cirq' for the relevant
            statevector simulator backend.
        nmeasurement (None or int) - Number of measurements per observable for
            MC-VQE parameter optimization step. None indicates infinite
            averaging.
        nmeasurement_subspace (None or int) - Number of measurements per
            observable for MC-VQE subspace Hamiltonian step. None indicates
            infinite averaging.
        nstate (int) - Number of electronic states to determine.
        vqe_circuit_type (str) - 'mark1x' or 'mark1z' or 'mark2x' or 'mark2z'
            to determine the construction of the MC-VQE entangler circuit.
        
    Returns:
        (dict) dictionary of results with the following fields:
            'fci_E' (np.ndarry of shape (nstate,)) - Electronic state energies
                computed with FCI (reference energy subtracted)
            'fci_O' (np.ndarry of shape (nstate-1,)) - Oscillator strengths
                between ground and excited states computed with FCI
            'cis_E' (np.ndarry of shape (nstate,)) - Electronic state energies
                computed with CIS (reference energy subtracted)
            'cis_O' (np.ndarry of shape (nstate-1,)) - Oscillator strengths
                between ground and excited states computed with CIS
            'vqe_E' (np.ndarry of shape (nstate,)) - Electronic state energies
                computed with VQE (reference energy subtracted)
            'vqe_O' (np.ndarry of shape (nstate-1,)) - Oscillator strengths
                between ground and excited states computed with VQE
            'ref_E' (float) - Self energy of AIEM model
    """

    if backend_name == 'quasar':
        backend = quasar.QuasarSimulatorBackend()
    elif backend_name == 'qiskit':
        backend = quasar.QiskitSimulatorBackend()
    elif backend_name == 'cirq':
        backend = quasar.CirqSimulatorBackend()
    else:
        raise RuntimeError('Unknown backend_name: %s' % backend_name) 

    aiem_monomer = AIEMMonomer.from_tc_exciton_files(
        filenames=filenames,
        N=N,
        connectivity=connectivity,
        )

    aiem_solver = MCVQE.from_options(
        backend=backend,
        nstate=nstate,
        aiem_monomer=aiem_monomer,
        )
    aiem_solver.compute_energy()

    results = {
        'fci_E' : aiem_solver.fci_E,
        'fci_O' : aiem_solver.fci_O,
        'cis_E' : aiem_solver.cis_E,
        'cis_O' : aiem_solver.cis_O,
        'vqe_E' : aiem_solver.vqe_E,
        'vqe_O' : aiem_solver.vqe_O,
        'ref_E' : aiem_solver.aiem_hamiltonian_pauli.E,
    }

    return results

    
