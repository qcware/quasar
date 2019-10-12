import numpy as np
from .backend import Backend

class QuasarSimulatorBackend(Backend):

    def __init__(
        self,
        ):
        pass

    def __str__(self):
        return 'Quasar Simulator Backend (Statevector)'

    @property
    def summary_str(self):
        s = ''
        s += 'Quasar: An Ultralite Quantum Circuit Simulator\n'
        s += '   By Rob Parrish (rob.parrish@qcware.com)    '
        return s

    @property
    def has_run_statevector(self):
        return True

    @property
    def has_statevector_input(self):
        return True

    def run_statevector(
        self,
        circuit,
        statevector=None,   
        dtype=np.complex128,
        ):

        if statevector is None:
            statevector = np.zeros((2**circuit.nqubit,), dtype=dtype)
            statevector[0] = 1.0

        statevector1 = statevector.copy()
        statevector2 = np.zeros_like(statevector1)

        # TODO: Compression?

        min_qubit = circuit.min_qubit
        for key, gate in circuit.gates.items(): 
            times, qubits = key
            gate.apply_to_statevector(
                statevector1=statevector1,
                statevector2=statevector2,
                qubits=tuple(_ - min_qubit for _ in qubits),
                dtype=dtype,
                )
            statevector1, statevector2 = statevector2, statevector1
        
        return statevector1
