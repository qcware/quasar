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
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ):

        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit

        if statevector is None:
            statevector = np.zeros((2**nqubit,), dtype=dtype)
            statevector[0] = 1.0

        statevector1 = statevector.copy()
        statevector2 = np.zeros_like(statevector1)

        # TODO: Compression?

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
