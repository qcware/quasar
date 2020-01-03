import numpy as np
from .backend import Backend
from .circuit import Gate, Circuit

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
            statevector1 = np.zeros((2**nqubit,), dtype=dtype)
            statevector1[0] = 1.0
        else:
            statevector1 = statevector.copy()
        statevector2 = np.zeros_like(statevector1)

        qubits = [_ - min_qubit for _ in range(circuit.min_qubit, circuit.min_qubit + circuit.nqubit)]

        return circuit.apply_to_statevector(
            statevector1=statevector1,
            statevector2=statevector2,
            qubits=qubits,
            dtype=dtype,
            )[0]

    def run_pauli_sigma(
        self,
        pauli,
        statevector,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        min_qubit = pauli.min_qubit if min_qubit is None else min_qubit
        nqubit = pauli.nqubit if nqubit is None else nqubit

        pauli_gates_mody = {
            'X' : Gate.X,
            'Y' : Gate.U1(np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)),
            'Z' : Gate.Z,
        }

        statevector2 = np.zeros_like(statevector)
        for string, value in pauli.items():
            circuit2 = Circuit()
            ny = 0
            for qubit, char in string:
                circuit2.add_gate(pauli_gates_mody[char], qubit)
                if char == 'Y':
                    ny += 1
            statevector3 = self.run_statevector(
                circuit=circuit2,
                statevector=statevector,
                dtype=dtype,
                min_qubit=min_qubit,
                nqubit=nqubit,
                )
            scal = (-1.j)**ny * value
            if dtype in (np.float32, np.float64):
                scal = dtype(scal.real)
            else:
                scal = dtype(scal)
            statevector2 += scal * statevector3

        return statevector2

