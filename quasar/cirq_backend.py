import numpy as np
from .backend import Backend

class CirqBackend(Backend):

    def __init__(
        self,
        ):
        
        pass 

    @staticmethod
    def quasar_to_cirq_angle(theta):
        return 2.0 * theta

    @staticmethod
    def cirq_to_quasar_angle(theta):
        return 0.5 * theta

    @property
    def native_circuit_type(self):
        import cirq
        return cirq.Circuit

    def build_native_circuit(
        self,
        circuit,
        bit_reversal=False,
        min_qubit=None,
        nqubit=None,
        ):

        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit

        import cirq
        q = [cirq.LineQubit(A) for A in range(nqubit)]
        qc = cirq.Circuit()
        for key, gate in circuit.gates.items():
            times, qubits = key
            if gate.nqubit == 1:
                qubit = qubits[0] - min_qubit
                if bit_reversal:
                    qubit = nqubit - qubit - 1
                if gate.name == 'I':
                    qc.append(cirq.I(q[qubit]))
                elif gate.name == 'X':
                    qc.append(cirq.X(q[qubit]))
                elif gate.name == 'Y':
                    qc.append(cirq.Y(q[qubit]))
                elif gate.name == 'Z':
                    qc.append(cirq.Z(q[qubit]))
                elif gate.name == 'H':
                    qc.append(cirq.H(q[qubit]))
                elif gate.name == 'S':
                    qc.append(cirq.S(q[qubit]))
                elif gate.name == 'T':
                    qc.append(cirq.T(q[qubit]))
                elif gate.name == 'Rx':
                    qc.append(cirq.Rx(CirqBackend.quasar_to_cirq_angle(gate.parameters['theta']))(q[qubit]))
                elif gate.name == 'Ry':
                    qc.append(cirq.Ry(CirqBackend.quasar_to_cirq_angle(gate.parameters['theta']))(q[qubit]))
                elif gate.name == 'Rz':
                    qc.append(cirq.Rz(CirqBackend.quasar_to_cirq_angle(gate.parameters['theta']))(q[qubit]))
                else:
                    raise RuntimeError('Gate translation to cirq not known: %s' % gate)
            elif gate.nqubit == 2:
                qubitA = qubits[0] - min_qubit
                qubitB = qubits[1] - min_qubit
                if bit_reversal:
                    qubitA = nqubit - qubitA - 1
                    qubitB = nqubit - qubitB - 1
                if gate.name == 'CX':
                    qc.append(cirq.CNOT(q[qubitA], q[qubitB]))
                # elif gate.name == 'CY':
                #     qc.cy(q[qubitA], q[qubitB])
                elif gate.name == 'CZ':
                    qc.append(cirq.CZ(q[qubitA], q[qubitB]))
                elif gate.name == 'SWAP':
                    qc.append(cirq.SWAP(q[qubitA], q[qubitB]))
                else:
                    raise RuntimeError('Gate translation to cirq not known: %s' % gate)
            else:
                raise RuntimeError('Cannot translate cirq for N > 2')

        for qubit in range(nqubit):
            qc.append(cirq.I(q[qubit]))
                
        return qc

    def build_native_circuit_measurement(
        self,
        circuit,
        **kwargs):

        import cirq
        qc = self.build_native_circuit(circuit, **kwargs).copy()
        for qubit in qc.all_qubits():
            qc.append(cirq.measure(qubit))
        return qc

class CirqSimulatorBackend(CirqBackend):

    def __init__(
        self,
        ):

        import cirq
        self.simulator = cirq.Simulator()

    def __str__(self):
        return 'Cirq Simulator Backend'

    @property
    def summary_str(self):
        return 'Cirq Simulator Backend'

    @property
    def has_statevector(self):
        return True

    def run_statevector(
        self,
        circuit,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        import cirq
        circuit_native = self.build_native_circuit(
            circuit, 
            bit_reversal=False, 
            min_qubit=min_qubit, 
            nqubit=nqubit,
            )
        statevector = np.array(statevector, dtype=np.complex64) if statevector is not None else statevector
        result = self.simulator.simulate(
            circuit_native, 
            initial_state=statevector,
            **kwargs)
        statevector = result.state_vector()
        statevector = np.array(statevector, dtype=dtype)
        return statevector

    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        **kwargs):
    
        raise NotImplementedError
