import numpy as np
from .circuit import Circuit
from .backend import Backend
from .measurement import Ket, MeasurementResult

# => Cirq <= #

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
        ):

        # Dropthrough
        if isinstance(circuit, self.native_circuit_type): return circuit
    
        # Can only convert quasar -> cirq
        if not isinstance(circuit, Circuit): 
            raise RuntimeError('circuit must be Circuit type for build_native_circuit: %s' % (circuit))

        import cirq
        q = [cirq.LineQubit(A) for A in range(circuit.N)]
        qc = cirq.Circuit()
        for key in sorted(circuit.gates.keys()):
            T, qubits = key
            gate = circuit.gates[key]
            if gate.N == 1:
                qubit = qubits[0]
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
                    qc.append(cirq.Rx(CirqBackend.quasar_to_cirq_angle(gate.params['theta']))(q[qubit]))
                elif gate.name == 'Ry':
                    qc.append(cirq.Ry(CirqBackend.quasar_to_cirq_angle(gate.params['theta']))(q[qubit]))
                elif gate.name == 'Rz':
                    qc.append(cirq.Rz(CirqBackend.quasar_to_cirq_angle(gate.params['theta']))(q[qubit]))
                else:
                    raise RuntimeError('Gate translation to cirq not known: %s' % gate)
            elif gate.N == 2:
                qubitA = qubits[0]
                qubitB = qubits[1]
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
                
        return qc

    def build_quasar_circuit(
        self,
        native_circuit,
        ):

        # Dropthrough
        if isinstance(native_circuit, Circuit): return native_circuit
    
        # Can only convert quasar -> cirq
        if not isinstance(native_circuit, self.native_circuit_type): 
            raise RuntimeError('native_circuit must be Circuit type for build_native_native_circuit: %s' % (native_circuit))

        import cirq
        if not all(isinstance(qubit, cirq.LineQubit) for qubit in native_circuit.all_qubits()):
            raise RuntimeError('Translation from cirq only valid from LineQubit-based circuits')

        circuit = Circuit(N=len(native_circuit.all_qubits()))

        for time, moment in enumerate(native_circuit):
            for gate in moment.operations:
                qubits = gate.qubits
                gate2 = gate.gate
                strname = str(gate2)
                if len(qubits) == 1:
                    qubit = qubits[0].x
                    if strname == 'I':
                        circuit.I(qubit, time=time)
                    elif strname == 'X':
                        circuit.X(qubit, time=time)
                    elif strname == 'Y':
                        circuit.Y(qubit, time=time)
                    elif strname == 'Z':
                        circuit.Z(qubit, time=time)
                    elif strname == 'H':
                        circuit.H(qubit, time=time)
                    elif strname == 'S':
                        circuit.S(qubit, time=time)
                    elif strname == 'T':
                        circuit.T(qubit, time=time)
                    elif strname == 'H':
                        circuit.H(qubit, time=time)
                    elif strname[:2] == 'Rx':
                        circuit.Rx(qubit, time=time, theta=CirqBackend.cirq_to_quasar_angle(np.pi * gate2.exponent))
                    elif strname[:2] == 'Ry':
                        circuit.Ry(qubit, time=time, theta=CirqBackend.cirq_to_quasar_angle(np.pi * gate2.exponent))
                    elif strname[:2] == 'Rz':
                        circuit.Rz(qubit, time=time, theta=CirqBackend.cirq_to_quasar_angle(np.pi * gate2.exponent))
                    else:
                        raise RuntimeError('Gate translation from cirq not known: %s' % gate)
                elif len(qubits) == 2:
                    qubitA = qubits[0].x
                    qubitB = qubits[1].x
                    if strname == 'CNOT':
                        circuit.CX(qubitA, qubitB, time=time)
                    # elif strname == 'CY':
                    #     circuit.CY(qubitA, qubitB, time=time)
                    elif strname == 'CZ':
                        circuit.CZ(qubitA, qubitB, time=time)
                    elif strname == 'SWAP':
                        circuit.SWAP(qubitA, qubitB, time=time)
                    else:
                        raise RuntimeError('Gate translation from cirq not known: %s' % gate)
                else:
                    raise RuntimeError('Cannot translate cirq for N > 2')

        return circuit

    def build_native_circuit_in_basis(
        self,
        circuit,
        basis,
        ):

        circuit = self.build_native_circuit(circuit)
    
        if len(basis) > len(circuit.all_qubits()): raise RuntimeError('len(basis) > circuit.N. Often implies pauli.N > circuit.N')
        
        import cirq
        q = [cirq.LineQubit(A) for A in range(len(circuit.all_qubits()))]
        basis_circuit = cirq.Circuit()
        for A, char in enumerate(basis):
            if char == 'X': basis_circuit.append(cirq.H(q[A]))
            elif char == 'Y': basis_circuit.append(cirq.Rx(CirqBackend.quasar_to_cirq_angle(-np.pi / 4.0))(q[A]))
            elif char == 'Z': continue
            else: raise RuntimeError('Unknown basis: %s' % char)
        
        return circuit + basis_circuit
            
    def build_native_circuit_measurement(
        self,
        circuit,
        ):

        import cirq
        qc = self.build_native_circuit(circuit).copy()
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

    @property
    def has_measurement(self):
        return True

    def run_statevector(
        self,
        circuit,
        **kwargs):

        import cirq
        circuit_native = self.build_native_circuit(circuit)
        result = self.simulator.simulate(circuit_native, **kwargs)
        statevector = result.state_vector()
        statevector = np.array(statevector, dtype=np.complex128)
        # TODO: verify ordering, particularly for large qubit counts (looks corrects! -Tim)
        return statevector

    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        **kwargs):
    
        import cirq
        circuit_native = self.build_native_circuit_measurement(circuit)
        result = self.simulator.run(circuit_native, repetitions=nmeasurement, **kwargs)
        table = np.hstack(tuple(result.measurements[key] for key in sorted(result.measurements.keys())))
        results = MeasurementResult()
        for A in range(table.shape[0]):
            ket = Ket(''.join('1' if table[A,B] else '0' for B in range(table.shape[1])))
            results[ket] = 1 + results.get(ket, 0)
        return results
        
