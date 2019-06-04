import numpy as np
from .circuit import Circuit
from .backend import Backend
from .measurement import Ket, Measurement

# => Qiskit <= #

class QiskitBackend(Backend):

    def __init__(
        self,
        ):
        
        pass 

    @staticmethod
    def quasar_to_qiskit_angle(theta):
        return 2.0 * theta

    @staticmethod
    def qiskit_to_quasar_angle(theta):
        return 0.5 * theta

    @property
    def native_circuit_type(self):
        import qiskit
        return qiskit.QuantumCircuit

    def build_native_circuit(
        self,
        circuit,
        ):

        # Dropthrough
        if isinstance(circuit, self.native_circuit_type): return circuit
    
        # Can only convert quasar -> qiskit
        if not isinstance(circuit, Circuit): raise RuntimeError('circuit must be Circuit type for build_native_circuit: %s' % (circuit))

        import qiskit
        q = qiskit.QuantumRegister(circuit.N)
        qc = qiskit.QuantumCircuit(q)
        for key in sorted(circuit.gates.keys()):
            T, qubits = key
            gate = circuit.gates[key]
            if gate.N == 1:
                qubit = qubits[0]
                if gate.name == 'I':
                    qc.iden(q[qubit])
                elif gate.name == 'X':
                    qc.x(q[qubit])
                elif gate.name == 'Y':
                    qc.y(q[qubit])
                elif gate.name == 'Z':
                    qc.z(q[qubit])
                elif gate.name == 'H':
                    qc.h(q[qubit])
                elif gate.name == 'S':
                    qc.s(q[qubit])
                elif gate.name == 'T':
                    qc.t(q[qubit])
                elif gate.name == 'Rx':
                    qc.rx(QiskitBackend.quasar_to_qiskit_angle(gate.params['theta']), q[qubit]) 
                elif gate.name == 'Ry':
                    qc.ry(QiskitBackend.quasar_to_qiskit_angle(gate.params['theta']), q[qubit]) 
                elif gate.name == 'Rz':
                    qc.rz(QiskitBackend.quasar_to_qiskit_angle(gate.params['theta']), q[qubit]) 
                else:
                    raise RuntimeError('Gate translation to qiskit not known: %s' % gate)
            elif gate.N == 2:
                qubitA = qubits[0]
                qubitB = qubits[1]
                if gate.name == 'CX':
                    qc.cx(q[qubitA], q[qubitB])
                elif gate.name == 'CY':
                    qc.cy(q[qubitA], q[qubitB])
                elif gate.name == 'CZ':
                    qc.cz(q[qubitA], q[qubitB])
                elif gate.name == 'SWAP':
                    qc.swap(q[qubitA], q[qubitB])
                else:
                    raise RuntimeError('Gate translation to qiskit not known: %s' % gate)
            else:
                raise RuntimeError('Cannot translate qiskit for N > 2')
                
        return qc

    def build_quasar_circuit(
        self,
        native_circuit,
        ):

        # Dropthrough
        if isinstance(native_circuit, Circuit): return native_circuit
    
        # Can only convert quasar -> qiskit
        if not isinstance(native_circuit, self.native_circuit_type): raise RuntimeError('native_circuit must be Circuit type for build_native_native_circuit: %s' % (native_circuit))

        if len(native_circuit.qregs) != 1: raise RuntimeError('Multiple qregs - translation error')
        circuit = Circuit(N=len(native_circuit.qregs[0]))

        for gate in native_circuit:
            if len(gate.qargs) == 1:
                qubit = gate.qargs[0][1]
                if gate.name == 'id':
                    circuit.I(qubit)
                elif gate.name == 'x':
                    circuit.X(qubit)
                elif gate.name == 'y':
                    circuit.Y(qubit)
                elif gate.name == 'z':
                    circuit.Z(qubit)
                elif gate.name == 'h':
                    circuit.H(qubit)
                elif gate.name == 's':
                    circuit.S(qubit)
                elif gate.name == 't':
                    circuit.T(qubit)
                elif gate.name == 'rx':
                    circuit.Rx(qubit, theta=QiskitBackend.qiskit_to_quasar_angle(float(gate.param[0])))
                elif gate.name == 'ry':
                    circuit.Ry(qubit, theta=QiskitBackend.qiskit_to_quasar_angle(float(gate.param[0])))
                elif gate.name == 'rz':
                    circuit.Rz(qubit, theta=QiskitBackend.qiskit_to_quasar_angle(float(gate.param[0])))
                else:
                    raise RuntimeError('Gate translation to quasar not known: %s' % gate)
            elif len(gate.qargs) == 2:
                qubitA = gate.qargs[0][1]
                qubitB = gate.qargs[1][1]
                if gate.name == 'cx':
                    circuit.CX(qubitA, qubitB)
                elif gate.name == 'cy':
                    circuit.CY(qubitA, qubitB)
                elif gate.name == 'cz':
                    circuit.CZ(qubitA, qubitB)
                elif gate.name == 'swap':
                    circuit.SWAP(qubitA, qubitB)
                else:
                    raise RuntimeError('Gate translation to quasar not known: %s' % gate)
            else:
                raise RuntimeError('Cannot translate qiskit for N > 2')

        return circuit

    def build_native_circuit_in_basis(
        self,
        circuit,
        basis,
        ):

        circuit = self.build_native_circuit(circuit)
    
        if len(basis) > len(circuit.qregs[0]): raise RuntimeError('len(basis) > circuit.N. Often implies pauli.N > circuit.N')
        
        import qiskit
        q = circuit.qregs[0]
        basis_circuit = qiskit.QuantumCircuit(q)
        for A, char in enumerate(basis): 
            if char == 'X': basis_circuit.h(q[A])
            elif char == 'Y': basis_circuit.rx(QiskitBackend.quasar_to_qiskit_angle(-np.pi / 4.0), q[A])
            elif char == 'Z': continue # Computational basis
            else: raise RuntimeError('Unknown basis: %s' % char)
        
        return circuit + basis_circuit

    def build_native_circuit_measurement(
        self,
        circuit,
        ):

        import qiskit
        qc = self.build_native_circuit(circuit)
        q = qc.qregs[0]
        c = qiskit.ClassicalRegister(len(q))
        measure = qiskit.QuantumCircuit(q, c)
        measure.measure(q, c)
        return qc + measure

class QiskitSimulatorBackend(QiskitBackend):

    def __init__(
        self,
        ):

        import qiskit
        self.backend = qiskit.BasicAer.get_backend('statevector_simulator')
        self.qasm_backend = qiskit.BasicAer.get_backend('qasm_simulator')
        
    def __str__(self):
        return 'Qiskit Simulator Backend'

    @property
    def summary_str(self):
        return 'Qiskit Simulator Backend'

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

        import qiskit
        circuit_native = self.build_native_circuit(circuit)
        wfn_native = qiskit.execute(circuit_native, self.backend).result().get_statevector()
        wfn = self.statevector_bit_reversal_permutation(wfn_native)
        # NOTE: Incredible hack: Qiskit does not apply Rz(theta), instead
        # applies u1(theta):
        # 
        # Rz = [exp(-i theta)              ]
        #      [             exp(i theta)]
        # 
        # u1 = [1                ]
        #      [  exp(2 i theta)]
        # 
        # To correct, we must apply a global phase of exp(-1j * theta) for each
        # Rz gate. We only do this if the user supplied a Circuit object.
        if isinstance(circuit, Circuit):
            phase_rz = 1.0 + 0.0j
            for key, gate in circuit.gates.items():
                if gate.name == 'Rz':
                    phase_rz *= np.exp(-1.0j * gate.params['theta'])
            wfn *= phase_rz
        return wfn

    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        **kwargs):
    
        import qiskit
        circuit_native = self.build_native_circuit_measurement(circuit)
        measurements_native = qiskit.execute(circuit_native, backend=self.qasm_backend, shots=nmeasurement).result().get_counts()
        results = Measurement()
        for k, v in measurements_native.items():
            results[Ket(k[::-1])] = v
        return results

class QiskitHardwareBackend(QiskitBackend):

    def __init__(
        self,
        backend_name='ibmq_20_tokyo',
        ):

        import qiskit
        self.backend = qiskit.IBMQ.get_backend(backend_name)
        
    def __str__(self):
        return 'Qiskit Hardware Backend (%s)' % self.backend

    @property
    def summary_str(self):
        return 'Qiskit Hardware Backend (%s)' % self.backend

    @property
    def has_statevector(self):
        return False

    @property
    def has_measurement(self):
        return True

    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        **kwargs):
    
        import qiskit
        circuit_native = self.build_native_circuit_measurement(circuit)
        measurements_native = qiskit.execute(circuit_native, backend=self.qasm_backend, shots=nmeasurement).result().get_counts()
        results = Measurement()
        for k, v in measurements_native.items():
            results[Ket(k[::-1])] = v
        return results
        
