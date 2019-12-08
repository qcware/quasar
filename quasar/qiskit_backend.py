import numpy as np
from .backend import Backend
from .measurement import ProbabilityHistogram

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
        bit_reversal=False,
        min_qubit=None,
        nqubit=None,
        ):

        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit

        import qiskit
        q = qiskit.QuantumRegister(nqubit)
        qc = qiskit.QuantumCircuit(q)
        for key, gate in circuit.gates.items():
            times, qubits = key
            if gate.nqubit == 1:
                qubit = qubits[0] - min_qubit
                if bit_reversal:
                    qubit = nqubit - qubit - 1
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
                    qc.rx(QiskitBackend.quasar_to_qiskit_angle(gate.parameters['theta']), q[qubit]) 
                elif gate.name == 'Ry':
                    qc.ry(QiskitBackend.quasar_to_qiskit_angle(gate.parameters['theta']), q[qubit]) 
                elif gate.name == 'Rz':
                    qc.rz(QiskitBackend.quasar_to_qiskit_angle(gate.parameters['theta']), q[qubit]) 
                else:
                    raise RuntimeError('Gate translation to qiskit not known: %s' % gate)
            elif gate.nqubit == 2:
                qubitA = qubits[0] - min_qubit
                qubitB = qubits[1] - min_qubit
                if bit_reversal:
                    qubitA = nqubit - qubitA - 1
                    qubitB = nqubit - qubitB - 1
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

    def build_native_circuit_measurement(
        self,
        circuit,
        **kwargs):

        import qiskit
        qc = self.build_native_circuit(circuit, **kwargs)
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
        **kwargs):

        import qiskit
        circuit_native = self.build_native_circuit(
            circuit, 
            bit_reversal=True, 
            min_qubit=min_qubit, 
            nqubit=nqubit,
            )
        backend_options = {}
        if statevector is not None:
            backend_options['initial_statevector'] = statevector
        statevector = qiskit.execute(
            circuit_native, 
            self.backend,
            backend_options=backend_options,
            ).result().get_statevector()
        # NOTE: Incredible hack: Qiskit does not apply Rz(theta), instead
        # applies u1(theta):
        # 
        # Rz = [exp(-i theta)            ]
        #      [             exp(i theta)]
        # 
        # u1 = [1               ]
        #      [  exp(2 i theta)]
        # 
        # To correct, we must apply a global phase of exp(-1j * theta) for each
        # Rz gate. 
        phase_rz = 1.0 + 0.0j
        for key, gate in circuit.gates.items():
            if gate.name == 'Rz':
                phase_rz *= np.exp(-1.0j * gate.parameters['theta'])
        statevector *= phase_rz
        statevector = np.array(statevector, dtype=dtype)
        return statevector

    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        min_qubit=None,
        nqubit=None,
        **kwargs):
    
        if not isinstance(nmeasurement, int):
            raise RuntimeError('nmeasurement must be int: %s' % nmeasurement)
    
        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit

        import qiskit
        circuit_native = self.build_native_circuit_measurement(
            circuit, 
            bit_reversal=True, 
            min_qubit=min_qubit, 
            nqubit=nqubit,
            )
        job = qiskit.execute(
            circuit_native, 
            backend=self.qasm_backend, 
            shots=nmeasurement,
            )
        result = job.result()
        counts = result.get_counts()
        probabilities = ProbabilityHistogram(
            nqubit=nqubit,
            nmeasurement=nmeasurement,
            histogram={int(k, base=2) : v / nmeasurement for k, v in counts.items()},
            )
        return probabilities

class QiskitHardwareBackend(QiskitBackend):

    def __init__(
        self,
        backend,
        optimization_level=0,
        initial_layout=None,
        ):

        self.backend = backend
        self.optimization_level = optimization_level
        self.initial_layout = initial_layout

    def __str__(self):
        s = 'Qiskit Hardware Backend:\n'
        s += '  name = %s\n' % (self.backend.name())
        return s

    @property
    def summary_str(self):
        s = 'Qiskit Hardware Backend:\n'
        s += '  name               = %s\n' % (self.backend.name())
        s += '  provider           = (hub=%s, group=%s, project=%s)\n' % (
            self.backend.hub,
            self.backend.group,
            self.backend.project,
            )
        s += '  operational        = %s\n' % (self.backend.status().operational)
        s += '  pending_jobs       = %s\n' % (self.backend.status().pending_jobs)
        s += '  optimization_level = %s\n' % (self.optimization_level)
        s += '  initial_layout     = %s\n' % (self.initial_layout)
        # s += '  configuration = %s\n' % (self.backend.configuration())
        # s += '  properties = %s\n' % (self.backend.properties())
        return s

    @property
    def has_run_statevector(self):
        return False

    @property
    def has_statevector_input(self):
        return False

    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        min_qubit=None,
        nqubit=None,
        **kwargs):

        if not isinstance(nmeasurement, int):
            raise RuntimeError('nmeasurement must be int: %s' % nmeasurement)
    
        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit

        print('@IBM Call')
        print(circuit)

        import qiskit
        circuit_native = self.build_native_circuit_measurement(
            circuit,
            bit_reversal=False, 
            min_qubit=min_qubit, 
            nqubit=nqubit,
            )
        circuit_transpiled = qiskit.transpile(
            circuit_native, 
            backend=self.backend,
            optimization_level=self.optimization_level,
            initial_layout=None if self.initial_layout is None else self.initial_layout[:nqubit],
            )
        qobj = qiskit.assemble(
            circuit_transpiled,
            backend=self.backend,
            shots=nmeasurement,
            )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        probabilities = ProbabilityHistogram(
            nqubit=nqubit,
            nmeasurement=nmeasurement,
            histogram={int(k[::-1], base=2) : v / nmeasurement for k, v in counts.items()}, # WRONG
            )
        return probabilities
