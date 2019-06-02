# => Qiskit <= #

class QiskitBackend(Backend):

    def __init__(
        self,
        ):

        pass 

    @staticmethod
    def quasar_to_qiskit_angle(theta):
        return 2.0 * theta

    def native_circuit(
        self,
        circuit,
        ):

        import qiskit
        q = qiskit.QuantumRegister(circuit.N)
        qc = qiskit.QuantumCircuit(q)
        for key, gate in circuit.gates.items():
            T, key2 = key
            if gate.N == 1:
                index = key2[0]
                if gate.name == 'I':
                    qc.iden(q[index])
                elif gate.name == 'X':
                    qc.x(q[index])
                elif gate.name == 'Y':
                    qc.y(q[index])
                elif gate.name == 'Z':
                    qc.z(q[index])
                elif gate.name == 'H':
                    qc.h(q[index])
                elif gate.name == 'S':
                    qc.s(q[index])
                elif gate.name == 'T':
                    qc.t(q[index])
                elif gate.name == 'Rx':
                    qc.rx(QiskitBackend.quasar_to_qiskit_angle(gate.params['theta']), q[index]) 
                elif gate.name == 'Ry':
                    qc.ry(QiskitBackend.quasar_to_qiskit_angle(gate.params['theta']), q[index]) 
                elif gate.name == 'Rz':
                    qc.rz(QiskitBackend.quasar_to_qiskit_angle(gate.params['theta']), q[index]) 
                else:
                    raise RuntimeError('Gate translation to qiskit not known: %s' % gate)
            elif gate.N == 2:
                indexA = key2[0]
                indexB = key2[1]
                if gate.name == 'CNOT':
                    qc.cx(q[indexA], q[indexB])
                elif gate.name == 'CX':
                    qc.cx(q[indexA], q[indexB])
                elif gate.name == 'CY':
                    qc.cy(q[indexA], q[indexB])
                elif gate.name == 'CZ':
                    qc.cz(q[indexA], q[indexB])
                elif gate.name == 'SWAP':
                    qc.swap(q[indexA], q[indexB])
                else:
                    raise RuntimeError('Gate translation to qiskit not known: %s' % gate)
            else:
                raise RuntimeError('Cannot emit qiskit for N > 2')
                
        return qc

    def native_circuit_measurement(
        self,
        circuit,
        ):

        import qiskit
        qc = self.native_circuit(circuit)
        q = qc.qregs[0]
        c = qiskit.ClassicalRegister(circuit.N)
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
        return 'Qiskit Simulator Backend (Basic Aer Statevector)'

    @property
    def has_statevector(self):
        return True

    @property
    def has_measurement(self):
        return True

    def run_statevector(
        self,
        circuit,
        ):

        import qiskit
        circuit_native = self.native_circuit(circuit)
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
        # Rz gate
        phase_rz = 1.0 + 0.0j
        for key, gate in circuit.gates.items():
            if gate.name == 'Rz':
                phase_rz *= np.exp(-1.0j * gate.params['theta'])
        wfn *= phase_rz
        return wfn

    def run_measurement(
        self,
        circuit,
        nmeasurement,
        ):
    
        import qiskit
        circuit_native = self.native_circuit_measurement(circuit)
        measurements_native = qiskit.execute(circuit_native, backend=self.qasm_backend, shots=nmeasurement).result().get_counts()
        results = Measurement()
        for k, v in measurements_native.items():
            results[Ket(k, base=2)] = v
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
    def has_statevector(self):
        return False

    @property
    def has_measurement(self):
        return True

    def run_measurement(
        self,
        circuit,
        nmeasurement,
        ):
    
        import qiskit
        circuit_native = self.native_circuit_measurement(circuit)
        measurements_native = qiskit.execute(circuit_native, backend=self.qasm_backend, shots=nmeasurement).result().get_counts()
        results = Measurement()
        for k, v in measurements_native.items():
            results[Ket(k, base=2)] = v
        return results
        
