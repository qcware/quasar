import numpy as np
from ..quasar import quasar
from . import pauli

class Backend(object):

    def __init__(
        self,
        ):
        pass

    @property
    def has_statevector(self):
        return False

    @property
    def has_shots(self):
        return self.has_statevector

    def __str__(self):
        raise NotImplemented

    # => Circuit Emission <= #

    def emit_circuit(
        self,
        circuit,
        ):
        raise NotImplemented

    def emit_circuit_shots(
        self,
        circuit,
        ):
        raise NotImplemented

    # => Statevector Simulation <= #

    def simulate_statevector(
        self,
        circuit,
        ):

        raise NotImplemented

    def simulate_statevector_native(
        self,
        circuit_native,
        ):

        raise NotImplemented

    # => Pauli Density Matrices <= #

    def compute_pauli_dm_from_statevector_ideal(
        self,
        statevector,
        pauli2,
        ): 

        if pauli2.max_order > 2: 
            raise NotImplemented

        pauli_dm = pauli.Pauli.zeros_like(pauli2)
        if '1' in pauli_dm:
            pauli_dm['1'] = 1.0
        for index in pauli_dm.indices(1):
            A = index[0]
            P = quasar.Circuit.compute_pauli_1(wfn=statevector, A=A)
            for dA, DA in zip([1, 2, 3], ['X', 'Y', 'Z']):
                key = '%s%d' % (DA, A)
                if key in pauli_dm:
                    pauli_dm[key] = P[dA]
        for index in pauli_dm.indices(2):
            A = index[0]
            B = index[1]
            P = quasar.Circuit.compute_pauli_2(wfn=statevector, A=A, B=B)
            for dA, DA in zip([1, 2, 3], ['X', 'Y', 'Z']):
                for dB, DB in zip([1, 2, 3], ['X', 'Y', 'Z']):
                    key = '%s%d*%s%d' % (DA, A, DB, B)
                    if key in pauli_dm:
                        pauli_dm[key] = P[dA, dB]

        return pauli_dm

    def compute_pauli_dm_from_statevector_shots(
        self,
        statevector,
        pauli,
        shots,
        ): 

        raise NotImplemented

    # => Counts <= #

    def compute_counts_from_statevector(
        self,
        statevector,
        shots,
        ):

        p = (statevector.conj() * statevector).real
        
        
    # => Key User-Facing Methods <= #

    def compute_pauli_dm(
        self,
        circuit,
        pauli,
        shots=None,
        ):

        if shots is None:
            # Ideal density matrix from
            if not self.has_statevector: raise RuntimeError('Backend does not have statevector, must provide shots')
            statevector = self.simulate_statevector(circuit)
            pauli_dm = self.compute_pauli_dm_from_statevector_ideal(statevector, pauli)
        elif self.has_statevector:
            statevector = self.simulate_statevector(circuit)
            pauli_dm = self.compute_pauli_dm_from_statevector_shots(statevector, pauli, shots)
        else:
            raise NotImplemented

        return pauli_dm

    def compute_counts(
        self,
        circuit,
        shots,
        ):

        if self.has_statevector:
            statevector = self.simulate_statevector(circuit)
            counts = self.compute_counts_from_statevector(statevector, shots)
        else:
            raise NotImplemented

        return counts

    # => Utility Methods <= #

    @staticmethod
    def bit_reversal_permutation(N):
        seq = [0]
        for k in range(N):
            seq = [2*_ for _ in seq] + [2*_+1 for _ in seq]
        return seq

    @staticmethod
    def statevector_bit_reversal_permutation(
        statevector_native,
        ):

        N = (statevector_native.shape[0]&-statevector_native.shape[0]).bit_length()-1
        statevector = statevector_native[Backend.bit_reversal_permutation(N=N)]
        return statevector

# => Quasar <= #

class QuasarSimulatorBackend(Backend):

    def __init__(
        self,
        ):
        pass

    def __str__(self):
        return 'Quasar Simulator Backend (Statevector)'

    @property
    def has_statevector(self):
        return True

    def emit_circuit(
        self,
        circuit,
        ):
        return circuit.copy()

    def emit_circuit_shots(
        self,
        circuit,
        ):
        return circuit.copy()

    def simulate_statevector(
        self,
        circuit,
        ):
        return circuit.compressed().simulate()

    def simulate_statevector_native(
        self,
        circuit_native,
        ):
        return circuit.simulate()

# => Qiskit <= #

import qiskit

class QiskitBackend(Backend):

    def __init__(
        self,
        ):

        pass 

    @staticmethod
    def quasar_to_qiskit_angle(theta):
        return 2.0 * theta

    def emit_circuit(
        self,
        circuit,
        ):

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

    def emit_circuit_shots(
        self,
        circuit,
        ):

        qc = self.emit_circuit(circuit)
        q = qc.qregs[0]
        c = qiskit.ClassicalRegister(circuit.N)
        measure = qiskit.QuantumCircuit(q, c)
        measure.measure(q, c)
        return qc + measure

class QiskitSimulatorBackend(QiskitBackend):

    def __init__(
        self,
        ):

        self.backend = qiskit.BasicAer.get_backend('statevector_simulator')
        
    def __str__(self):
        return 'Qiskit Simulator Backend (Basic Aer Statevector)'

    @property
    def has_statevector(self):
        return True

    def simulate_statevector(
        self,
        circuit,
        ):

        circuit_native = self.emit_circuit(circuit)
        wfn_native = self.simulate_statevector_native(circuit_native)
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

    def simulate_statevector_native(
        self,
        circuit_native,
        ):

        return qiskit.execute(
            circuit_native, 
            self.backend,
            ).result().get_statevector()

def test_statevector_order(
    N=5,
    backend1=QuasarSimulatorBackend(),
    backend2=QiskitSimulatorBackend(),
    ):

    for I in range(N):
        circuit = quasar.Circuit(N=N)
        circuit.add_gate(T=0, key=I, gate=quasar.Gate.X)
        wfn1 = backend1.simulate_statevector(circuit)
        wfn2 = backend2.simulate_statevector(circuit)
        print(np.sum(wfn1*wfn2))
    

if __name__ == '__main__':

    import quasar
    circuit = quasar.Circuit(N=3)
    circuit.add_gate(T=0, key=0, gate=quasar.Gate.H)
    circuit.add_gate(T=1, key=(0,1), gate=quasar.Gate.CNOT)
    circuit.add_gate(T=2, key=(1,2), gate=quasar.Gate.CNOT)
    print(circuit)

    backend = QiskitSimulatorBackend()
    circuit2 = backend.emit_circuit(circuit)
    print(circuit2)
    circuit2 = backend.emit_circuit_shots(circuit)
    print(circuit2)

    print(backend.simulate_statevector(circuit))
    print(backend.simulate_statevector(circuit).dtype)

    test_statevector_order() 

