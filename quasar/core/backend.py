import numpy as np
from .circuit import Circuit, Gate
from .pauli import Pauli

""" File backend.py contains some utility classes the standardize the
    input/output data for quantum circuits and some utility functions that
    abstract the details of the quantum backend from the user.

    It seems that there are two primary input data types for quantum circuits: 
        (1) a circuit - the specification of the quantum circuit, starting from
            the all-zero reference state. Here, we require that the input
            circuit be a quasar.Circuit object.
        (2) a many-body Pauli operator - the specification of the bases and
            sparsity patterns of the relevant outputs of the quantum circuit,
            e.g., as would be needed to construct a sparse Pauli-basis
            many-body density matrix. Here, we require that the input Pauli
            operator be a Pauli object.

    It seems that there are several primary output data types for NISQ-era
    quantum circuit manipulations:
        (1) the native circuit - the representation of the quantum circuit in
            the native object representation of the backend API. This can be
            useful for printing, inspection, etc. Here, the output type depends
            on the quantum backend.
        (2) the discrete quantum measurements - A set of kets |AB...Z> and the
            corresponding count of observations. Here, we represent this by the
            Measurement class, which is a dict of Ket : count pairs. Class Ket
            wraps an int an makes it unambiguous as to the qubit ordering in
            the ket.
        (3) the simulated statevector - a 2**N-dimensional real or complex
            vector of Hilbert-space amplitudes. Here, we use a np.ndarray of
            shape (2**N,) in Quasar order.
        (4) a Pauli density matrix - a higher-level object formed by
            infinite-sampling contraction of a simulated statevector or by
            statistical expectation value of a many-body Pauli operator. Here,
            we compute this ourselves using either a simulated statevector or a
            set of Measurement objects computed in the "commuting group" space
            for the relevant Pauli operator. This object is returned as a Pauli
            object.

    The point here is that you spin up a Backend object of a specific type
    (such as a QuasarSimulatorBackend, QiskitSimulatorBackend,
    QiskitHardwareBackend, etc), pass Quasar circuits and Pauli objects as
    arguments into the backend object's functions, and receive output in the
    Quasar output types and ordering conventions described above.
""" 

class Ket(int):

    """ Class Ket represents a binary string labeling a ket like |1010>, with
        the index order (endianness) fixed. 

        >>> ket = Ket(5)
        >>> print(ket.string(N=4))
        >>> print(ket[0])
        >>> print(ket[1])
        >>> print(ket[2])
        >>> print(ket[3])

    """

    def __getitem__(self, index):    
        """ Returns the bit value at indicated qubit index. 
        
        Params:
            index (int) - qubit index
        Returns:
            (int - 0 or 1) - value of bit at indicated qubit index
        """
        
        return (self & (1 << index)) >> index

    def string(self, N):
        return ''.join('%d' % self[A] for A in range(N))
    
class Measurement(dict):

    """ Class Measurement represents the output measurements (sometimes
        referred to as "counts" or "shots") from repeated observations of a
        quantum circuit.

    """

    def nmeasurement(self):
        return sum(v for v in self.values())

    def string(self, N):
        s = ''  
        for key in sorted(self.keys()):
            s += '|%s>: %d\n' % (key.string(N=N), self[key])
        return s
        
class Backend(object):

    """ Class Backend represents a physical or simulated quantum circuit
        resource, which might support statevector simulation and/or measurement. 

        Class Backend declares an abstract set of API functions that must be
        OVERLOADed by each specific Backend subclass, as well as some utility
        functions that are common to all Backend subclasses.
    """ 

    def __init__(
        self,
        ):
        pass

        """ Constructor, initializes and holds quantum resource pointers such
            as API keys.

        Backend subclasses should OVERLOAD this method.
        """

    @property
    def has_statevector(self):
        """ Does this Backend support run_statevector? 

        Returns:
            (bool) - True if run_statevector is supported else False.

        Backend subclasses should OVERLOAD this method.
        """ 
        raise NotImplementedError

    @property
    def has_measurement(self):
        """ Does this Backend support run_measurement? 

        Returns:
            (bool) - True if run_measurement is supported else False.

        Backend subclasses should OVERLOAD this method.
        """ 
        raise NotImplementedError

    def __str__(self):
        """ A 1-line string representation of this Backend

        Returns:
            (str) - 1-line string representation of this Backend
        
        Backend subclasses should OVERLOAD this method.
        """ 
        raise NotImplementedError

    def native_circuit(
        self,
        circuit,
        ):
        """ Return the native object represenation of the input Quasar circuit. 
        
        Params:
            circuit (quasar.Circuit) - Quasar circuit to translate to native
                representation.
        Returns:
            (? - native datatype) - Native circuit object representation.

        Backend subclasses should OVERLOAD this method.
        """
        raise NotImplementedError

    def run_statevector(
        self,
        circuit,
        ):
        """ Return the statevector after the action of circuit on the reference
            ket. Generally this involves the translation of circuit to native
            form, a call to the native statevector simulator (possibly
            including high-performance components or noise channels), and then
            a reordering/retyping step to return the statevector in Quasar
            convention.

            The output from this function is usually deterministic, though this
            can change depending on the specific backend.

        Params:
            circuit (quasar.Circuit) - Quasar circuit to simulate.
        Returns:
            (np.ndarray of shape (2**N,), dtype determined by backend) - the
                statevector in Quasar Hilbert space order.

        Backend subclasses should OVERLOAD this method.
        """
        raise NotImplementedError

    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        ):
        """ Return a Measurement object with the results of repeated quantum
            circuit preparation and measurement in the computational basis.

            The output from this function is generally stochastic.

        Params:
            circuit (quasar.Circuit) - Quasar circuit to measure.
            nmeasurement (int) - number of measurement
        Returns:
            (Measurement) - a Measurement object with the observed measurements
                in the computational basis, nmeasurement total measurements.
    
        Backend subclasses should OVERLOAD this method.
        """
        raise NotImplementedError

    def compute_pauli_dm(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        ):

        """ Return a Pauli object representating the density matrix of the quantum circuit. 

        Params:
            circuit (quasar.Circuit) - Quasar circuit to measure.
            pauli (Pauli) - Pauli object to use as a stencil for required Pauli
                density matrix elements. The strings in 
            nmeasurement (int or None) - integer number of measurements
                (backend must support run_measurement) or None to indicate
                infinite-sampling statevector contraction (backend must support
                run_statevector).
        Returns:
            (Pauli) - Pauli object representing the Pauli density matrix.
            
        Note that the number of measurements for each Pauli string are
        guaranteed to be *at least* nmeasurement, but more measurements may be
        taken for certain Pauli strings. The reason for this is that generally
        several versions of the quantum circuit must be constructed with
        one-qubit basis-transformation gates applied at the end (e.g., H to
        measure in the X basis), and then each version is sampled nmeasurement
        times. However, some Pauli strings might appear in multiple versions of
        the circuit, and we will take advantage of this to provide increased
        statistical convergence of these operators. For example, consider a
        2-qubit circuit with all X/Z Pauli density matrices requested: XA, ZA,
        XB, ZB, XX, XZ, ZX, and ZZ. This set of Pauli operators falls in the
        linear X/Z commuting group of XX, XZ, ZX, and ZZ, so 4x versions of
        circuit are prepared and measured nmeasurement times each. This
        provides nmeasurement observations for the Pauli density matrix
        elements XX, XZ, ZX, and ZZ, but 2*nmeasurement observations for the
        Pauli density matrix elements XA, ZA, XB, and ZB.
        """

        if nmeasurement is None:
            return self.compute_pauli_dm_from_statevector(circuit, pauli)
        else:
            return self.compute_pauli_dm_from_measurement(circuit, pauli, nmeasurement)

        return pauli_dm

    # => Utility Methods (Users should generally not call these) <= #

    def compute_pauli_dm_from_statevector(
        self,
        circuit,
        pauli,
        ): 

        if not self.has_statevector: raise RuntimeError('Backend does not have statevector')

        if pauli.max_order > 2: 
            raise NotImplementedError

        statevector = self.run_statevector(circuit)

        pauli_dm = Pauli.zeros_like(pauli)
        if '1' in pauli_dm:
            pauli_dm['1'] = 1.0
        for index in pauli_dm.indices(1):
            A = index[0]
            P = Circuit.compute_pauli_1(wfn=statevector, A=A)
            for dA, DA in zip([1, 2, 3], ['X', 'Y', 'Z']):
                key = '%s%d' % (DA, A)
                if key in pauli_dm:
                    pauli_dm[key] = P[dA]
        for index in pauli_dm.indices(2):
            A = index[0]
            B = index[1]
            P = Circuit.compute_pauli_2(wfn=statevector, A=A, B=B)
            for dA, DA in zip([1, 2, 3], ['X', 'Y', 'Z']):
                for dB, DB in zip([1, 2, 3], ['X', 'Y', 'Z']):
                    key = '%s%d*%s%d' % (DA, A, DB, B)
                    if key in pauli_dm:
                        pauli_dm[key] = P[dA, dB]

        return pauli_dm

    def compute_pauli_dm_from_measurement(
        self,
        circuit,
        pauli,
        nmeasurement,
        ):
    
        if not self.has_measurement: raise RuntimeError('Backend does not have measurement')

        # Commuting group
        if Backend.is_all_z(pauli):
            groups = Backend.z_commuting_group(pauli)
        else:
            groups = Backend.linear_xz_commuting_group(pauli)
        # Else exception will be raised if unknown commuting group

        # Modified circuits for basis transformations
        circuits = []
        for group in groups.keys():
            basis = Circuit(N=circuit.N)
            for A, char in enumerate(group):
                if char in ['I', 'Z']: continue
                elif char == 'X': basis.add_gate(T=0, key=A, gate=Gate.H)
                else: raise RuntimeError('Unknown basis: %s' % char)
            circuits.append(Circuit.concatenate([circuit, basis]))
    
        # Measurements in commuting group (quantum heavy)
        results = [self.run_measurement(
            circuit=_,
            nmeasurement=nmeasurement) for _ in circuits]
            
        # Counts for pauli strings
        counts = { _ : 0 for _ in pauli.strings }
        ns = { _ : 0 for _ in pauli.strings }
        for group, result in zip(groups.keys(), results):
            strings = groups[group]
            for string in strings:
                indices = string.indices
                ns[string] += nmeasurement
                for ket, count in result.items():
                    parity = sum(ket[_] for _ in indices) % 2
                    counts[string] += (-count) if parity else (+count)
                
        # Pauli density matrix values
        values = np.array([counts[_] / float(ns[_]) for _ in pauli.strings])
        pauli_dm = Pauli(
            strings=pauli.strings,
            values=values,
            )
        return pauli_dm

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

    @staticmethod
    def is_all_z(pauli):
        for string in pauli.strings:
            if any(_ != 'Z' for _ in string.chars):
                return False
        return True

    @staticmethod
    def z_commuting_group(pauli):

        groups = {}
        groups[tuple(['Z']*pauli.N)] = []

        for string in pauli.strings:
            
            # Do not do the identity operator
            if string.order == 0: continue

            # Add to all valid commuting groups
            found = False
            for group, strings2 in groups.items():
                valid = True
                for operator in string.operators:
                    index = operator.index
                    char = operator.char
                    if group[index] != 'I' and group[index] != char:
                        valid = False
                        break
                if not valid: continue
                strings2.append(string)
                found = True
            if not found: raise RuntimeError('Invalid string - not in Z commuting groups: %s' % string)

        return groups

    @staticmethod
    def linear_xz_commuting_group(pauli):

        groups = {}
        groups[tuple(['X']*pauli.N)] = []
        groups[tuple(['X', 'Z']*pauli.N)[:pauli.N]] = []
        groups[tuple(['Z', 'X']*pauli.N)[:pauli.N]] = []
        groups[tuple(['Z']*pauli.N)] = []

        for string in pauli.strings:
            
            # Do not do the identity operator
            if string.order == 0: continue

            # Add to all valid commuting groups
            found = False
            for group, strings2 in groups.items():
                valid = True
                for operator in string.operators:
                    index = operator.index
                    char = operator.char
                    if group[index] != 'I' and group[index] != char:
                        valid = False
                        break
                if not valid: continue
                strings2.append(string)
                found = True
            if not found: raise RuntimeError('Invalid string - not in linear XZ commuting groups: %s' % string)

        return groups

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

    @property
    def has_measurement(self):
        return False 

    def native_circuit(
        self,
        circuit,
        ):
        return circuit.copy()

    def run_statevector(
        self,
        circuit,
        ):
        return circuit.compressed().simulate()

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
        
def test_statevector_order(
    N,
    backend1,
    backend2,
    ):

    for I in range(N):
        circuit = Circuit(N=N)
        circuit.add_gate(T=0, key=I, gate=quasar.Gate.X)
        wfn1 = backend1.run_statevector(circuit)
        wfn2 = backend2.run_statevector(circuit)
        print(np.sum(wfn1*wfn2))
    

if __name__ == '__main__':

    circuit = Circuit(N=3)
    circuit.add_gate(T=0, key=0, gate=Gate.H)
    circuit.add_gate(T=1, key=(0,1), gate=Gate.CX)
    circuit.add_gate(T=2, key=(1,2), gate=Gate.CX)
    print(circuit)

    backend = QiskitSimulatorBackend()
    circuit2 = backend.native_circuit(circuit)
    print(circuit2)

    print(backend.run_statevector(circuit))
    print(backend.run_statevector(circuit).dtype)

    test_statevector_order() 

