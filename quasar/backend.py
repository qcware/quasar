import numpy as np
from .pauli import Pauli, PauliString

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
            wraps an str an makes it unambiguous as to the qubit ordering in
            the ket.
        (3) the simulated statevector - a 2**N-dimensional real or complex
            vector of Hilbert-space amplitudes. Here, we use a np.ndarray of
            shape (2**N,) in Quasar/Cirq/Nielson-Chuang order.
        (4) a Pauli expectation value set - a higher-level object formed by
            infinite-sampling contraction of a simulated statevector or by
            statistical expectation value over a set of strings of many-body
            Pauli operators. Here, we compute this ourselves using either a
            simulated statevector or a set of Measurement objects computed in
            the "commuting group" space for the relevant Pauli operator. This
            object is returned as a Pauli object.

    The point here is that you spin up a Backend object of a specific type
    (such as a QuasarSimulatorBackend, QiskitSimulatorBackend,
    QiskitHardwareBackend, etc), pass Quasar circuits and Pauli objects as
    arguments into the backend object's functions, and receive output in the
    Quasar output types and ordering conventions described above.
""" 

class Backend(object):

    """ Class Backend represents a physical or simulated quantum circuit
        resource, which might support statevector simulation and/or measurement. 

        Class Backend declares an abstract set of API functions that must be
        overloaded by each specific Backend subclass, as well as some utility
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

    def __str__(self):
        """ A 1-line string representation of this Backend

        Returns:
            (str) - 1-line string representation of this Backend
        
        Backend subclasses should OVERLOAD this method.
        """ 
        raise NotImplementedError

    @property
    def summary_str(self):
        """ A more-extensive string representation of this Backend, optionally
            including current hardware state.

        Returns:
            (str) - multiline string representation of this Backend
        
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

    def build_native_circuit(
        self,
        circuit,
        **kwargs):
        """ Return the native object representation of the input Quasar circuit. 
        
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
        **kwargs):
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
        **kwargs):
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

    def run_pauli_expectation(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        **kwargs):

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
            return self.run_pauli_expectation_from_statevector(circuit, pauli, **kwargs)
        else:
            return self.run_pauli_expectation_from_measurement(circuit, pauli, nmeasurement, **kwargs)

        return pauli_expectation

    # => Utility Methods (Users should generally not call these) <= #

    def run_pauli_expectation_from_statevector(
        self,
        circuit,
        pauli,
        **kwargs): 

        if not self.has_statevector: raise RuntimeError('Backend does not have statevector')

        if pauli.max_order > 2: 
            raise NotImplementedError

        statevector = self.run_statevector(circuit, **kwargs)

        pauli_expectation = Pauli.zeros_like(pauli)
        if PauliString.I in pauli_expectation:
            pauli_expectation[PauliString.I] = 1.0
        # TODO
        for index in pauli_expectation.indices(1):
            A = index[0]
            P = Circuit.compute_pauli_1(wfn=statevector, A=A)
            for dA, DA in zip([1, 2, 3], ['X', 'Y', 'Z']):
                key = '%s%d' % (DA, A)
                if key in pauli_expectation:
                    pauli_expectation[key] = P[dA]
        for index in pauli_expectation.indices(2):
            A = index[0]
            B = index[1]
            P = Circuit.compute_pauli_2(wfn=statevector, A=A, B=B)
            for dA, DA in zip([1, 2, 3], ['X', 'Y', 'Z']):
                for dB, DB in zip([1, 2, 3], ['X', 'Y', 'Z']):
                    key = '%s%d*%s%d' % (DA, A, DB, B)
                    if key in pauli_expectation:
                        pauli_expectation[key] = P[dA, dB]

        return pauli_expectation

    def run_pauli_expectation_from_measurement(
        self,
        circuit,
        pauli,
        nmeasurement,
        **kwargs):
    
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
            nmeasurement=nmeasurement,
            **kwargs) for _ in circuits]
            
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
        pauli_expectation = Pauli(
            strings=pauli.strings,
            values=values,
            )
        return pauli_expectation

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
