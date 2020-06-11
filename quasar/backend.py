import numpy as np
import collections
import sortedcontainers
from .pauli import Pauli, PauliExpectation, PauliString
from .algebra import Algebra
from .circuit import Gate, Circuit

class Backend(object):

    """ Class Backend represents a physical or simulated quantum circuit
        resource. Backends must implement `run_measurement`, from which many
        higher-order quantities may be computed, e.g., `run_pauli_expectation`,
        `run_pauli_expectation_value`, `run_pauli_expectation_value_gradient`,
        etc. Backends supporting classical statevector-based simulation may
        also optionally implement `run_statevector,` from which many additional
        higher-order quantities may be computed, e.g., `run_unitary`,
        `run_density_matrix`, and ideal-infinite-sampled versions of the
        previously-discussed higher-order quantities. Backends may additionally
        overload any of the stock higher-order methods declared here,
        potentially providing increased performance or additional methodology.
    """ 

    def __init__(
        self,
        ):

        """ Constructor, initializes and holds quantum resource pointers such
            as API keys.

        Backend subclasses should OVERLOAD this method.
        """
        pass

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
        raise NotImplementedError

    @property
    def has_run_statevector(
        self,
        ):

        """ Does this Backend support run_statevector? 

        Returns:
            (bool) - True if run_statevector is supported else False.

        Backend subclasses should OVERLOAD this method.
        """ 
        raise NotImplementedError

    @property
    def has_statevector_input(
        self,
        ):

        """ Does this Backend allow statevector to be passed as input argument
            to various run methods?

        Returns:
            (bool) - True if statevector input arguments can be supplied else
                False.

        Backend subclasses should OVERLOAD this method.
        """

        raise NotImplementedError

    def run_statevector(
        self,
        circuit,
        statevector=None,   
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        raise NotImplementedError

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

        pauli_gates = {
            'X' : Gate.X,
            'Y' : Gate.Y,
            'Z' : Gate.Z,
        }

        statevector2 = np.zeros_like(statevector)
        for string, value in pauli.items():
            circuit2 = Circuit()
            for qubit, char in string:
                circuit2.add_gate(pauli_gates[char], qubit)
            statevector3 = self.run_statevector(
                circuit=circuit2,
                statevector=statevector,
                dtype=dtype,
                min_qubit=min_qubit,
                nqubit=nqubit,
                )
            statevector2 += value * statevector3

        return statevector2

    def run_pauli_diagonal(
        self,
        pauli,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        min_qubit = pauli.min_qubit if min_qubit is None else min_qubit
        nqubit = pauli.nqubit if nqubit is None else nqubit

        # All I or Z strings
        pauli2 = Pauli.zero()
        for string, value in pauli.items():
            if len(string) == 0 or all(_ == 'Z' for _ in string.chars):
                pauli2[string] = value

        statevector = np.ones((2**nqubit,), dtype=dtype)
        return self.run_pauli_sigma(
            pauli=pauli2,
            statevector=statevector,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            **kwargs)

    def run_unitary(
        self,
        circuit,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        nqubit = circuit.nqubit if nqubit is None else nqubit

        U = np.zeros((2**nqubit,)*2, dtype=dtype)
        for i in range(2**nqubit):
            statevector = np.zeros((2**nqubit,), dtype=dtype)
            statevector[i] = 1.0
            U[:, i] = self.run_statevector(circuit, statevector=statevector, dtype=dtype, **kwargs)

        return U

    def run_density_matrix(
        self,
        circuit,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        statevector = self.run_statevector(
            circuit=circuit,
            statevector=statevector,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            **kwargs)

        return np.outer(statevector, statevector.conj())

    def run_measurement(
        self,
        circuit,
        nmeasurement=None,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        statevector = self.run_statevector(
            circuit=circuit,
            statevector=statevector,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            **kwargs)

        return Algebra.sample_histogram_from_probabilities(
            probabilities=(np.conj(statevector) * statevector).real,
            nmeasurement=nmeasurement,
            )

    def run_pauli_expectation(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):
        
        if nmeasurement is None:
            return self.run_pauli_expectation_ideal(
                circuit=circuit,
                pauli=pauli,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
        else:
            return self.run_pauli_expectation_measurement(
                circuit=circuit,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)

    def run_pauli_expectation_value(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        if nmeasurement is None:
            return self.run_pauli_expectation_value_ideal(
                circuit=circuit,
                pauli=pauli,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
        else:
            pauli_expectation = self.run_pauli_expectation_measurement(
                circuit=circuit,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            return pauli_expectation.dot(pauli)

    def run_pauli_expectation_value_gradient(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        parameter_indices=None,
        **kwargs):
        
        # Current circuit parameter values
        parameter_values = circuit.parameter_values

        # Default to taking the gradient with respect to all parameters
        if parameter_indices is None:
            parameter_indices = list(range(len(parameter_values)))

        # Check that the gradient formula is known for these parameters (i.e., Rx, Ry, Rz gates)
        parameter_keys = circuit.parameter_keys
        for parameter_index in parameter_indices:
            key = parameter_keys[parameter_index]
            times, qubits, key2 = key
            gate = circuit.gates[(times, qubits)]
            if not gate.name in ('Rx', 'Ry', 'Rz'): 
                raise RuntimeError('Unknown gradient rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)

        # Evaluate the gradient by the parameter shift rule
        gradient = np.zeros((len(parameter_indices),), dtype=dtype)
        circuit2 = circuit.copy()
        for index, parameter_index in enumerate(parameter_indices):
            # +
            parameter_values2 = parameter_values.copy()
            parameter_values2[parameter_index] += np.pi / 4.0
            circuit2.set_parameter_values(parameter_values2)
            Ep = self.run_pauli_expectation_value(
                circuit=circuit2,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            # -
            parameter_values2 = parameter_values.copy()
            parameter_values2[parameter_index] -= np.pi / 4.0
            circuit2.set_parameter_values(parameter_values2)
            Em = self.run_pauli_expectation_value(
                circuit=circuit2,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            # Assembly
            gradient[index] = Ep - Em

        return gradient

    def run_pauli_expectation_value_hessian(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        parameter_pair_indices=None,
        **kwargs):

        # Current circuit parameter values
        parameter_values = circuit.parameter_values

        # Default to taking the gradient with respect to all parameters
        if parameter_pair_indices is None:
            parameter_pair_indices = []
            for i in range(len(parameter_values)):
                for j in range(len(parameter_values)):
                    parameter_pair_indices.append((i,j))

        # Check that the Hessian formula is known for these parameters (i.e., Rx, Ry, Rz gates)
        parameter_keys = circuit.parameter_keys
        for parameter_index1, parameter_index2 in parameter_pair_indices:
            key = parameter_keys[parameter_index1]
            times, qubits, key2 = key
            gate = circuit.gates[(times, qubits)]
            if not gate.name in ('Rx', 'Ry', 'Rz'): 
                raise RuntimeError('Unknown Hessian rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)
            key = parameter_keys[parameter_index2]
            times, qubits, key2 = key
            gate = circuit.gates[(times, qubits)]
            if not gate.name in ('Rx', 'Ry', 'Rz'): 
                raise RuntimeError('Unknown Hessian rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)

        # Evaluate the gradient by the parameter shift rule
        hessian = np.zeros((len(parameter_pair_indices),), dtype=dtype)
        circuit2 = circuit.copy()
        for index, parameter_pair_index in enumerate(parameter_pair_indices):
            parameter_index1, parameter_index2 = parameter_pair_index
            symmetric = (parameter_index2, parameter_index1) in parameter_pair_indices
            if symmetric and parameter_index1 > parameter_index2: continue
            # ++
            parameter_values2 = parameter_values.copy()
            parameter_values2[parameter_index1] += np.pi / 4.0
            parameter_values2[parameter_index2] += np.pi / 4.0
            circuit2.set_parameter_values(parameter_values2)
            Epp = self.run_pauli_expectation_value(
                circuit=circuit2,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            # +-
            parameter_values2 = parameter_values.copy()
            parameter_values2[parameter_index1] += np.pi / 4.0
            parameter_values2[parameter_index2] -= np.pi / 4.0
            circuit2.set_parameter_values(parameter_values2)
            Epm = self.run_pauli_expectation_value(
                circuit=circuit2,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            # -+
            parameter_values2 = parameter_values.copy()
            parameter_values2[parameter_index1] -= np.pi / 4.0
            parameter_values2[parameter_index2] += np.pi / 4.0
            circuit2.set_parameter_values(parameter_values2)
            Emp = self.run_pauli_expectation_value(
                circuit=circuit2,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            # --
            parameter_values2 = parameter_values.copy()
            parameter_values2[parameter_index1] -= np.pi / 4.0
            parameter_values2[parameter_index2] -= np.pi / 4.0
            circuit2.set_parameter_values(parameter_values2)
            Emm = self.run_pauli_expectation_value(
                circuit=circuit2,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            # Assembly
            hessian[index] = Epp - Epm - Emp + Emm
            if symmetric:
                hessian[parameter_pair_indices.index((parameter_index2, parameter_index1))] = hessian[index]

        return hessian

    def run_pauli_expectation_value_gradient_pauli_contraction(
        self,
        circuit,
        pauli,
        parameter_coefficients,
        nmeasurement=None,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        parameter_indices=None,
        **kwargs):
        
        # Current circuit parameter values
        parameter_values = circuit.parameter_values

        # Default to taking the gradient with respect to all parameters
        if parameter_indices is None:
            parameter_indices = list(range(len(parameter_values)))

        # Check that parameter coefficients make sense
        if len(parameter_coefficients) != len(parameter_indices):
           raise RuntimeError('len(parameter_coefficients) != len(parameter_indices)')

        # Check that the gradient formula is known for these parameters (i.e., Rx, Ry, Rz gates)
        parameter_keys = circuit.parameter_keys
        for parameter_index in parameter_indices:
            key = parameter_keys[parameter_index]
            times, qubits, key2 = key
            gate = circuit.gates[(times, qubits)]
            if not gate.name in ('Rx', 'Ry', 'Rz'): 
                raise RuntimeError('Unknown gradient rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)

        # Evaluate the gradient by the parameter shift rule
        pauli_gradient = PauliExpectation.zero()
        circuit2 = circuit.copy()
        for index, parameter_index in enumerate(parameter_indices):
            # +
            parameter_values2 = parameter_values.copy()
            parameter_values2[parameter_index] += np.pi / 4.0
            circuit2.set_parameter_values(parameter_values2)
            Gp = self.run_pauli_expectation(
                circuit=circuit2,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            # -
            parameter_values2 = parameter_values.copy()
            parameter_values2[parameter_index] -= np.pi / 4.0
            circuit2.set_parameter_values(parameter_values2)
            Gm = self.run_pauli_expectation(
                circuit=circuit2,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            # Assembly
            pauli_gradient += parameter_coefficients[index] * (Gp - Gm)

        return pauli_gradient

    # => Utility Methods <= #

    def run_pauli_expectation_ideal(
        self,
        circuit,
        pauli,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        min_qubit = pauli.min_qubit if min_qubit is None else min_qubit
        nqubit = pauli.nqubit if nqubit is None else nqubit

        statevector = self.run_statevector(
            circuit=circuit,
            statevector=statevector,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            **kwargs)

        pauli_expectation = PauliExpectation.zero()
        for string in pauli.keys():

            pauli2 = Pauli.zero()
            pauli2[string] = 1.0 
            statevector2 = self.run_pauli_sigma(
                pauli=pauli2,
                statevector=statevector,
                dtype=dtype,
                min_qubit=min_qubit,
                nqubit=nqubit,
                )
            scal = np.sum(statevector.conj() * statevector2)
            pauli_expectation[string] = scal

        return pauli_expectation

    def run_pauli_expectation_value_ideal(
        self,
        circuit,
        pauli,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        statevector = self.run_statevector(
            circuit=circuit,
            statevector=statevector,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            **kwargs)

        statevector2 = self.run_pauli_sigma(
            pauli=pauli,
            statevector=statevector,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            **kwargs)

        return np.sum(statevector.conj() * statevector2)

    # => Measurement-based Pauli expectations <= #

    # TODO: As always, there remains much work to be done in the conceptual,
    # pragmatical, and syntactical elements of this functionality

    def run_pauli_expectation_measurement(
        self,
        circuit,
        pauli,
        nmeasurement,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        min_qubit = pauli.min_qubit if min_qubit is None else min_qubit
        nqubit = pauli.nqubit if nqubit is None else nqubit

        # Determine commuting group
        groups = self.linear_commuting_group(
            pauli,
            min_qubit=min_qubit,
            nqubit=nqubit,
            )
        # Else exception will be raised if unknown commuting group
        # TODO: Optimally cover all commuting groups

        # Modified circuits for basis transformations
        circuits = [self.circuit_in_basis(
            circuit, 
            basis,
            min_qubit=min_qubit,
            nqubit=nqubit,
            ) 
            for basis in groups.keys()]
    
        # Measurements in commuting group (quantum heavy)
        probabilities = [self.run_measurement(
            circuit=circuit,
            nmeasurement=nmeasurement,
            statevector=statevector,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            **kwargs) for circuit in circuits]

        # Convert to counts
        results = [_.to_count_histogram() for _ in probabilities]

        # Counts for pauli strings
        counts = { _ : 0 for _ in pauli.keys() }
        ns = { _ : 0 for _ in pauli.keys() }
        for group, result in zip(groups.keys(), results):
            strings = groups[group]
            for string in strings:
                qubits = string.qubits
                ns[string] += nmeasurement
                for ket, count in result.items():
                    parity = sum((ket & (1 << (nqubit - 1 - (_ - min_qubit)))) >> (nqubit - 1 - (_ - min_qubit)) for _ in qubits) % 2
                    counts[string] += (-count) if parity else (+count)

        # Pauli density matrix values
        pauli_expectation = PauliExpectation(collections.OrderedDict([
            (_, counts[_] / max(ns[_], 1)) for _ in pauli.keys()]))
        if PauliString.I() in pauli_expectation:
            pauli_expectation[PauliString.I()] = 1.0 
        return pauli_expectation

    @staticmethod
    def linear_commuting_group(
        pauli,
        min_qubit=None,
        nqubit=None,
        ):

        min_qubit = pauli.min_qubit if min_qubit is None else min_qubit
        nqubit = pauli.nqubit if nqubit is None else nqubit

        keys = sortedcontainers.SortedSet()
        for string in pauli.keys():
            for qubit, char in string:
                keys.add(char)
        
        groups = collections.OrderedDict()
        for keyA in keys:
            for keyB in keys:
                groups[((keyA + keyB)*nqubit)[:nqubit]] = []

        for string in pauli.keys():

            # Do not do the identity operator
            if string.order == 0: continue

            # Add to all valid commuting groups
            found = False
            for group, strings2 in groups.items():
                valid = True
                for qubit, char in string:
                    if group[qubit - min_qubit] != char:
                        valid = False
                        break
                if not valid: continue
                strings2.append(string)
                found = True
            if not found: raise RuntimeError('Invalid string - not in linear commuting group: %s' % str(string))

        return groups

    @staticmethod
    def circuit_in_basis(
        circuit,
        basis,
        min_qubit=None,
        nqubit=None,
        ):

        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit

        if len(basis) != nqubit: raise RuntimeError('len(basis) != nqubit')

        basis_circuit = Circuit()
        for A, char in enumerate(basis): 
            qubit = A - min_qubit
            if char == 'X': basis_circuit.H(qubit)
            elif char == 'Y': basis_circuit.Rx2(qubit)
            elif char == 'Z': continue # Computational basis
            else: raise RuntimeError('Unknown basis: %s' % char)
    
        return Circuit.join_in_time([circuit, basis_circuit])
            
    # => Subset Hamiltonian Utilities <= #

    def run_pauli_matrix_subset(
        self,
        pauli,
        kets,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ):

        min_qubit = pauli.min_qubit if min_qubit is None else min_qubit
        nqubit = pauli.nqubit if nqubit is None else nqubit

        if len(set(kets)) != len(kets): 
            raise RuntimeError('Kets are not unique')

        kets2 = { ket : index for index, ket in enumerate(kets) }

        H = np.zeros((len(kets),)*2, dtype=np.complex128)
        for ket_index, ket in enumerate(kets):
            for string, value in pauli.items():
                value = value + 0.j # Make sure value is complex
                bra = ket
                for qubit2, char in string:
                    qubit = qubit2 - min_qubit
                    if char == 'Z':
                        value *= -1.0 if (ket & (1 << (nqubit - 1 - qubit))) else 1.0
                    elif char == 'X':
                        bra ^= (1 << (nqubit - 1 - qubit))
                    elif char == 'Y':
                        value *= -1.j if (ket & (1 << (nqubit - 1 - qubit))) else 1.j
                        bra ^= (1 << (nqubit - 1 - qubit))
                bra_index = kets2.get(bra, None)
                if bra_index is None: continue
                H[bra_index, ket_index] += value
        
        return np.array(H, dtype=dtype)
                    
                 
        

        
        
    
