import numpy as np
from .algebra import Algebra

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


    def has_run_statevector(
        self,
        ):

        """ Does this Backend support run_statevector? 

        Returns:
            (bool) - True if run_statevector is supported else False.

        Backend subclasses should OVERLOAD this method.
        """ 
        return NotImplementedError

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

        return NotImplementedError

    def run_statevector(
        self,
        circuit,
        statevector=None,   
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        return NotImplementedError

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
        nmeasurement=1000,
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

        return Algebra.sample_measurements_from_probabilities(
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

        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit

        statevector = self.run_statevector(
            circuit=circuit,
            statevector=statevector,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            **kwargs)

        pauli_expectation = Pauli.zero()
        for string in pauli.keys():
            pauli2 = Pauli.zero()
            pauli2[string] = 1.0
            statevector2 = pauli2.compute_hilbert_matrix_vector_product(
                statevector=statevector,
                dtype=dtype,
                min_qubit=min_qubit,
                nqubit=nqubit,
                )
            pauli_expectation[string] = np.sum(statevector.conj() * statevector2)

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

        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit

        statevector = self.run_statevector(
            circuit=circuit,
            statevector=statevector,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            **kwargs)

        statevector2 = pauli.compute_hilbert_matrix_vector_product(
            statevector=statevector,
            dtype=dtype,
            min_qubit=min_qubit,
            nqubit=nqubit,
            )

        return np.sum(statevector.conj() * statevector2)
