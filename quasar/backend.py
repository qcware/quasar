import collections

import numpy as np
import sortedcontainers
from itertools import repeat
from .algebra import Algebra
from .circuit import Circuit, Gate
from .pauli import Pauli, PauliExpectation, PauliString


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
        Backend subclasses should OVERLOAD this method.

        Returns:
            str: 1-line string representation of this Backend
        
        """ 
        raise NotImplementedError

    @property
    def summary_str(self):
        """ A more-extensive string representation of this Backend, optionally
        including current hardware state.

        Backend subclasses should OVERLOAD this method.

        Returns:
            str: multiline string representation of this Backend
    
        """ 
        raise NotImplementedError

    @property
    def has_run_statevector(
        self,
        ):

        """ Does this Backend support run_statevector? 

        Backend subclasses should OVERLOAD this method.

        Returns:
            bool: True if run_statevector is supported else False.

        """ 
        raise NotImplementedError

    @property
    def has_statevector_input(
        self,
        ):

        """ Does this Backend allow statevector to be passed as input argument
        to various run methods?

        Backend subclasses should OVERLOAD this method.

        Returns:
            bool: `True` if statevector input arguments can be supplied else `False`.

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

        """ Run a circuit using statevector simulation.

        Backend subclasses should OVERLOAD this method.

        Args:
            circuit (Circuit): the circuit to run
            statevector (``np.ndarray``, shape `2**N`): input statevector (default - `None`)
            min_qubit (int): the minimum occupied qubit index (default - `None`)
            nqubit (int): number of qubits (default - `None`)
            dtype (real or complex dtype): the dtype to perform the computation at. The gate operator
                will be cast to this dtype. Note that using real dtypes (`float64` 
                or `float32`) can reduce storage and runtime, but the imaginary parts
                of the input wfn and all gate unitary operators will be discarded
                without checking. In these cases, the user is responsible for
                ensuring that the circuit works on `O(2^N)` rather than `U(2^N)`
                and that the output is valid. (default - `np.complex128`)

        Returns:
            ``np.ndarray``, shape `2**N`: full output statevector
        """

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

        """ Run a circuit using measurement-based computing. 

        Args:
            circuit (Circuit): the circuit to run
            nmeasurement (int or None): number of measurements to sample. If
                None, infinite sampling is assumed and the probabilities are
                directly returned in ProbabilityHistogram format (default - `None`)
            statevector (``np.ndarray``, shape `2**N`): input statevector (default - `None`) 
            min_qubit (int): the minimum occupied qubit index (default - `None`)
            nqubit (int): number of qubits (default - `None`)
            dtype (real or complex dtype): the dtype to perform the computation at. The gate operator
                will be cast to this dtype. Note that using real dtypes (`float64` 
                or `float32`) can reduce storage and runtime, but the imaginary parts
                of the input wfn and all gate unitary operators will be discarded
                without checking. In these cases, the user is responsible for
                ensuring that the circuit works on `O(2^N)` rather than `U(2^N)`
                and that the output is valid. (default - `np.complex128`)

        Returns:
            ProbabilityHistogram: histogram of projective measurements taken while running
            the circuit
        """

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

    def run_measurements(
        self,
        circuits,
        nmeasurement=None
    ):

        """ Run a sequence of circuits using measurement-based computing. 

        Returns a sequence of ProbabilityHistogram objects.

        As the circuits could be very different, statevector input or min_qubit/nqubit
        are not supported.  The default behaviour is to simply run the circuits in
        sequence, but on some hardware backends a speed advantage can be gained by
        batching circuits into jobs.

        Args:
            circuits (Sequence[Circuit]): the circuits to run
            nmeasurement (int or Sequence[int] or None): number of measurements to sample. If
                None, infinite sampling is assumed and the probabilities are
                directly returned in ProbabilityHistogram format (default - `None`)
                If an int, that number of measurements will be used for all circuits.  If a sequence,
                run_measurements returns the shortest of len(circuits) and len(nmeasurement).

        Returns:
            list[ProbabilityHistogram]: list of histograms of projective measurements taken while running
            the circuits
        """
        nmeasurements = repeat(nmeasurement) if (isinstance(nmeasurement, int) or (nmeasurement is None)) else nmeasurement
        return [self.run_measurement(circuit=circuit, nmeasurement=nmeasurement) for (circuit, nmeasurement) in zip(circuits, nmeasurements)]

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
        """ Compute the total observable expectation value of a Pauli-sparse Hermitian 
        operator using the statevector generated by the circuit at the current parameter set.

        Args:
            circuit (Circuit): the circuit for which to calculate the expectation value.
            pauli (Pauli): the Hermitian operator supplied in sparse Pauli form.
            nmeasurement (int): number of measurements to sample. If
                default value `None`, assume infinite sampling. 
            statevector (``np.ndarray`` of shape 2**N, complex dtype): the statevector 
                generated by the circuit. If default value `None`, all qubits in the statevector 
                set equal to `0.`
            min_qubit (int): the minimum occupied qubit index (default - `None`). 
            nqubit (int): the total number of qubit indices in the circuit, including empty qubit 
                indices (default - `None`).
            dtype (complex dtype): the dtype to perform the computation at. (default - ``np.complex128``).
            
        Returns: 
            float, complex dtype: total observable expectation value 

        """
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
        """ Compute the gradient, with respect to a circuit's current
        parameter set, of the total observable expectation value of a
        Pauli-sparse Hermitianoperatorusing the statevector generated by the 
        circuit at the current parameter set.  
        
        Args:
            circuit (Circuit): the circuit for which to calculate the expectation value.
            pauli (Pauli): the Hermitian operator supplied in sparse Pauli form.
            nmeasurement (int): number of measurements to sample. If
                default value `None`, assume infinite sampling. 
            statevector (``np.ndarray`` of shape 2**N, complex dtype): the statevector 
                generated by the circuit. If default value `None`, all qubits in the statevector 
                set equal to `0.`
            min_qubit (int): the minimum occupied qubit index (default - `None`). 
            nqubit (int): the total number of qubit indices in the circuit, including empty qubit 
                indices (default - `None`).
            dtype (complex dtype): the dtype to perform the computation at. (default - ``np.complex128``).
            parameter_indices (list of ints): take the gradient with respect to these parameter 
                indices. If default value `None`, take the gradient with respect to all parameters.
        
        Returns:
            ``np.ndarray`` with shape equal to **parameter_indices**,
            complex dtype: gradient of total observable expectation value with respect
            to the circuit's current parameter set.
    
        """
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
            if not gate.name in ('Rx', 'Ry', 'Rz', 'G', 'PX'):
                raise RuntimeError('Unknown gradient rule: presently can only '
                        'differentiate Rx, Ry, Rz, G  gates: %s' % gate)

        # Evaluate the gradient by the parameter shift rule
        gradient = np.zeros((len(parameter_indices),), dtype=dtype)
        circuit2 = circuit.copy()
        pi_half_factor = 0.5 * (1.0 - 2**0.5)
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
            key = parameter_keys[parameter_index]
            times, qubits, key2 = key
            gate = circuit.gates[(times, qubits)]
            Ep_half = 0.0
            Em_half = 0.0
            # For Given's gate we need 4 term parameter shift rule.
            if gate.name in ('G', 'PX'):
                # + pi / 2
                parameter_values2 = parameter_values.copy()
                parameter_values2[parameter_index] += np.pi / 2.0
                circuit2.set_parameter_values(parameter_values2)
                Ep_half = self.run_pauli_expectation_value(
                    circuit=circuit2,
                    pauli=pauli,
                    nmeasurement=nmeasurement,
                    statevector=statevector,
                    min_qubit=min_qubit,
                    nqubit=nqubit,
                    dtype=dtype,
                    **kwargs)
                # - pi / 2
                parameter_values2 = parameter_values.copy()
                parameter_values2[parameter_index] -= np.pi / 2.0
                circuit2.set_parameter_values(parameter_values2)
                Em_half = self.run_pauli_expectation_value(
                    circuit=circuit2,
                    pauli=pauli,
                    nmeasurement=nmeasurement,
                    statevector=statevector,
                    min_qubit=min_qubit,
                    nqubit=nqubit,
                    dtype=dtype,
                    **kwargs)
                pi_half = (Ep_half - Em_half)
            # Assembly
            gradient[index] = Ep - Em + pi_half_factor * (Ep_half - Em_half)

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
                    
                 
        

        
        
    
