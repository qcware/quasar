import numpy as np
from .algebra import Algebra

class Backend(object):

    """ Class Backend represents a physical or simulated quantum circuit
        resource. Backends must support `run_measurement`, from which many
        higher-order

        Class Backend declares an abstract set of API functions that must be
        overloaded by each specific Backend subclass, as well as some utility
        functions that are common to all Backend subclasses.
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
        dtype=np.complex128,
        **kwargs):

        return NotImplementedError

    def run_unitary(
        self,
        circuit,
        dtype=np.complex128,
        **kwargs):

        U = np.zeros((2**circuit.nqubit,)*2, dtype=dtype)
        for i in range(2**circuit.nqubit):
            statevector = np.zeros((2**circuit.nqubit,), dtype=dtype)
            statevector[i] = 1.0
            U[:, i] = self.run_statevector(circuit, statevector=statevector, dtype=dtype, **kwargs)

        return U

    def run_density_matrix(
        self,
        circuit,
        statevector=None,
        dtype=np.complex128,
        **kwargs):

        statevector = self.run_statevector(
            circuit=circuit,
            statevector=statevector,
            dtype=dtype,
            **kwargs)

        return np.outer(statevector, statevector.conj())

    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        statevector=None,
        dtype=np.complex128,
        **kwargs):

        statevector = self.run_statevector(
            circuit=circuit,
            statevector=statevector,
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
        **kwargs):
        
        return NotImplementedError

    def run_pauli_expectation_value(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        statevector=None,
        **kwargs):

        return NotImplementedError

    def run_pauli_expectation_value_gradient(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        statevector=None,
        parameter_indices=None,
        **kwargs):

        return NotImplementedError

    def run_pauli_expectation_value_hessian(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        statevector=None,
        parameter_pair_indices=None,
        **kwargs):

        return NotImplementedError
