import numpy as np
from .circuit import Circuit

class VariationalObservable(object):

    def __init__(
        self,
        backend,
        reference_weights,
        reference_circuits,
        entangler_circuit,
        entangler_parameter_group,
        pauli,
        nmeasurement=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ):

        self.backend = backend
        self.reference_weights = reference_weights
        self.reference_circuits = reference_circuits
        self.entangler_circuit = entangler_circuit
        self.entangler_parameter_group = entangler_parameter_group
        self.pauli = pauli
        self.nmeasurement = nmeasurement
        self.min_qubit = min_qubit
        self.nqubit = nqubit
        self.dtype = dtype
    
    @property
    def nparameter(self):
        return self.entangler_parameter_group.nparameter

    def run_observable(
        self,
        parameters,
        ):

        raw_parameters = self.entangler_parameter_group.compute_raw(parameters)
        entangler_circuit = self.entangler_circuit.copy()
        entangler_circuit.set_parameter_values(raw_parameters)

        E = 0.0
        for reference_weight, reference_circuit in zip(self.reference_weights, self.reference_circuits):
            circuit = Circuit.join_in_time([reference_circuit, entangler_circuit])
            E += reference_weight * self.backend.run_pauli_expectation_value(
                circuit=circuit,
                pauli=self.pauli,
                nmeasurement=self.nmeasurement,
                min_qubit=self.min_qubit,
                nqubit=self.nqubit,
                dtype=self.dtype,
                ).real

        return E
                
    def run_observable_gradient(
        self,
        parameters,
        ):

        raw_parameters = self.entangler_parameter_group.compute_raw(parameters)
        entangler_circuit = self.entangler_circuit.copy()
        entangler_circuit.set_parameter_values(raw_parameters)

        G = np.zeros((len(raw_parameters),), dtype=np.float64)
        for reference_weight, reference_circuit in zip(self.reference_weights, self.reference_circuits):
            circuit = Circuit.join_in_time([reference_circuit, entangler_circuit])
            G += reference_weight * self.backend.run_pauli_expectation_value_gradient(
                circuit=circuit,
                pauli=self.pauli,
                nmeasurement=self.nmeasurement,
                min_qubit=self.min_qubit,
                nqubit=self.nqubit,
                dtype=self.dtype,
                parameter_indices=list(range(reference_circuit.nparameter, reference_circuit.nparameter + entangler_circuit.nparameter)),
                ).real

        return self.entangler_parameter_group.compute_chain_rule1(parameters, G)

