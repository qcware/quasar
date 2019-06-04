from .run import run_pauli_expectation
from .resolution import resolve_and_emit_quasar_circuit
from .pauli import Pauli
import numpy as np

def run_observable_expectation_value(
    backend,
    circuit,
    pauli,
    nmeasurement=None,
    **kwargs):

    # No dropthrough - always need quasar.Circuit to manipulate
    circuit = resolve_and_emit_quasar_circuit(circuit).copy()

    return run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli).real # TODO: do we need this to be real

def run_observable_expectation_value_gradient(
    backend,
    circuit,
    pauli,
    nmeasurement=None,
    param_indices=None,
    **kwargs):

    # No dropthrough - always need quasar.Circuit to manipulate
    circuit = resolve_and_emit_quasar_circuit(circuit).copy()
    param_values = circuit.param_values

    # Default to taking the gradient with respect to all params
    if param_indices is None:
        param_indices = tuple(range(circuit.nparam))

    # Check that the gradient formula is known for these parameters (i.e., Rx, Ry, Rz gates)
    param_keys = circuit.param_keys
    for param_index in param_indices:
        key = param_keys[param_index]
        time, qubits, name = key
        gate = circuit.gates[(time, qubits)]
        if not gate.name in ('Rx', 'Ry', 'Rz'): 
            raise RuntimeError('Unknown gradient rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)

    # Evaluate the gradient
    G = np.zeros((len(param_indices),))
    for I, param_index in enumerate(param_indices):
        param_values2 = param_values.copy()
        param_values2[param_index] += np.pi / 4.0
        circuit.set_param_values(param_values2)
        Ep = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli)
        param_values2 = param_values.copy()
        param_values2[param_index] -= np.pi / 4.0
        circuit.set_param_values(param_values2)
        Em = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli)
        G[I] = (Ep - Em).real # TODO: do we need this to be real?

    return G
        
def run_observable_expectation_value_hessian(
    backend,
    circuit,
    pauli,
    nmeasurement=None,
    param_indices1=None,
    param_indices2=None,
    **kwargs):

    # No dropthrough - always need quasar.Circuit to manipulate
    circuit = resolve_and_emit_quasar_circuit(circuit).copy()
    param_values = circuit.param_values

    # Default to taking the Hessian with respect to all params
    if param_indices1 is None:
        param_indices1 = tuple(range(circuit.nparam))
    if param_indices2 is None:
        param_indices2 = tuple(range(circuit.nparam))

    # Check that the gradient formula is known for these parameters (i.e., Rx, Ry, Rz gates)
    param_keys = circuit.param_keys
    for param_index in param_indices1:
        key = param_keys[param_index]
        time, qubits, name = key
        gate = circuit.gates[(time, qubits)]
        if not gate.name in ('Rx', 'Ry', 'Rz'): 
            raise RuntimeError('Unknown gradient rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)
    for param_index in param_indices2:
        key = param_keys[param_index]
        time, qubits, name = key
        gate = circuit.gates[(time, qubits)]
        if not gate.name in ('Rx', 'Ry', 'Rz'): 
            raise RuntimeError('Unknown gradient rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)

    # Evaluate the Hessian
    H = np.zeros((len(param_indices1), len(param_indices2)))
    for I1, param_index1 in enumerate(param_indices1):
        for I2, param_index2 in enumerate(param_indices2):
            symmetric = param_index1 in param_indices2 and param_index2 in param_indices1
            if symmetric and param_index1 > param_index2: continue
            param_values2 = param_values.copy()
            param_values2[param_index1] += np.pi / 4.0
            param_values2[param_index2] += np.pi / 4.0
            circuit.set_param_values(param_values2)
            Epp = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli)
            param_values2 = param_values.copy()
            param_values2[param_index1] += np.pi / 4.0
            param_values2[param_index2] -= np.pi / 4.0
            circuit.set_param_values(param_values2)
            Epm = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli)
            param_values2 = param_values.copy()
            param_values2[param_index1] -= np.pi / 4.0
            param_values2[param_index2] += np.pi / 4.0
            circuit.set_param_values(param_values2)
            Emp = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli)
            param_values2 = param_values.copy()
            param_values2[param_index1] -= np.pi / 4.0
            param_values2[param_index2] -= np.pi / 4.0
            circuit.set_param_values(param_values2)
            Emm = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli)
            H[I1, I2] = (Epp - Epm - Emp + Emm).real # TODO: Do we need this to be real?
            if symmetric:
                H[I2, I1] = H[I1, I2]

    return H

def run_observable_expectation_value_hessian_selected(
    backend,
    circuit,
    pauli,
    param_index_pairs,
    nmeasurement=None,
    **kwargs):

    # No dropthrough - always need quasar.Circuit to manipulate
    circuit = resolve_and_emit_quasar_circuit(circuit).copy()
    param_values = circuit.param_values

    # Check that the gradient formula is known for these parameters (i.e., Rx, Ry, Rz gates)
    param_keys = circuit.param_keys
    for param_index1, param_index2 in param_index_pairs:
        key = param_keys[param_index1]
        time, qubits, name = key
        gate = circuit.gates[(time, qubits)]
        if not gate.name in ('Rx', 'Ry', 'Rz'): 
            raise RuntimeError('Unknown gradient rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)
    for param_index1, param_index2 in param_index_pairs:
        key = param_keys[param_index2]
        time, qubits, name = key
        gate = circuit.gates[(time, qubits)]
        if not gate.name in ('Rx', 'Ry', 'Rz'): 
            raise RuntimeError('Unknown gradient rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)

    # Evaluate the Hessian
    H = np.zeros((len(param_index_pairs),))
    for I, param_index_pair in enumerate(param_index_pairs):
        param_index1, param_index2 = param_index_pair
        symmetric = (param_index2, param_index1) in param_index_pairs
        if symmetric and param_index1 > param_index2: continue
        param_values2 = param_values.copy()
        param_values2[param_index1] += np.pi / 4.0
        param_values2[param_index2] += np.pi / 4.0
        circuit.set_param_values(param_values2)
        Epp = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli)
        param_values2 = param_values.copy()
        param_values2[param_index1] += np.pi / 4.0
        param_values2[param_index2] -= np.pi / 4.0
        circuit.set_param_values(param_values2)
        Epm = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli)
        param_values2 = param_values.copy()
        param_values2[param_index1] -= np.pi / 4.0
        param_values2[param_index2] += np.pi / 4.0
        circuit.set_param_values(param_values2)
        Emp = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli)
        param_values2 = param_values.copy()
        param_values2[param_index1] -= np.pi / 4.0
        param_values2[param_index2] -= np.pi / 4.0
        circuit.set_param_values(param_values2)
        Emm = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli)
        H[I] = (Epp - Epm - Emp + Emm).real # TODO: Do we need this to be real?
        if symmetric:
            H[param_index_pairs.index((param_index2, param_index1))] = H[I]

    return H

def run_observable_expectation_value_gradient_pauli_contraction(
    backend,
    circuit,
    pauli,
    param_coefs,
    nmeasurement=None,
    param_indices=None,
    **kwargs):
    
    # No dropthrough - always need quasar.Circuit to manipulate
    circuit = resolve_and_emit_quasar_circuit(circuit).copy()
    param_values = circuit.param_values

    # Default to taking the gradient with respect to all params
    if param_indices is None:
        param_indices = tuple(range(circuit.nparam))

    # Check that the gradient formula is known for these parameters (i.e., Rx, Ry, Rz gates)
    param_keys = circuit.param_keys
    for param_index in param_indices:
        key = param_keys[param_index]
        time, qubits, name = key
        gate = circuit.gates[(time, qubits)]
        if not gate.name in ('Rx', 'Ry', 'Rz'): 
            raise RuntimeError('Unknown gradient rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)

    # Check that the contraction makes sense
    if len(param_coefs) != len(param_indices): raise RuntimeError('len(param_coefs) != len(param_indices)')
    
    # Evaluate the contracted gradient
    pauli_G = Pauli.zeros_like(pauli)
    for I, param_index in enumerate(param_indices):
        param_values2 = param_values.copy()
        param_values2[param_index] += np.pi / 4.0
        circuit.set_param_values(param_values2)
        paulip = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs)
        param_values2 = param_values.copy()
        param_values2[param_index] -= np.pi / 4.0
        circuit.set_param_values(param_values2)
        paulim = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs)
        pauli_G += param_coefs[I] * (paulip - paulim)

    return pauli_G
