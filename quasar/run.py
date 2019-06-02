def build_native_circuit(
    backend,
    circuit,
    **kwargs):

    return backend.build_native_circuit(circuit, **kwargs)

def run_measurement(
    backend,
    circuit,
    nmeasurement=1000,
    **kwargs):

    return backend.run_measurement(circuit, nmeasurement, **kwargs)

def run_statevector(
    backend,
    circuit,
    **kwargs):

    return backend.run_statevector(circuit, **kwargs)

def run_pauli_expectation(
    backend,
    circuit,
    pauli,
    nmeasurement=None,
    **kwargs):

    return backend.run_pauli_expectation(circuit, pauli, nmeasurement, **kwargs)
