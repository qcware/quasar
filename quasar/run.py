from .circuit import Circuit
from .resolution import resolve_and_emit_quasar_circuit
    
def build_native_circuit(
    backend,
    circuit,
    ):

    if not isinstance(circuit, (backend.native_circuit_type, Circuit)):
        circuit = resolve_and_emit_quasar_circuit(circuit)

    return backend.build_native_circuit(circuit)

def run_measurement(
    backend,
    circuit,
    nmeasurement=1000,
    **kwargs):

    if not isinstance(circuit, (backend.native_circuit_type, Circuit)):
        circuit = resolve_and_emit_quasar_circuit(circuit)

    return backend.run_measurement(circuit, nmeasurement, **kwargs)

def run_statevector(
    backend,
    circuit,
    **kwargs):

    if not isinstance(circuit, (backend.native_circuit_type, Circuit)):
        circuit = resolve_and_emit_quasar_circuit(circuit)

    return backend.run_statevector(circuit, **kwargs)

def run_pauli_expectation(
    backend,
    circuit,
    pauli,
    nmeasurement=None,
    **kwargs):

    if not isinstance(circuit, (backend.native_circuit_type, Circuit)):
        circuit = resolve_and_emit_quasar_circuit(circuit)

    return backend.run_pauli_expectation(circuit, pauli, nmeasurement, **kwargs)
