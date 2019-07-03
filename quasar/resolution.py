from .cirq_backend import CirqBackend
from .qiskit_backend import QiskitBackend
from .quasar_backend import QuasarSimulatorBackend
from .forest_backend import ForestBackend


def build_quasar_circuit(
    circuit,    
    ):

    backend = QuasarSimulatorBackend()
    if isinstance(circuit, backend.native_circuit_type):
        return backend.build_quasar_circuit(circuit)
    backend = CirqBackend()
    if isinstance(circuit, backend.native_circuit_type):
        return backend.build_quasar_circuit(circuit)
    backend = QiskitBackend()
    if isinstance(circuit, backend.native_circuit_type):
        return backend.build_quasar_circuit(circuit)
    backend = ForestBackend()
    if isinstance(circuit, backend.native_circuit_type):
        return backend.build_quasar_circuit(circuit)

    raise RuntimeError('Unknown circuit type: %s' % circuit)
