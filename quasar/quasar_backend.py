from .backend import Backend
from .circuit import Circuit

class QuasarSimulatorBackend(Backend):

    def __init__(
        self,
        ):
        pass

    def __str__(self):
        return 'Quasar Simulator Backend (Statevector)'

    @property
    def summary_str(self):
        s = ''
        s += 'Quasar: An Ultralite Quantum Circuit Simulator\n'
        s += '   By Rob Parrish (rob.parrish@qcware.com)    '
        return s

    @property
    def has_statevector(self):
        return True

    @property
    def has_measurement(self):
        return True

    @property
    def native_circuit_type(self):
        return Circuit

    def build_native_circuit(
        self,
        circuit,
        ):

        # Dropthrough
        if isinstance(circuit, self.native_circuit_type): return circuit

        # Can only convert quasar -> quasar
        if not isinstance(circuit, Circuit): raise RuntimeError('circuit must be Circuit type for build_native_circuit: %s' % (circuit))

    def build_quasar_circuit(
        self,
        native_circuit,
        ):

        # Dropthrough
        if isinstance(native_circuit, self.native_circuit_type): return circuit

        # Can only convert quasar -> quasar
        if not isinstance(native_circuit, Circuit): raise RuntimeError('circuit must be Circuit type for build_native_circuit: %s' % (native_circuit))

    def run_statevector(
        self,
        circuit,
        compressed=True,
        ):
        return (circuit.compressed() if compressed else circuit).simulate()

    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        compressed=True,
        ):
        return (circuit.compressed() if compressed else circuit).measure(nmeasurement)


