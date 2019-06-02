from .backend import Backend

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
        return False 

    def build_native_circuit(
        self,
        circuit,
        ):
        return circuit.copy()

    def run_statevector(
        self,
        circuit,
        compressed=True,
        ):
        return (circuit.compressed() if compressed else circuit).simulate()


