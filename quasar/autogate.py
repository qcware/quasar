from .circuit import CompositeGate

class AutoGate(CompositeGate):
    @property
    def ntime(self):
        return 1
    @property
    def logical_n(self):
        return len(list(filter(lambda x: x >= 0, self.circuit.qubits)))
