from .circuit import Circuit
from .indexallocator import IndexAllocator
from .autogate import AutoGate

class AutoCircuit(Circuit):
    
    def __init__(self):
        super().__init__()
        self.ancilla_pool = IndexAllocator(negative_convention=True)

    def from_quasar_circuit(circ):

        converted = AutoCircuit()
        for key in circ.gates:
            converted.add_gate(circ.gates[key], key[1], copy=False)
            
        return converted

    def get_ancillas(self, num):
        
        return [self.ancilla_pool.allocate() for i in range(0, num)]

    def free_ancillas(self,ancs):
        for anc in ancs:
            self.ancilla_pool.deallocate(anc)
            
            
    def add_gate(
        self,
        gate,
        qubits,
        times=None, 
        time_start=None, 
        time_placement='early',
        copy=True,
        name=None,
        ascii_symbols=None,
        ):

        if isinstance(gate, AutoGate) and gate.logical_n < gate.circuit.nqubit:
            ancillas = self.get_ancillas(gate.circuit.nqubit - gate.logical_n)
            
            if qubits == int:
                qubits = tuple(qubits)
            super().add_gate(        
                gate,
                tuple(ancillas[::-1]) + qubits,
                times=None, 
                time_start=None, 
                time_placement='early',
                copy=True,
                name=None,
                ascii_symbols=None,
            )     
            
            self.free_ancillas(ancillas)
        
        elif isinstance(gate, AutoGate) and gate.logical_n < gate.circuit.nqubit:
            raise RuntimeError("Logical n exceeds real n", gate)
            
        else:
            super().add_gate(        
                gate,
                qubits,
                times=None, 
                time_start=None, 
                time_placement='early',
                copy=True,
                name=None,
                ascii_symbols=None,
            )
            
        return self
