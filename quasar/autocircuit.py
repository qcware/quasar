from .circuit import Circuit
from .index_allocator import IndexAllocator
from .autogate import AutoGate
from copy import deepcopy

class AutoCircuit(Circuit):
    r'''
    Adds additional functionality to Circuit, including automatically hanlding ancillas.
    '''

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
        copy=False,
        name=None,
        ascii_symbols=None,
        ):

        if isinstance(gate, AutoGate) and gate.logical_n <= gate.circuit.nqubit:
            ancillas = self.get_ancillas(gate.circuit.nqubit_sparse - gate.logical_n)
            
            if isinstance(qubits, int):
                qubits = tuple([qubits])

            super().add_gate(        
                gate,
                tuple(ancillas[::-1]) + qubits,
                times=None, 
                time_start=None, 
                time_placement='early',
                copy=copy,
                name=None,
                ascii_symbols=None,
            )     
            
            self.free_ancillas(ancillas)
        
        elif isinstance(gate, AutoGate) and gate.logical_n > gate.circuit.nqubit:
            raise RuntimeError("Logical n exceeds real n", gate)
            
        else:
            super().add_gate(        
                gate,
                qubits,
                times=None, 
                time_start=None, 
                time_placement='early',
                copy=copy,
                name=None,
                ascii_symbols=None,
            )
            
        return self

    def add_gates(
        self,
        circuit,
        qubits,
        copy=False,
        ):

        for timeslice in circuit.gates:

            gate = circuit.gates[timeslice]

            self.add_gate(gate, qubits[:gate.nqubit-1])

        return self


    def ascii_diagram(
        self,
        time_lines='both',
        show_ancillas=False,
        ):

        """ Return a simple ASCII string diagram of the circuit.

        Params:
            time_lines (str) - specification of time lines:
                "both" - time lines on top and bottom (default)
                "top" - time lines on top 
                "bottom" - time lines on bottom
                "neither" - no time lines
        Returns:
            (str) - the ASCII string diagram
        """

        # Left side states
        Wd = max(len(str(_)) for _ in range(self.min_qubit, self.max_qubit+1)) if self.nqubit else 0
        lstick = '%-*s : |\n' % (1+Wd, 'T')
        for qubit in range(self.min_qubit, self.max_qubit+1): 
            lstick += '%*s\n' % (5+Wd, ' ')
            lstick += 'q%-*d : -\n' % (Wd, qubit)

        # Build moment strings
        moments = []
        for time in range(self.min_time, self.max_time+1):
            moments.append(self.ascii_diagram_time(
                time=time,
                adjust_for_time=False if time_lines=='neither' else True,
                ))

        # Unite strings
        lines = lstick.split('\n')
        for moment in moments:
            for i, tok in enumerate(moment.split('\n')):
                lines[i] += tok
        # Time on top and bottom
        lines.append(lines[0])

        # Adjust for time lines
        if time_lines == 'both':
            pass
        elif time_lines == 'top':
            lines = lines[:-2]
        elif time_lines == 'bottom':
            lines = lines[2:]
        elif time_lines == 'neither':
            lines = lines[2:-2]
        else:
            raise RuntimeError('Invalid time_lines argument: %s' % time_lines)

        if not show_ancillas:
            lines_sans_ancillas = []
            for index, line in enumerate(lines):
                if line[:2] != "q-" and lines[index-1][:2] != "q-":
                    lines_sans_ancillas.append(line)
            strval = '\n'.join(lines_sans_ancillas)
        else:
            strval = '\n'.join(lines)

        return strval


    def copy(self, copy_gates = True):
        copy = AutoCircuit()

        if copy_gates:
            for entry in self.gates:
                recipients = entry[1]
                gate = self.gates[entry]
                copy.add_gate(gate, recipients)

        copy.ancilla_pool = deepcopy(self.ancilla_pool)

        return copy

