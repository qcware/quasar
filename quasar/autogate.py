from .circuit import Gate, CompositeGate
from functools import reduce

from copy import deepcopy

class AutoGate(CompositeGate):
    r'''
    Adds additional functionality to CompositeGate, including automatically adding controls to a Gate,
    using a multi-toffoli circuit.
    '''

    @property
    def ntime(self):
        return 1
    @property
    def logical_n(self):
        return len(list(filter(lambda x: x >= 0, self.circuit.qubits)))


    def add_controls(self, num_of_controls, mtof_builder):

        if self.name == "MTOF":
            return mtof_builder(self.nqubit-1+num_of_controls)

        elif num_of_controls == 1 and reduce(lambda x,y: x and y, [self.circuit.gates[timeslice].nqubit == 1 for timeslice in self.circuit.gates]):

            controlled_circuit = self.circuit.copy(copy_gates=False)

            for slc in self.circuit.gates:

                gate = self.circuit.gates[slc]
                recipients = slc[1]
                
                if isinstance(gate, AutoControlledGate):

                    recipients = tuple(filter(lambda x: x >= 0, list(recipients)))
                    #print(recipients)

                    controlled_circuit.add_gate(    
                        gate=gate.uncontrolled_gate.add_controls(1+gate.num_of_controls, mtof_builder),
                        qubits= (self.nqubit, ) + recipients
                    )

                else:

                    cu_qubits = (self.nqubit, ) + recipients

                    controlled_circuit.add_controlled_gate(    
                        gate=gate,
                        controls=[True],
                        qubits= cu_qubits,
                        copy= False,
                    )

            return AutoControlledGate(
                controlled_circuit, deepcopy(self), num_of_controls, 
                name = "Controlled-"+self.name, ascii_symbols = self.ascii_symbols + ["@" for i in range(num_of_controls)])

        else: #else use an ancilla to control every other gate

            controlled_circuit = self.circuit.copy(copy_gates=False)

            ancilla = controlled_circuit.get_ancillas(1)[0]

            control_indices = [i for i in range(self.logical_n, self.logical_n + num_of_controls)]

            #initial m_tof
            
            mtof_instance = mtof_builder(num_of_controls)

            mtof_qubits = tuple(control_indices) + (ancilla, )

            controlled_circuit.add_gate(mtof_instance, mtof_qubits)

            #control every other gate with this ancilla
            for slc in self.circuit.gates:

                gate = self.circuit.gates[slc]
                recipients = slc[1]
                
                if isinstance(gate, AutoControlledGate):

                    recipients = tuple(filter(lambda x: x >= 0, list(recipients)))
                    #print(recipients)

                    controlled_circuit.add_gate(    
                        gate=gate.uncontrolled_gate.add_controls(1+gate.num_of_controls, mtof_builder),
                        qubits= (ancilla, ) + recipients
                    )

                else:

                    cu_qubits = (ancilla, ) + recipients

                    controlled_circuit.add_controlled_gate(    
                        gate=gate,
                        controls=[True],
                        qubits= cu_qubits,
                        copy= False,
                    )

            how_many_ancillas = len(list(filter(lambda x: x < 0, controlled_circuit.qubits)))
            ascii_symbols = ["anc" for i in range(how_many_ancillas)] + self.ascii_symbols + ["@" for i in range(num_of_controls)]



            #undo initial mtof

            controlled_circuit.add_gate(mtof_instance, mtof_qubits)
            controlled_circuit.free_ancillas([ancilla])



            return AutoControlledGate(
                controlled_circuit, deepcopy(self), num_of_controls, 
                name = "Controlled-"+self.name, ascii_symbols = ascii_symbols)



class AutoControlledGate(AutoGate):
    r'''
    Used to manage controlled AutoGates.
    '''
    def __init__(
        self,
        circuit,
        uncontrolled_gate,
        num_of_controls,
        name=None,
        ascii_symbols=None,
        ):

        self.circuit = circuit
        self.uncontrolled_gate = uncontrolled_gate
        self.num_of_controls = num_of_controls
        self.name = 'ControlledAutoG' if name is None else name
        self.ascii_symbols = ['CAG'] * self.circuit.nqubit if ascii_symbols is None else ascii_symbols


    @property
    def is_controled(self):
        return True

    def add_controls(self, num_of_controls, mtof_builder):
        return self.uncontrolled_gate.add_controls(self.num_of_controls + num_of_controls, mtof_builder)

