import quasar
from . import vulcan_plugin as vulcan

class VulcanSimulatorBackend(quasar.Backend):

    @staticmethod
    def vulcan_gate(
        gate,
        ):
    
        return vulcan.Gate_complex128(
            gate.nqubit,    
            gate.name,
            [vulcan.complex128(_.real, _.imag) for _ in gate.operator.ravel()],
            )
    
    @staticmethod
    def vulcan_circuit(
        circuit,
        ):
    
        gates = []
        qubits = []
        for key, gate in circuit.gates.items():
            times, qubits2 = key
            qubits.append(qubits2)
            gates.append(VulcanSimulatorBackend.vulcan_gate(gate))
    
        circuit2 = vulcan.Circuit_complex128(
            circuit.nqubit,
            gates,
            qubits,
            )
    
        circuit2 = circuit2.bit_reversal() # quasar/vulcan ordering
    
        return circuit2
    
    @staticmethod
    def vulcan_pauli(
        pauli,
        ):
    
        types = []
        qubits = []
        values = []
        for string, value in pauli.items():
            qubits2 = string.qubits
            types2 = [0 if _ == 'X' else 1 if _ == 'Y' else 2 for _ in string.chars]
            types.append(types2)
            qubits.append(qubits2)
            values.append(vulcan.complex128(value.real, value.imag))
    
        pauli2 = vulcan.Pauli_complex128(
            pauli.nqubit,
            types,
            qubits,
            values,
            )
    
        pauli2 = pauli2.bit_reversal() # quasar/vulcan ordering
    
        return pauli2
    
    def __init__(
        self,
        ):
        pass

    def __str__(self):
        return 'Vulcan Simulator Backend (Statevector)'

    @property
    def summary_str(self):
        s = ''
        s += 'Vulcan: An Ultrafast GPU Quantum Circuit Simulator\n'
        s += '   By Rob Parrish (rob.parrish@qcware.com)    '
        return s

    @property
    def has_run_statevector(self):
        return True

    @property
    def has_run_pauli_sigma(self):
        return True

    @property
    def has_statevector_input(self):
        return True

    def run_statevector(
        self,
        circuit,
        ):
    
        circuit2 = VulcanSimulatorBackend.vulcan_circuit(circuit)
    
        return vulcan.run_statevector_complex128(circuit2)
    
    def run_pauli_sigma(
        self,
        pauli,
        statevector,
        ):
    
        pauli2 = VulcanSimulatorBackend.vulcan_pauli(pauli)
    
        return vulcan.run_pauli_sigma_complex128(pauli2, statevector)
            
    def run_pauli_expectation_value(
        self,
        circuit,
        pauli,
        ): 
             
    
        circuit2 = VulcanSimulatorBackend.vulcan_circuit(circuit)
        pauli2 = VulcanSimulatorBackend.vulcan_pauli(pauli)
    
        return vulcan.run_pauli_expectation_value_complex128(circuit2, pauli2)
            
    def run_pauli_expectation_value_gradient(
        self,
        circuit,
        pauli,
        ): 
             
    
        circuit2 = VulcanSimulatorBackend.vulcan_circuit(circuit)
        pauli2 = VulcanSimulatorBackend.vulcan_pauli(pauli)
    
        return vulcan.run_pauli_expectation_value_gradient_complex128(circuit2, pauli2)
