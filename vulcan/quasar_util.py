from . import vulcan_plugin as vulcan

def vulcan_gate(
    gate,
    ):

    return vulcan.Gate_complex128(
        gate.nqubit,    
        gate.name,
        [vulcan.complex128(_.real, _.imag) for _ in gate.operator.ravel()],
        )
        

def vulcan_circuit(
    circuit,
    ):

    gates = []
    qubits = []
    for key, gate in circuit.gates.items():
        times, qubits2 = key
        qubits.append(qubits2)
        gates.append(vulcan_gate(gate))

    circuit2 = vulcan.Circuit_complex128(
        circuit.nqubit,
        gates,
        qubits,
        )

    circuit2 = circuit2.bit_reversal() # quasar/vulcan ordering

    return circuit2

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

def run_statevector(
    circuit,
    ):

    circuit2 = vulcan_circuit(circuit)

    return vulcan.run_statevector_complex128(circuit2)

def run_pauli_sigma(
    pauli,
    statevector,
    ):

    pauli2 = vulcan_pauli(pauli)

    return vulcan.run_pauli_sigma_complex128(pauli2, statevector)
        
def run_pauli_expectation_value(
    circuit,
    pauli,
    ): 
         

    circuit2 = vulcan_circuit(circuit)
    pauli2 = vulcan_pauli(pauli)

    return vulcan.run_pauli_expectation_value_complex128(circuit2, pauli2)
        
def run_pauli_expectation_value_gradient(
    circuit,
    pauli,
    ): 
         

    circuit2 = vulcan_circuit(circuit)
    pauli2 = vulcan_pauli(pauli)

    return vulcan.run_pauli_expectation_value_gradient_complex128(circuit2, pauli2)
