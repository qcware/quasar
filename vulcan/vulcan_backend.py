import quasar
import numpy as np
from . import vulcan_plugin as vulcan

class VulcanSimulatorBackend(quasar.Backend):

    vulcan_dtype_class = {
        np.float32 : vulcan.float32,
        np.float64 : vulcan.float64,
        np.complex64 : vulcan.complex64,
        np.complex128 : vulcan.complex128,
    }

    vulcan_Gate_class = {
        np.float32 : vulcan.Gate_float32,
        np.float64 : vulcan.Gate_float64,
        np.complex64 : vulcan.Gate_complex64,
        np.complex128 : vulcan.Gate_complex128,
    }

    vulcan_Circuit_class = {
        np.float32 : vulcan.Circuit_float32,
        np.float64 : vulcan.Circuit_float64,
        np.complex64 : vulcan.Circuit_complex64,
        np.complex128 : vulcan.Circuit_complex128,
    }

    vulcan_Pauli_class = {
        np.float32 : vulcan.Pauli_float32,
        np.float64 : vulcan.Pauli_float64,
        np.complex64 : vulcan.Pauli_complex64,
        np.complex128 : vulcan.Pauli_complex128,
    }

    vulcan_run_statevector_method = {
        np.float32 : vulcan.run_statevector_float32,
        np.float64 : vulcan.run_statevector_float64,
        np.complex64 : vulcan.run_statevector_complex64,
        np.complex128 : vulcan.run_statevector_complex128,
    }

    vulcan_run_pauli_sigma_method = {
        np.float32 : vulcan.run_pauli_sigma_float32,
        np.float64 : vulcan.run_pauli_sigma_float64,
        np.complex64 : vulcan.run_pauli_sigma_complex64,
        np.complex128 : vulcan.run_pauli_sigma_complex128,
    }

    vulcan_run_pauli_expectation_value_method = {
        np.float32 : vulcan.run_pauli_expectation_value_float32,
        np.float64 : vulcan.run_pauli_expectation_value_float64,
        np.complex64 : vulcan.run_pauli_expectation_value_complex64,
        np.complex128 : vulcan.run_pauli_expectation_value_complex128,
    }

    vulcan_run_pauli_expectation_value_gradient_method = {
        np.float32 : vulcan.run_pauli_expectation_value_gradient_float32,
        np.float64 : vulcan.run_pauli_expectation_value_gradient_float64,
        np.complex64 : vulcan.run_pauli_expectation_value_gradient_complex64,
        np.complex128 : vulcan.run_pauli_expectation_value_gradient_complex128,
    }

    @staticmethod
    def vulcan_gate(
        gate,
        dtype=np.complex128,
        ):
    
        return VulcanSimulatorBackend.vulcan_Gate_class[dtype](
            gate.nqubit,    
            gate.name,
            [VulcanSimulatorBackend.vulcan_dtype_class[dtype](_.real, _.imag) for _ in gate.operator.ravel()],
            )
    
    @staticmethod
    def vulcan_circuit(
        circuit,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ):

        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit
    
        gates = []
        qubits = []
        for key, gate in circuit.gates.items():
            times, qubits2 = key
            qubits.append(tuple(_ - min_qubit for _ in qubits2))
            gates.append(VulcanSimulatorBackend.vulcan_gate(gate, dtype=dtype))
    
        circuit2 = VulcanSimulatorBackend.vulcan_Circuit_class[dtype](
            nqubit,
            gates,
            qubits,
            )
    
        circuit2 = circuit2.bit_reversal() # quasar/vulcan ordering
    
        return circuit2
    
    @staticmethod
    def vulcan_pauli(
        pauli,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ):
    
        min_qubit = pauli.min_qubit if min_qubit is None else min_qubit
        nqubit = pauli.nqubit if nqubit is None else nqubit
    
        types = []
        qubits = []
        values = []
        for string, value in pauli.items():
            qubits2 = string.qubits
            types2 = [0 if _ == 'X' else 1 if _ == 'Y' else 2 for _ in string.chars]
            types.append(types2)
            qubits.append(tuple(_ - min_qubit for _ in qubits2))
            values.append(VulcanSimulatorBackend.vulcan_dtype_class[dtype](value.real, value.imag))
    
        pauli2 = VulcanSimulatorBackend.vulcan_Pauli_class[dtype](
            nqubit,
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
        statevector=None, # TODO 
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ):
    
        circuit2 = VulcanSimulatorBackend.vulcan_circuit(
            circuit=circuit, 
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            )
    
        return VulcanSimulatorBackend.vulcan_run_statevector_method[dtype](
            circuit2,
            )
    
    def run_pauli_sigma(
        self,
        pauli,
        statevector,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ):
    
        pauli2 = VulcanSimulatorBackend.vulcan_pauli(
            pauli=pauli,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            )
    
        return VulcanSimulatorBackend.vulcan_run_pauli_sigma_method[dtype](
            pauli2,
            statevector,
            )
            
    def run_pauli_expectation_value(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        statevector=None, # TODO
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ): 

        if nmeasurement is not None: 
            raise NotImplementedError
             
        circuit2 = VulcanSimulatorBackend.vulcan_circuit(
            circuit=circuit, 
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            )
        pauli2 = VulcanSimulatorBackend.vulcan_pauli(
            pauli=pauli,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            )
    
        return VulcanSimulatorBackend.vulcan_run_pauli_expectation_value_method[dtype](
            circuit2,
            pauli2,
            )
            
    def run_pauli_expectation_value_gradient(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        statevector=None, # TODO
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        parameter_indices=None,
        ): 
             
        if nmeasurement is not None: 
            raise NotImplementedError

        # Default to taking the gradient with respect to all parameters
        if parameter_indices is None:
            parameter_indices = list(range(circuit.nparameter))

        # Check that the gradient formula is known for these parameters (i.e., Rx, Ry, Rz gates)
        parameter_keys = circuit.parameter_keys
        for parameter_index in parameter_indices:
            key = parameter_keys[parameter_index]
            times, qubits, key2 = key
            gate = circuit.gates[(times, qubits)]
            if not gate.name in ('Rx', 'Ry', 'Rz'): 
                raise RuntimeError('Unknown gradient rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)

        circuit2 = VulcanSimulatorBackend.vulcan_circuit(
            circuit=circuit, 
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            )
        pauli2 = VulcanSimulatorBackend.vulcan_pauli(
            pauli=pauli,
            min_qubit=min_qubit,
            nqubit=nqubit,
            dtype=dtype,
            )
    
        return VulcanSimulatorBackend.vulcan_run_pauli_expectation_value_gradient_method[dtype](
            circuit2,
            pauli2,
            )[parameter_indices]
