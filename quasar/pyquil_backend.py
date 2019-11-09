import numpy as np
from .backend import Backend

class PyquilBackend(Backend):

    def __init__(
        self,
        ):

        pass 
    
    @property
    def native_circuit_type(self):
        import pyquil
        return pyquil.Program
    
    @staticmethod
    def quasar_to_pyquil_angle(theta):
        return 2.0 * theta
        
    @staticmethod
    def pyquil_to_quasar_angle(theta):
        return 0.5 * theta    
    
    def build_native_circuit(
        self,
        circuit,
        bit_reversal=False,
        min_qubit=None,
        nqubit=None,
        ):

        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit

        import pyquil
        circuit_native = pyquil.Program()
        for key, gate in circuit.gates.items():
            times, qubits = key
            if gate.nqubit == 1:
                qubit = qubits[0] - min_qubit
                if bit_reversal:
                    qubit = nqubit - qubit - 1
                if gate.name == 'I':
                    circuit_native += pyquil.gates.I(qubit)
                elif gate.name == 'X':
                    circuit_native += pyquil.gates.X(qubit)
                elif gate.name == 'Y':
                    circuit_native += pyquil.gates.Y(qubit)
                elif gate.name == 'Z':
                    circuit_native += pyquil.gates.Z(qubit)
                elif gate.name == 'H':
                    circuit_native += pyquil.gates.H(qubit)
                elif gate.name == 'S':
                    circuit_native += pyquil.gates.S(qubit)
                elif gate.name == 'T':
                    circuit_native += pyquil.gates.T(qubit)
                elif gate.name == 'Rx':
                    circuit_native += pyquil.gates.RX(self.quasar_to_pyquil_angle(gate.parameters['theta']), qubit) 
                elif gate.name == 'Ry':
                    circuit_native += pyquil.gates.RY(self.quasar_to_pyquil_angle(gate.parameters['theta']), qubit)  
                elif gate.name == 'Rz':
                    circuit_native += pyquil.gates.RZ(self.quasar_to_pyquil_angle(gate.parameters['theta']), qubit) 
                else:
                    raise RuntimeError('Gate translation to pyquil not known: %s' % gate)
            elif gate.nqubit == 2:
                qubitA = qubits[0] - min_qubit
                qubitB = qubits[1] - min_qubit
                if bit_reversal:
                    qubitA = nqubit - qubitA - 1
                    qubitB = nqubit - qubitB - 1
                if gate.name == 'CNOT':
                    circuit_native += pyquil.gates.CNOT(qubitA, qubitB)
                elif gate.name == 'CX':
                    circuit_native += pyquil.gates.CNOT(qubitA, qubitB)
                elif gate.name == 'CY':
                    circuit_native += pyquil.gates.RZ(-np.pi/2,qubitB)
                    circuit_native += pyquil.gates.CNOT(qubitA, qubitB)
                    circuit_native += pyquil.gates.RZ(+ np.pi/2,qubitB)
                elif gate.name == 'CZ':
                    circuit_native += pyquil.gates.CZ(qubitA, qubitB)
                elif gate.name == 'SWAP':
                    circuit_native += pyquil.gates.SWAP(qubitA, qubitB)
                else:
                    raise RuntimeError('Gate translation to pyquil not known: %s' % gate)
            else:
                raise RuntimeError('Cannot emit qiskit for N > 2')
                
        return circuit_native

    def build_native_circuit_measurement(
        self,
        circuit,
        ):
        # Note: Might be unecessary because there is a run_and_measure() function in simulation.
        circuit_native = self.build_native_circuit(circuit)
        circuit_native.measure_all()
        
        return circuit_native

class PyquilSimulatorBackend(PyquilBackend):

    def __init__(
        self,
        ):

        import pyquil        
        self.wavefunction_backend = pyquil.api.WavefunctionSimulator()
        # self.measurement_backend = pyquil.get_qc(backend_name)
        
    def __str__(self):
        return 'Pyquil Simulator Backend'
    
    @property
    def summary_str(self):
        return 'Pyquil Simulator Backend'

    @property
    def has_statevector(self):
        return True

    @property
    def has_measurement(self):
        return True

    def run_statevector(
        self,
        circuit,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        **kwargs):

        # TODO: Input statevector

        import pyquil
        circuit_native = self.build_native_circuit(circuit, bit_reversal=True)
        statevector = self.wavefunction_backend.wavefunction(circuit_native).amplitudes
        statevector = np.array(statevector, dtype=dtype)
        return statevector
        
    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        ):
    
        raise NotImplementedError
