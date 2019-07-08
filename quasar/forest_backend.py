import numpy as np
from .circuit import Circuit
from .backend import Backend
from .measurement import Ket, MeasurementResult

# => Forest <= #
class ForestBackend(Backend):

    def __init__(
        self,
        ):

        pass 
    
    @property
    def native_circuit_type(self):
        import pyquil
        return pyquil.Program
    
    @staticmethod
    def quasar_to_forest_angle(theta):
        return 2.0 * theta
        
    @staticmethod
    def forest_to_quasar_angle(theta):
        return 0.5 * theta    
    
    @staticmethod
    def forest_to_quasar_results(results_native, nmeasurement):
        # [Note] Forest's format: {'qubit1':[0,1,0,1,1,...],'qubit2':[1,1,0,0,1,...]}
        #        Target format: {'00': 52, '01':45, '10': 67, '11': 65}
        results = MeasurementResult()
        qubits = [k for k in results_native.keys()]
        values = [v for v in results_native.values()]
        
        for i in range(nmeasurement):
            k = ''
            for q in range(len(qubits)):
                k += str(values[q][i])
            
            if Ket(k) in results.keys():
                results[Ket(k)]+=1            
            else:
                results[Ket(k)]=1 
        
        return results
    
    
    def build_native_circuit(
        self,
        circuit,
        ):
        
        # Dropthrough
        if isinstance(circuit, self.native_circuit_type): return circuit
    
        # Can only convert quasar -> forest
        if not isinstance(circuit, Circuit): 
            raise RuntimeError('circuit must be Circuit type for build_native_circuit: %s' % (circuit))
        
        import pyquil
        import math
        circuit_native = pyquil.Program()
        
        for key in sorted(circuit.gates.keys()):
            T, qubits = key
            gate = circuit.gates[key]
            if gate.N == 1:
                qubit = qubits[0]
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
                    circuit_native += pyquil.gates.RX(self.quasar_to_forest_angle(gate.params['theta']), qubit) 
                elif gate.name == 'Ry':
                    circuit_native += pyquil.gates.RY(self.quasar_to_forest_angle(gate.params['theta']), qubit)  
                elif gate.name == 'Rz':
                    circuit_native += pyquil.gates.RZ(self.quasar_to_forest_angle(gate.params['theta']), qubit) 
                else:
                    raise RuntimeError('Gate translation to forest not known: %s' % gate)
            elif gate.N == 2:
                qubitA = qubits[0]
                qubitB = qubits[1]         
                if gate.name == 'CNOT':
                    circuit_native += pyquil.gates.CNOT(qubitA, qubitB)
                elif gate.name == 'CX':
                    circuit_native += pyquil.gates.CNOT(qubitA, qubitB)
                elif gate.name == 'CY':
                    circuit_native += pyquil.gates.RZ(-math.pi/2,qubitB)
                    circuit_native += pyquil.gates.CNOT(qubitA, qubitB)
                    circuit_native += pyquil.gates.RZ( math.pi/2,qubitB)
                elif gate.name == 'CZ':
                    circuit_native += pyquil.gates.CZ(qubitA, qubitB)
                elif gate.name == 'SWAP':
                    circuit_native += pyquil.gates.SWAP(qubitA, qubitB)
                else:
                    raise RuntimeError('Gate translation to forest not known: %s' % gate)
            else:
                raise RuntimeError('Cannot emit qiskit for N > 2')
                
        return circuit_native
    
    
    def build_quasar_circuit(
        self,
        native_circuit,
        ):

        # Dropthrough
        if isinstance(native_circuit, Circuit): return native_circuit
    
        # Can only convert forest -> quasar
        if not isinstance(native_circuit, self.native_circuit_type): 
            raise RuntimeError('native_circuit must be Forest type for build_native_circuit: %s' % (native_circuit))

        circuit = Circuit(N=len(native_circuit.get_qubits()))

        for gate in native_circuit:
            if len(gate.qubits) == 1:
                qubit = gate.qubits[0].index
                if gate.name == 'I':
                    circuit.I(qubit)
                elif gate.name == 'X':
                    circuit.X(qubit)
                elif gate.name == 'Y':
                    circuit.Y(qubit)
                elif gate.name == 'Z':
                    circuit.Z(qubit)
                elif gate.name == 'H':
                    circuit.H(qubit)
                elif gate.name == 'S':
                    circuit.S(qubit)
                elif gate.name == 'T':
                    circuit.T(qubit)   
                elif gate.name == 'RX':
                    circuit.Rx(qubit, theta=self.forest_to_quasar_angle(float(gate.params[0])))
                elif gate.name == 'RY':
                    circuit.Ry(qubit, theta=self.forest_to_quasar_angle(float(gate.params[0])))
                elif gate.name == 'RZ':
                    circuit.Rz(qubit, theta=self.forest_to_quasar_angle(float(gate.params[0])))
                else:
                    raise RuntimeError('Gate translation to quasar not known: %s' % gate)
            
            elif len(gate.qubits) == 2:
                qubitA = gate.qubits[0].index
                qubitB = gate.qubits[1].index
                if gate.name == 'CNOT':
                    circuit.CX(qubitA, qubitB)
                elif gate.name == 'CZ':
                    circuit.CZ(qubitA, qubitB)
                elif gate.name == 'SWAP':
                    circuit.SWAP(qubitA, qubitB)
                else:
                    raise RuntimeError('Gate translation to quasar not known: %s' % gate)
            
            elif len(gate.qubits) == 3:
                qubitA = gate.qubits[0].index
                qubitB = gate.qubits[1].index
                qubitC = gate.qubits[2].index
                if gate.name == 'CCNOT':
                    circuit.CCX(qubitA, qubitB, qubitC)
                elif gate.name == 'CSWAP':
                    circuit.CSWAP(qubitA, qubitB, qubitC)    
                else:
                    raise RuntimeError('Gate translation to quasar not known: %s' % gate)
            else:
                raise RuntimeError('Cannot translate qiskit for N > 2')

        return circuit
    
    
    def build_native_circuit_in_basis(
        self,
        circuit,
        basis,
        ):

        circuit = self.build_native_circuit(circuit)
    
        if len(basis) > len(circuit.get_qubits()): raise RuntimeError('len(basis) > circuit.N. Often implies pauli.N > circuit.N')
        
        import pyquil
        q = [A for A in range(len(circuit.get_qubits()))]
        basis_circuit = pyquil.Program()
        for A, char in enumerate(basis):
            if char == 'X': basis_circuit += pyquil.gates.H(q[A])
            elif char == 'Y': basis_circuit += pyquil.gates.Rx(self.quasar_to_forest_angle(-np.pi / 4.0))(q[A])
            elif char == 'Z': continue
            else: raise RuntimeError('Unknown basis: %s' % char)
        
        return circuit + basis_circuit
    
    def build_native_circuit_measurement(
        self,
        circuit,
        ):
        # Note: Might be unecessary
        circuit_native = self.build_native_circuit(circuit)
        idx_qubit = circuit_native.get_qubits()
        ro = circuit_native.declare('ro', memory_size=max(idx_qubit))
        circuit_native.measure_all()
        
        return circuit_native
        
    def build_native_circuit_unitary(
        self,
        circuit,
        qubit_setup=None
        ):
        """
        qubit_setup (np.ndarray of shape (circuit.N,) or None)
                - the initial setup of qubit array. If None, the reference setup
                  [0,0,0,0,0,...] will be used.
        """
        import pyquil
        
        if qubit_setup is None:
            qubit_setup = np.zeros((circuit.N,), dtype=np.int)
        else:
            qubit_setup = np.array(qubit_setup, dtype=np.int)
        
        circuit_setup = pyquil.Program()
        for i in range(len(qubit_setup)):
            if qubit_setup[i]==1:
                circuit_setup += pyquil.gates.X(i)
        
        circuit_native = self.build_native_circuit(circuit)
        
        return circuit_setup + circuit_native    


class ForestSimulatorBackend(ForestBackend):

    def __init__(
        self,
        backend_name
        ):

        import pyquil        
        self.wavefunction_backend = pyquil.api.WavefunctionSimulator()
        self.measurement_backend = pyquil.get_qc(backend_name)
        
    def __str__(self):
        return 'Forest Simulator Backend'
    
    @property
    def summary_str(self):
        return 'Forest Simulator Backend'

    @property
    def has_statevector(self):
        return True

    @property
    def has_measurement(self):
        return True

    def run_statevector(
        self,
        circuit,
        ):

        import pyquil
        circuit_native = self.build_native_circuit(circuit)
        wfn_native = self.wavefunction_backend.wavefunction(circuit_native).amplitudes
        wfn = self.statevector_bit_reversal_permutation(wfn_native)

        return wfn

        
    def run_measurement(
        self,
        circuit,
        nmeasurement,
        ):
    
        import pyquil
        circuit_native = self.build_native_circuit(circuit)
        results_native = self.measurement_backend.run_and_measure(circuit_native, nmeasurement)
        results = self.forest_to_quasar_results(results_native, nmeasurement)

        return results

        
    def run_unitary(
        self,
        circuit,
        ):

        import pyquil
        circuit_native = self.build_native_circuit(circuit)
        
        unitary = []
        for i in range(2**circuit.N):
            # initial setup
            ket_format = '{0:0' + str(circuit.N) + 'b}'
            ket = ket_format.format(i)
            qubit_setup = []
            for k in ket:
                qubit_setup.append(1) if k=='1' else qubit_setup.append(0)
            circuit_native_unitary = self.build_native_circuit_unitary(circuit, qubit_setup=qubit_setup)
            # simulate wfn
            wfn_native = self.wavefunction_backend.wavefunction(circuit_native_unitary).amplitudes
            wfn = self.statevector_bit_reversal_permutation(wfn_native)
            unitary.append(wfn)
        
        unitary = np.array(unitary, dtype=np.complex128)
        return unitary

        
    def run_density_matrix(
        self,
        circuit,
        ):
        unitary = self.run_unitary(circuit)
        dm = np.matmul(unitary, unitary.transpose().conjugate())
        return dm
        

class ForestHardwareBackend(ForestBackend):

    def __init__(
        self,
        backend_name='Aspen-4-16Q-A',
        ):

        import pyquil
        self.backend = pyquil.get_qc(backend_name)
        
    def __str__(self):
        return 'Forest Hardware Backend (%s)' % self.backend
    
    @property
    def summary_str(self):
        return 'Forest Hardware Backend (%s)' % self.backend
    
    @property
    def has_statevector(self):
        return False

    @property
    def has_measurement(self):
        return True

    def run_measurement(
        self,
        circuit,
        nmeasurement,
        ):
    
        import pyquil
        circuit_native = self.build_native_circuit(circuit)
        results_native = self.backend.run_and_measure(circuit_native, nmeasurement)
        results = self.forest_to_quasar_results(results_native, nmeasurement) 

        return results

     
