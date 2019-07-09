from .backend import Backend
from .circuit import Circuit
from .jobsummary import JobSummary

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
        return True

    @property
    def native_circuit_type(self):
        return Circuit

    def build_native_circuit(
        self,
        circuit,
        ):

        # Dropthrough
        if isinstance(circuit, self.native_circuit_type): return circuit

        # Can only convert quasar -> quasar
        if not isinstance(circuit, Circuit): raise RuntimeError('circuit must be Circuit type for build_native_circuit: %s' % (circuit))

    def build_native_circuit_in_basis(
        self,
        circuit,
        basis,
        ):

        circuit = self.build_native_circuit(circuit)
    
        if len(basis) > circuit.N: raise RuntimeError('len(basis) > circuit.N. Often implies pauli.N > circuit.N')
        
        basis_circuit = Circuit(N=circuit.N)
        for A, char in enumerate(basis): 
            if char == 'X': basis_circuit.H(A)
            elif char == 'Y': basis_circuit.Rx2(A)
            elif char == 'Z': continue # Computational basis
            else: raise RuntimeError('Unknown basis: %s' % char)
        
        return Circuit.concatenate([circuit, basis_circuit])

    def build_quasar_circuit(
        self,
        native_circuit,
        ):

        # Dropthrough
        if isinstance(native_circuit, self.native_circuit_type): return native_circuit

        # Can only convert quasar -> quasar
        if not isinstance(native_circuit, Circuit): raise RuntimeError('circuit must be Circuit type for build_native_circuit: %s' % (native_circuit))

    def run_statevector(
        self,
        circuit,
        compressed=True,
        ):

        statevector = (circuit.compressed() if compressed else circuit).simulate()
        # summary = JobSummary(
        #     resources={ 'nstatevector' : 1, },
        #     attributes={ 'name' : str(self), },
        #     )
        # return statevector, summary
        return statevector

    def run_measurement(
        self,
        circuit,
        nmeasurement=1000,
        compressed=True,
        ):

        measurement = (circuit.compressed() if compressed else circuit).measure(nmeasurement)
        # summary = JobSummary(
        #     resources={ 'nstatevector' : 1, 'nmeasurement' : nmeasurement, 'nmeasurent_call' : 1, },
        #     attributes={ 'name' : str(self), },
        #     )
        # return measurement, summary
        return measurement


