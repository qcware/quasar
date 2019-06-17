import collections
import numpy as np
import quasar
    
class QAOA(object):

    @staticmethod
    def default_options():
        
        if hasattr(QAOA, '_default_options'): return QAOA._default_options.copy()
        opt = quasar.Options() 

        # > Print Control < #

        opt.add_option(
            key='print_level',
            value=2,
            allowed_types=[int],
            doc='Level of detail to print (0 - nothing, 1 - minimal, 2 - verbose)')

        # > Quantum Resources < #

        opt.add_option(
            key='backend',
            required=True,
            allowed_types=[quasar.Backend],
            doc='Quantum simulator or hardware backend')
        opt.add_option(
            key='nmeasurement',
            value=None,
            allowed_types=[int],
            doc='Number of nmeasurement per observable, or None for infinite sampling')
        opt.add_option(
            key='nmeasurement_final',
            value=1000,
            allowed_types=[int],
            doc='Number of nmeasurement for final observables, or None for infinite sampling')

        # > Problem Structure < #

        opt.add_option(
            key='hamiltonian',
            required=True,
            allowed_types=[quasar.Pauli],
            doc='Pauli operator to diagonalize. Must contain only Z operators.')
        opt.add_option(
            key='reference_circuits',
            required=True,
            allowed_types=[list],
            doc='State preparation circuits.')
        opt.add_option(
            key='reference_weights',
            required=True,
            allowed_types=[list],
            doc='State preparation circuit weights.')
        opt.add_option(
            key='hamiltonian_circuit',
            required=True,
            allowed_types=[quasar.Circuit],
            doc='Hamiltonian time evolution circuit.')
        opt.add_option(
            key='driver_circuit',
            required=True,
            allowed_types=[quasar.Circuit],
            doc='Driver time evolution circuit.')
        opt.add_option(
            key='hamiltonian_parameter_group',
            required=True,
            allowed_types=[quasar.ParameterGroup],
            doc='Parameter group for Hamiltonian time evolution circuit')
        opt.add_option(
            key='driver_parameter_group',
            required=True,
            allowed_types=[quasar.ParameterGroup],
            doc='Parameter group for driver time evolution circuit')
        opt.add_option(
            key='nlayer',
            value=1,
            allowed_types=[int],
            doc='Number of layers of Hamiltonian/driver steps. Often denoted as p')

        # > VQA Optimizer < #

        # Default Optimizer
        # default_optimizer = PowellOptimizer.from_options(
        #     maxiter=100,
        #     ftol=1.0E-16,
        #     xtol=1.0E-6,
        #     ) 
        # Default Optimizer
        default_optimizer = quasar.BFGSOptimizer.from_options(
            maxiter=100,
            g_convergence=1.0E-6,
            ) 
    
        opt.add_option(
            key='optimizer',
            value=default_optimizer,
            required=True,
            allowed_types=[quasar.Optimizer],
            doc='Variational Quantum Algorithm Optimizer')

        QAOA._default_options = opt
        return QAOA._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ QAOA initialization - no computational effort performed. 
        """
        
        self.options = options

        # Validity check: Hamiltonian must be all Z
        if self.hamiltonian.unique_chars != ('Z',):
            raise RuntimeError('Invalid QAOA problem: Hamiltonian must be all Z')

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return QAOA(QAOA.default_options().set_values(kwargs))

    @property
    def print_level(self):
        return self.options['print_level']

    @property
    def backend(self):
        return self.options['backend']

    @property
    def nmeasurement(self):
        return self.options['nmeasurement']

    @property
    def nmeasurement_final(self):
        return self.options['nmeasurement_final']

    @property
    def hamiltonian(self):
        return self.options['hamiltonian']

    @property
    def reference_circuits(self):
        return self.options['reference_circuits']

    @property
    def reference_weights(self):
        return self.options['reference_weights']

    @property
    def nreference(self):
        return len(self.reference_circuits)

    @property
    def hamiltonian_circuit(self):
        return self.options['hamiltonian_circuit']

    @property
    def driver_circuit(self):
        return self.options['driver_circuit']

    @property
    def hamiltonian_parameter_group(self):
        return self.options['hamiltonian_parameter_group']

    @property
    def driver_parameter_group(self):
        return self.options['driver_parameter_group']

    @property
    def nlayer(self):
        return self.options['nlayer']
        
    @property
    def optimizer(self):
        return self.options['optimizer']
        
    def run(self, guess_params=None):

        # => Header <= #

        if self.print_level:
            print('==> QAOA <==\n')

        if self.print_level:
            print('%-11s = %s' % ('Backend', self.backend))
            print('%-11s = %s' % ('Shots', self.nmeasurement))
            print('%-11s = %s' % ('Shots Final', self.nmeasurement_final))
            print('')

        if self.print_level:
            print('Pauli Hamiltonian:')
            print(self.hamiltonian)
            print('')

        # => Reference Circuits <= #

        if self.print_level:
            print('Nreference = %d\n' % self.nreference)
            print('Reference Weights:\n')
            print('%-5s: %11s' % ('State', 'Weight'))
            for I, w in enumerate(self.reference_weights):
                print('%-5d: %11.3E' % (I, w))
            print('')
    
        if self.print_level > 1:
            for I, reference_circuit in enumerate(self.reference_circuits):
                print('Reference Circuit %d:\n' % I)
                print(reference_circuit)
            print('')

        # => Total Entangler Circuit <= #

        if self.print_level > 1:
            print('Hamiltonian Circuit:\n')
            print(self.hamiltonian_circuit)
            print('')
            print('Driver Circuit:\n')
            print(self.driver_circuit)
            print('')


        if self.print_level:
            print('Nlayer = %d' % (self.nlayer))
            print('')
        
        self.entangler_circuit = quasar.Circuit.concatenate([self.hamiltonian_circuit, self.driver_circuit]*self.nlayer)
        self.entangler_parameter_group = quasar.CompositeParameterGroup(groups=[self.hamiltonian_parameter_group, self.driver_parameter_group]*self.nlayer)

        # TODO: User-provided entangler circuit 

        # => Parameter Guess <= #

        if guess_params is None:
            if self.print_level:
                print('Params guessed as zero.\n')
            guess_params = np.zeros((self.entangler_parameter_group.nparam,))
        else:
            if self.print_level:
                print('Params guessed by user.\n')

        # => Parameter Optimization <= #

        self.entangler_parameters, self.entangler_circuit, self.entangler_history = self.optimizer.optimize(
            print_level=self.print_level,
            backend=self.backend,
            nmeasurement=self.nmeasurement,
            hamiltonian=self.hamiltonian,
            reference_circuits=self.reference_circuits,
            reference_weights=self.reference_weights,
            entangler_circuit=self.entangler_circuit,
            entangler_circuit_parameter_group=self.entangler_parameter_group,
            guess_params=guess_params,
            )

        # => Final QAOA Measurements (by Reference Circuit) <= #

        self.measurements = []
        for reference_circuit in self.reference_circuits:
            self.measurements.append(quasar.run_measurement(
                backend=self.backend,
                circuit=quasar.Circuit.concatenate([reference_circuit, self.entangler_circuit]),   
                nmeasurement=self.nmeasurement_final,
                ))

        # => Final QAOA Solutions (by Reference Circuit) <= #

        self.solutions = []
        for measurement in self.measurements:
            self.solutions.append(QAOA.build_optimization_result(
                hamiltonian=self.hamiltonian,
                measurement=measurement, 
                ))

        self.full_solution = quasar.OptimizationResult.merge(self.solutions).energy_sorted
        print(self.full_solution)

        # => Trailer <= #

        if self.print_level:
            print('==> End QAOA <==\n')

    @staticmethod
    def build(
        hamiltonian,
        reference_circuits=None,
        reference_weigts=None,
        hamiltonian_circuit=None,   
        driver_circuit=None,
        hamiltonian_parameter_group=None,   
        driver_parameter_group=None,
        **kwargs):

        if reference_circuits is None:
            reference_circuits = [QAOA.build_reference_circuit(hamiltonian)]
            reference_weights = [1.0]
        if hamiltonian_circuit is None:
            hamiltonian_circuit, hamiltonian_parameter_group = QAOA.build_hamiltonian_circuit(hamiltonian)
        if driver_circuit is None:
            driver_circuit, driver_parameter_group = QAOA.build_driver_circuit(hamiltonian)

        return QAOA.from_options(
            hamiltonian=hamiltonian,
            reference_circuits=reference_circuits,
            reference_weights=reference_weights,
            hamiltonian_circuit=hamiltonian_circuit,
            driver_circuit=driver_circuit,
            hamiltonian_parameter_group=hamiltonian_parameter_group,
            driver_parameter_group=driver_parameter_group,
            **kwargs,
            )
    
    @staticmethod
    def build_reference_circuit(hamiltonian):
        circuit = quasar.Circuit(N=hamiltonian.N)
        for A in range(hamiltonian.N):
            circuit.add_gate(time=0, qubits=A, gate=quasar.Gate.H)
        return circuit
    
    @staticmethod
    def build_hamiltonian_circuit(hamiltonian):
        if hamiltonian.max_order > 2: raise NotImplemented
        circuit = quasar.Circuit(N=hamiltonian.N)
        Hs = []
        # One-body terms
        circuit1 = quasar.Circuit(N=hamiltonian.N)
        for A in range(hamiltonian.N):
            key = 'Z%d' % A
            if key not in hamiltonian: continue
            circuit1.add_gate(time=0, qubits=A, gate=quasar.Gate.Rz(theta=0.0))
            Hs.append(hamiltonian[key])
        circuit = quasar.Circuit.concatenate([circuit, circuit1])
        # Two-body terms
        circuit2 = quasar.Circuit(N=hamiltonian.N)
        T = 0
        for D in range(1, hamiltonian.N):
            for O in range(0,D+1):
                added = False
                for A in range(O, hamiltonian.N, D+1):
                    B = A + D
                    if B >= hamiltonian.N: continue
                    added = True
                    key = 'Z%d*Z%d' % (A,B)
                    if key not in hamiltonian: continue
                    circuit2.add_gate(time=T+0, qubits=(A,B), gate=quasar.Gate.CX)
                    circuit2.add_gate(time=T+1, qubits=B, gate=quasar.Gate.Rz(theta=0.0))
                    circuit2.add_gate(time=T+2, qubits=(A,B), gate=quasar.Gate.CX)
                    Hs.append(hamiltonian[key])
                if added: T += 3
        circuit = quasar.Circuit.concatenate([circuit, circuit2])
        parameter_group = quasar.LinearParameterGroup(
            transform=np.array([Hs]).T,
            )
        return circuit, parameter_group
        
    @staticmethod
    def build_driver_circuit(hamiltonian):
        circuit = quasar.Circuit(N=hamiltonian.N)
        for A in range(hamiltonian.N):
            circuit.add_gate(time=0, qubits=A, gate=quasar.Gate.Rx(theta=1.0))
        parameter_group = quasar.LinearParameterGroup(
            transform=np.ones((hamiltonian.N, 1)),
            )
        return circuit, parameter_group
        
    @staticmethod
    def ket_energy(
        hamiltonian,
        ket,
        ):

        ket = quasar.Ket(ket)
        E = 0.0
        for string, value in hamiltonian.items():
            parity = +1
            for qubit in string.qubits:
                parity *= -1 if ket[qubit] == 1 else +1
            E += parity * value
        return E

    @staticmethod
    def build_optimization_result(
        hamiltonian,
        measurement,
        ):

        result = quasar.OptimizationResult()
        for ket, N in measurement.items():
            E = QAOA.ket_energy(hamiltonian, ket)
            result[ket] = (E, N)
        return result.energy_sorted
