import numpy as np
import quasar
from . import pauli
from . import options
from . import parameters
from . import backend
from . import optimizer
from . import results
    
class QAOA(object):

    @staticmethod
    def default_options():
        
        if hasattr(QAOA, '_default_options'): return QAOA._default_options.copy()
        opt = options.Options() 

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
            allowed_types=[backend.Backend],
            doc='Quantum simulator or hardware backend')
        opt.add_option(
            key='shots',
            value=None,
            allowed_types=[int],
            doc='Number of shots per observable, or None for infinite sampling')
        opt.add_option(
            key='shots_final',
            value=1000,
            allowed_types=[int],
            doc='Number of shots for final observables, or None for infinite sampling')

        # > Problem Structure < #

        opt.add_option(
            key='hamiltonian',
            required=True,
            allowed_types=[pauli.Pauli],
            doc='Pauli operator to diagonalize. Must contain only Z operators.')
        opt.add_option(
            key='reference_circuit',
            required=True,
            allowed_types=[quasar.Circuit],
            doc='State preparation circuit.')
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
            allowed_types=[parameters.ParameterGroup],
            doc='Parameter group for Hamiltonian time evolution circuit')
        opt.add_option(
            key='driver_parameter_group',
            required=True,
            allowed_types=[parameters.ParameterGroup],
            doc='Parameter group for driver time evolution circuit')
        opt.add_option(
            key='nlayer',
            value=1,
            allowed_types=[int],
            doc='Number of layers of Hamiltonian/driver steps. Often denoted as p')

        # > VQA Optimizer < #

        opt.add_option(
            key='optimizer',
            required=True,
            allowed_types=[optimizer.Optimizer],
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
        for string in self.hamiltonian.strings:
            if any(_ != 'Z' for _ in string.chars):
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
    def shots(self):
        return self.options['shots']

    @property
    def shots_final(self):
        return self.options['shots_final']

    @property
    def hamiltonian(self):
        return self.options['hamiltonian']

    @property
    def reference_circuit(self):
        return self.options['reference_circuit']

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
            print('%-11s = %s' % ('Shots', self.shots))
            print('%-11s = %s' % ('Shots Final', self.shots_final))
            print('')

        if self.print_level:
            print(self.hamiltonian)

        if self.print_level > 1:
            print('Reference Circuit:\n')
            print(self.reference_circuit)
            print('')
            print('Hamiltonian Circuit:\n')
            print(self.hamiltonian_circuit)
            print('')
            print('Driver Circuit:\n')
            print(self.driver_circuit)
            print('')

        # => Total Entangler Circuit <= #

        if self.print_level:
            print('Nlayer = %d' % (self.nlayer))
            print('')
        
        self.entangler_circuit = quasar.Circuit.concatenate([self.hamiltonian_circuit, self.driver_circuit]*self.nlayer)
        self.entangler_parameter_group = parameters.CompositeParameterGroup(groups=[self.hamiltonian_parameter_group, self.driver_parameter_group]*self.nlayer)

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
            shots=self.shots,
            hamiltonian=self.hamiltonian,
            reference_circuits=[self.reference_circuit],
            reference_weights=[1.0],
            entangler_circuit=self.entangler_circuit,
            entangler_circuit_parameter_group=self.entangler_parameter_group,
            guess_params=guess_params,
            )

        # => Final Shots <= #

        self.final_shots = self.backend.compute_counts(
            circuit=quasar.Circuit.concatenate([self.reference_circuit, self.entangler_circuit]),   
            shots=self.shots_final,
            )

        # => Exact Solution <= #

        sorted_kets = list(sorted([(QAOA.ket_energy(self.hamiltonian, _), _) for _ in results.Ket.build_kets(N=self.hamiltonian.N)]))
        self.exact_Es = [_[0] for _ in sorted_kets]
        self.exact_kets = [_[1] for _ in sorted_kets]
    
        for I in range(min(10, len(self.exact_kets))):
            print(I, self.exact_Es[I], self.exact_kets[I])

        # => Trailer <= #

        if self.print_level:
            print('==> End QAOA <==\n')

    @staticmethod
    def build(
        hamiltonian,
        reference_circuit=None,
        hamiltonian_circuit=None,   
        driver_circuit=None,
        hamiltonian_parameter_group=None,   
        driver_parameter_group=None,
        **kwargs):

        if reference_circuit is None:
            reference_circuit = QAOA.build_reference_circuit(hamiltonian)
        if hamiltonian_circuit is None:
            hamiltonian_circuit, hamiltonian_parameter_group = QAOA.build_hamiltonian_circuit(hamiltonian)
        if driver_circuit is None:
            driver_circuit, driver_parameter_group = QAOA.build_driver_circuit(hamiltonian)

        return QAOA.from_options(
            hamiltonian=hamiltonian,
            reference_circuit=reference_circuit,
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
            circuit.add_gate(T=0, key=A, gate=quasar.Gate.H)
        return circuit
    
    @staticmethod
    def build_hamiltonian_circuit(hamiltonian):
        if hamiltonian.max_order > 2: raise NotImplemented
        # TODO: Check for all Z
        canonical = hamiltonian.canonical
        circuit = quasar.Circuit(N=canonical.N)
        Hs = []
        # One-body terms
        circuit1 = quasar.Circuit(N=canonical.N)
        for A in range(canonical.N):
            key = 'Z%d' % A
            if key not in canonical: continue
            circuit1.add_gate(T=0, key=A, gate=quasar.Gate.Rz(theta=0.0))
            Hs.append(canonical[key])
        circuit = quasar.Circuit.concatenate([circuit, circuit1])
        # Two-body terms
        circuit2 = quasar.Circuit(N=canonical.N)
        T = 0
        for D in range(1, canonical.N):
            for O in range(0,D+1):
                added = False
                for A in range(O, canonical.N, D+1):
                    B = A + D
                    if B >= canonical.N: continue
                    added = True
                    key = 'Z%d*Z%d' % (A,B)
                    if key not in canonical: continue
                    circuit2.add_gate(T=T+0, key=(A,B), gate=quasar.Gate.CNOT)
                    circuit2.add_gate(T=T+1, key=B, gate=quasar.Gate.Rz(theta=0.0))
                    circuit2.add_gate(T=T+2, key=(A,B), gate=quasar.Gate.CNOT)
                    Hs.append(canonical[key])
                if added: T += 3
        circuit = quasar.Circuit.concatenate([circuit, circuit2])
        parameter_group = parameters.LinearParameterGroup(
            transform=np.array([Hs]).T,
            )
        return circuit, parameter_group
        
    @staticmethod
    def build_driver_circuit(hamiltonian):
        circuit = quasar.Circuit(N=hamiltonian.N)
        for A in range(hamiltonian.N):
            circuit.add_gate(T=0, key=A, gate=quasar.Gate.Rx(theta=1.0))
        parameter_group = parameters.LinearParameterGroup(
            transform=np.ones((hamiltonian.N, 1)),
            )
        return circuit, parameter_group
        
    @staticmethod
    def ket_energy(
        hamiltonian,
        ket,
        ):
        
        if hamiltonian.max_order > 2: raise NotImplemented
        E = 0.0
        for string, value in zip(hamiltonian.strings, hamiltonian.values):
            if string.order == 0:
                E += value
            elif string.order == 1:
                A = string.indices[0]
                E += +value if ket[A] == 0 else -value
            elif string.order == 2:
                A = string.indices[0]
                B = string.indices[1]
                E += +value if ket[A] == ket[B] else -value
            else:
                raise NotImplemented
        return E
    
