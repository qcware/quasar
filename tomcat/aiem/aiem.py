import numpy as np
from ..quasar import quasar
from ..util import Options
from ..util import memoized_property
from ..core import Pauli
from ..core import Backend
from ..core import QuasarSimulatorBackend
from ..core import Collocation
from ..core import Optimizer
from ..core import BFGSOptimizer
from ..core import PowellOptimizer
from ..core import IdentityParameterGroup
from .aiem_data import AIEMMonomer
from .aiem_data import AIEMMonomerGrad
from .aiem_data import AIEMPauli
from .aiem_data import AIEMUtil

class AIEM(object):

    @staticmethod

    def default_options():
        
        if hasattr(AIEM, '_default_options'): return AIEM._default_options.copy()
        opt = Options() 

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
            allowed_types=[Backend],
            doc='Quantum simulator or hardware backend')
        opt.add_option(
            key='nmeasurement',
            value=None,
            allowed_types=[int],
            doc='Number of nmeasurement per observable, or None for infinite sampling (backend must support statevector simulation)')
        opt.add_option(
            key='nmeasurement_subspace',
            value=None,
            allowed_types=[int],
            doc='Number of nmeasurement per observable for final subspace Hamiltonian observation, or None for infinite sampling (backend must support statevector simulation)')

        # > Problem Description < #

        opt.add_option(
            key='nstate',
            required=True,
            allowed_types=[int],
            doc='Number of states to target')
        opt.add_option(
            key='vqe_weights',
            required=False,
            allowed_types=[np.ndarray],
            doc='Weights for SA-VQE optimization (optional - 1/nstate if not provided)')
        opt.add_option(
            key='aiem_monomer',
            required=True,
            allowed_types=[AIEMMonomer],
            doc='AIEM Monomer properties')
        opt.add_option(
            key='aiem_monomer_grad',
            required=False,
            allowed_types=[AIEMMonomerGrad],
            doc='AIEM Monomer gradient properties')
        opt.add_option(
            key='aiem_hamiltonian_pauli',
            required=False,
            allowed_types=[AIEMPauli],
            doc='AIEM Hamiltonian in Pauli basis (optional - generated from aiem_monomer if not provided')

        # > Quantum Circuit Specification/Recipe < #

        opt.add_option(
            key='cis_circuit_type',
            value='mark2',
            required=True,
            allowed_types=[str],
            allowed_values=['mark1', 'mark2'],
            doc='CIS state preparation circuit recipe')
        opt.add_option(
            key='vqe_circuit',
            value=None,
            allowed_types=[quasar.Circuit],
            doc='Explicit SA-VQE Entangler circuit (1st priority)')
        opt.add_option(
            key='vqe_circuit_type',
            value='mark1',
            required=True,
            allowed_types=[str],
            allowed_values=['mark1', 'mark2'],
            doc='SA-VQE Entangler circuit recipe (2nd priority)')

        # > Variational Quantum Algorithm Optimizer < #

        # Default Optimizer
        # default_optimizer = PowellOptimizer.from_options(
        #     maxiter=100,
        #     ftol=1.0E-16,
        #     xtol=1.0E-6,
        #     ) 
        # Default Optimizer
        default_optimizer = BFGSOptimizer.from_options(
            maxiter=100,
            g_convergence=1.0E-6,
            ) 
    
        opt.add_option(
            key='optimizer',
            value=default_optimizer,
            required=True,
            allowed_types=[Optimizer],
            doc='Variational Quantum Algorithm Optimizer')

        AIEM._default_options = opt
        return AIEM._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ AIEM initialization - no computational effort performed. 
        """
        
        self.options = options

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return AIEM(AIEM.default_options().set_values(kwargs))

    @property
    def print_level(self):
        return self.options['print_level']

    @property
    def backend(self):
        return self.options['backend']

    @memoized_property
    def backend_quasar_simulator(self):
        return QuasarSimulatorBackend()

    @property
    def nmeasurement(self):
        return self.options['nmeasurement']

    @property
    def nmeasurement_subspace(self):
        return self.options['nmeasurement_subspace']

    @property
    def N(self):
        return self.aiem_monomer.N

    @property
    def ncis(self):
        return self.N + 1 

    @property
    def nstate(self):
        return self.options['nstate']    

    @memoized_property
    def vqe_weights(self):
        if self.options['vqe_weights']:
            return self.options['vqe_weights']
        return np.ones((self.nstate,)) / self.nstate

    @property
    def aiem_monomer(self):
        return self.options['aiem_monomer']

    @property
    def aiem_monomer_grad(self):
        return self.options['aiem_monomer_grad']

    @memoized_property
    def aiem_hamiltonian_pauli(self):
        # Possible custom Pauli Hamiltonian Representation
        if self.options['aiem_hamiltonian_pauli']:
            return self.options['aiem_hamiltonian_pauli']
        return AIEMUtil.operator_to_pauli(AIEMUtil.monomer_to_hamiltonian(self.aiem_monomer))

    @memoized_property
    def hamiltonian_pauli(self):
        """ Normal Ordered (self energy E removed) """
        return AIEMUtil.aiem_pauli_to_pauli(self.aiem_hamiltonian_pauli, self_energy=False)

    @property
    def cis_circuit_type(self):
        return self.options['cis_circuit_type']

    @property
    def cis_circuit_function(self):
        if self.cis_circuit_type == 'mark1': return AIEM.build_cis_circuit_mark1
        elif self.cis_circuit_type == 'mark2': return AIEM.build_cis_circuit_mark2
        else: raise RuntimeError('Unknown cis_circuit_type: %s' % self.cis_circuit_type)

    @property
    def vqe_circuit_function(self):
        if self.options['vqe_circuit_type'] == 'mark1' : return AIEM.build_vqe_circuit_mark1
        if self.options['vqe_circuit_type'] == 'mark2' : return AIEM.build_vqe_circuit_mark2
        else: raise RuntimeError('Unknown vqe_circuit_type: %s' % self.options['vqe_circuit_type'])
    
    @property
    def optimizer(self):
        return self.options['optimizer']

    def compute_energy(
        self,
        param_values_ref=None,
        cis_C_ref=None,
        vqe_C_ref=None,
        ):

        # > Header < #
    
        if self.print_level:
            print('==> MC-VQE+AIEM <==\n')
    
            print('Quantum Resources:')
            print('  %-14s = %s' % ('Backend', self.backend))
            print('  %-14s = %s' % ('Shots', self.nmeasurement))
            print('  %-14s = %s' % ('Shots Subspace', self.nmeasurement_subspace))
            print('')

            print('AIEM Problem:') 
            print(' %-12s = %d' % ('N', self.N))
            print(' %-12s = %d' % ('Ncis', self.ncis))
            print(' %-12s = %d' % ('Nstate', self.nstate))
            print(' %-12s = %s' % ('Connectivity', self.aiem_monomer.connectivity_str))
            print('')

        # > CIS States < #

        self.cis_E, self.cis_C = AIEM.solve_cis(
            hamiltonian=self.aiem_hamiltonian_pauli,
            nstate=self.nstate,
            )

        # > CIS State Phases < #

        if cis_C_ref is not None:
            if cis_C_ref.shape != self.cis_C.shape: raise RuntimeError('cis_C_ref.shape != cis_C.shape')
            P = np.diag(np.dot(cis_C_ref.T, self.cis_C))
            if self.print_level: print('CIS Phasing:')
            for I in range(self.cis_C.shape[1]):
                if np.abs(P[I]) < 0.5: print('Warning: probable CIS state rotation: I=%d' % I)
                if self.print_level: print('Applying sign of %2d to state %d. Overlap is %.3E' % (np.sign(P[I]), I, P[I]))
                self.cis_C[:,I] *= np.sign(P[I])
            if self.print_level: print('')

        # > CIS Angles < #        

        self.cis_angles = [AIEM.compute_cis_angles(
            cs=self.cis_C[:,T],
            ) for T in range(self.cis_C.shape[1])]

        # > CIS Circuits < #

        if self.print_level:
            print('CIS Circuits:')
            print('  %16s = %s' % ('cis_circuit_type', self.cis_circuit_type))
            print('')

        self.cis_circuits = [self.cis_circuit_function(
            thetas=thetas,
            ) for thetas in self.cis_angles]

        # > CIS Circuit Verification (Quasar Simulator) < #

        if self.print_level:
            print('CIS Verification:\n')
            print('%-5s: %11s' % ('State', 'Match'))
            Ecis2 = self.verify_cis(
                backend=self.backend_quasar_simulator,
                nmeasurement=None,
                )[0]
            for I in range(self.nstate):
                dEcis = Ecis2[I] - self.cis_E[I]
                print('%-5d: %11.3E' % (I, dEcis))
            print('')

        # > SA-VQE Weights < #

        if self.print_level:
            print('SA-VQE Weights:\n')
            print('%-5s: %11s' % ('State', 'Weight'))
            for I, w in enumerate(self.vqe_weights):
                print('%-5d: %11.3E' % (I, w))
            print('')

        # > VQE Circuit Construction < #

        if self.options['vqe_circuit']:
            self.vqe_circuit = self.options['vqe_circuit'].copy()
        else:
            self.vqe_circuit = self.vqe_circuit_function(self.aiem_hamiltonian_pauli)

        if self.print_level:
            print('VQE Entangler Circuit:')
            print('  %-16s = %s' % ('vqe_circuit_type', 'custom' if self.options['vqe_circuit'] else self.options['vqe_circuit_type']))
            print('  %-16s = %d' % ('vqe_nparam', self.vqe_circuit.nparam))
            print('')

        # > Param Values < #

        if param_values_ref:
            if self.print_level:
                print('Initial VQE params taken from input guess.\n')
            self.vqe_circuit.set_param_values(param_values_ref)
        else:
            if self.print_level:
                if self.options['vqe_circuit']:
                    print('Initial VQE params taken from input circuit.\n')
                else:
                    print('Initital VQE params guessed as zero.\n')

        # > Entangler Parameter Group < # 

        self.vqe_parameter_group = IdentityParameterGroup(nparam=self.vqe_circuit.nparam)

        # > MC-VQE Parameter Optimization < #

        self.vqe_parameters, self.vqe_circuit, self.vqe_history = self.optimizer.optimize(
            print_level=self.print_level,
            backend=self.backend,
            nmeasurement=self.nmeasurement,
            hamiltonian=self.hamiltonian_pauli,
            reference_circuits=self.cis_circuits,
            reference_weights=self.vqe_weights,
            entangler_circuit=self.vqe_circuit,
            entangler_circuit_parameter_group=self.vqe_parameter_group,
            guess_params=self.vqe_circuit.param_values,
            )

        # > Finished MC-VQE Circuit Parameters < #

        if self.print_level:
            print('Finished VQE Circuit:\n')
            print(self.vqe_circuit)
            print('')

            print('Finished VQE Parameters:\n')
            print(self.vqe_circuit.param_str)

        # > Subspace Eigenproblem < #

        self.subspace_eigenproblem()

        # > VQE Phasing <= #

        if vqe_C_ref is not None:
            if vqe_C_ref.shape != self.vqe_C.shape: raise RuntimeError('vqe_C_ref.shape != vqe_C.shape')
            P = np.diag(np.dot(vqe_C_ref.T, self.vqe_C))
            if self.print_level: print('VQE Phasing:')
            for I in range(self.vqe_C.shape[1]):
                if np.abs(P[I]) < 0.5: print('Warning: probable VQE state rotation: I=%d' % I)
                if self.print_level: print('Applying sign of %2d to state %d. Overlap is %.3E' % (np.sign(P[I]), I, P[I]))
                self.vqe_C[:,I] *= np.sign(P[I])
                self.vqe_V[:,I] *= np.sign(P[I])
            if self.print_level: print('')

        # > VQE Angles < #        

        self.vqe_angles = [AIEM.compute_cis_angles(
            cs=self.vqe_C[:,T],
            ) for T in range(self.vqe_C.shape[1])]

        # > VQE Circuits < #

        self.vqe_circuits = [self.cis_circuit_function(
            thetas=thetas,
            ) for thetas in self.vqe_angles]

        # > VQE Pauli DMs < #

        self.rotate_vqe_dms()

        # > FCI States < #

        # FCI (explicit or Davidson)
        self.fci_E, self.fci_C = AIEM.compute_fci(
            hamiltonian=self.aiem_hamiltonian_pauli,
            nstate=self.nstate,
            )
    
        # > Total Energies (Incl. Reference Energies) < #

        self.fci_tot_E = self.fci_E + self.aiem_hamiltonian_pauli.E
        self.vqe_tot_E = self.vqe_E + self.aiem_hamiltonian_pauli.E
        self.cis_tot_E = self.cis_E + self.aiem_hamiltonian_pauli.E

        # > Properties/Analysis < #

        self.fci_O, self.vqe_O, self.cis_O = self.compute_oscillator_strengths()

        if self.print_level:
            self.analyze_energies()
            self.analyze_transitions()
            self.analyze_excitations()

        # > Trailer < #

        if self.print_level:
            print('  Viper:    "That\'s pretty arrogant, considering the company you\'re in."')
            print('  Maverick: "Yes, sir."')
            print('  Viper:    "I like that in a pilot."')
            print('')
            print('==> End MC-VQE+AIEM <==\n')

    # => Analysis <= #

    def analyze_energies(self):

        print('State Energies (Total):\n')
        print('%-5s: %24s %24s %24s %24s %24s' % (
            'State',
            'FCI',
            'VQE',
            'CIS',
            'dVQE',
            'dCIS',
            ))
        for I in range(self.nstate):
            print('%-5d: %24.16E %24.16E %24.16E %24.15E %24.16E' % (
                I,
                self.fci_tot_E[I],
                self.vqe_tot_E[I],
                self.cis_tot_E[I],
                self.vqe_tot_E[I] - self.fci_tot_E[I],
                self.cis_tot_E[I] - self.fci_tot_E[I],
                ))
        print('')
            
        print('State Energies (Normal):\n')
        print('%-5s: %24s %24s %24s %24s %24s' % (
            'State',
            'FCI',
            'VQE',
            'CIS',
            'dVQE',
            'dCIS',
            ))
        for I in range(self.nstate):
            print('%-5d: %24.16E %24.16E %24.16E %24.15E %24.16E' % (
                I,
                self.fci_E[I],
                self.vqe_E[I],
                self.cis_E[I],
                self.vqe_E[I] - self.fci_E[I],
                self.cis_E[I] - self.fci_E[I],
                ))
        print('')
            
        print('Excitation Energies:\n')
        print('%-5s: %24s %24s %24s %24s %24s' % (
            'State',
            'FCI',
            'VQE',
            'CIS',
            'dVQE',
            'dCIS',
            ))
        for I in range(1,self.nstate):
            print('%-5d: %24.16E %24.16E %24.16E %24.15E %24.16E' % (
                I,
                self.fci_E[I] - self.fci_E[0],
                self.vqe_E[I] - self.vqe_E[0],
                self.cis_E[I] - self.cis_E[0],
                self.vqe_E[I] - self.vqe_E[0] - self.fci_E[I] + self.fci_E[0],
                self.cis_E[I] - self.cis_E[0] - self.fci_E[I] + self.fci_E[0],
                ))
        print('')

    def compute_oscillator_strengths(self):

        mu_pauli = AIEMUtil.monomer_to_pauli_dipole(self.aiem_monomer)

        fci_O = []
        for J in range(1, self.nstate):
            T_pauli = self.compute_fci_tdm(I=0, J=J)
            X = AIEMUtil.pauli_energy(pauli_hamiltonian=mu_pauli[0], pauli_dm=T_pauli)
            Y = AIEMUtil.pauli_energy(pauli_hamiltonian=mu_pauli[1], pauli_dm=T_pauli)
            Z = AIEMUtil.pauli_energy(pauli_hamiltonian=mu_pauli[2], pauli_dm=T_pauli)
            O = 2.0 / 3.0 * (self.fci_E[J] - self.fci_E[0]) * (X**2 + Y**2 + Z**2)
            fci_O.append(O)
        fci_O = np.array(fci_O)

        vqe_O = []
        for J in range(1, self.nstate):
            T_pauli = self.compute_vqe_tdm(I=0, J=J)
            X = AIEMUtil.pauli_energy(pauli_hamiltonian=mu_pauli[0], pauli_dm=T_pauli)
            Y = AIEMUtil.pauli_energy(pauli_hamiltonian=mu_pauli[1], pauli_dm=T_pauli)
            Z = AIEMUtil.pauli_energy(pauli_hamiltonian=mu_pauli[2], pauli_dm=T_pauli)
            O = 2.0 / 3.0 * (self.vqe_E[J] - self.vqe_E[0]) * (X**2 + Y**2 + Z**2)
            vqe_O.append(O)
        vqe_O = np.array(vqe_O)

        cis_O = []
        for J in range(1, self.nstate):
            T_pauli = self.compute_cis_tdm(I=0, J=J)
            X = AIEMUtil.pauli_energy(pauli_hamiltonian=mu_pauli[0], pauli_dm=T_pauli)
            Y = AIEMUtil.pauli_energy(pauli_hamiltonian=mu_pauli[1], pauli_dm=T_pauli)
            Z = AIEMUtil.pauli_energy(pauli_hamiltonian=mu_pauli[2], pauli_dm=T_pauli)
            O = 2.0 / 3.0 * (self.cis_E[J] - self.cis_E[0]) * (X**2 + Y**2 + Z**2)
            cis_O.append(O)
        cis_O = np.array(cis_O)

        return fci_O, vqe_O, cis_O
        
    def analyze_transitions(self):

        print('Oscillator Strengths:\n')
        print('%-5s: %24s %24s %24s %24s %24s' % (
            'State',
            'FCI',
            'VQE',
            'CIS',
            'dVQE',
            'dCIS',
            ))
        for I in range(1,self.nstate):
            print('%-5d: %24.16E %24.16E %24.16E %24.15E %24.16E' % (
                I,
                self.fci_O[I-1],
                self.vqe_O[I-1],
                self.cis_O[I-1],
                self.vqe_O[I-1] - self.fci_O[I-1],
                self.cis_O[I-1] - self.fci_O[I-1],
                ))
        print('')

    def analyze_excitations(self):
    
        fci_D = []        
        for I in range(self.nstate):
            D_pauli = self.compute_fci_dm(I=I)
            fci_D.append(D_pauli)
    
        vqe_D = []        
        for I in range(self.nstate):
            D_pauli = self.compute_vqe_dm(I=I)
            vqe_D.append(D_pauli)
    
        cis_D = []        
        for I in range(self.nstate):
            D_pauli = self.compute_cis_dm(I=I)
            cis_D.append(D_pauli)

        print('FCI Excitation Fractions:\n')
        print('%-3s: ' % ('A'), end=' ')
        for I in range(self.nstate):
            print('I=%3d ' % (I), end=' ')
        print('')
        for A in range(self.N):
            print('%-3d: ' % (A), end=' ')
            for I in range(self.nstate):
                print('%5.3f ' % (0.5 * (1.0 - fci_D[I].Z[A])), end=' ')
            print('')
        print('')

        print('VQE Excitation Fractions:\n')
        print('%-3s: ' % ('A'), end=' ')
        for I in range(self.nstate):
            print('I=%3d ' % (I), end=' ')
        print('')
        for A in range(self.N):
            print('%-3d: ' % (A), end=' ')
            for I in range(self.nstate):
                print('%5.3f ' % (0.5 * (1.0 - vqe_D[I].Z[A])), end=' ')
            print('')
        print('')

        print('CIS Excitation Fractions:\n')
        print('%-3s: ' % ('A'), end=' ')
        for I in range(self.nstate):
            print('I=%3d ' % (I), end=' ')
        print('')
        for A in range(self.N):
            print('%-3d: ' % (A), end=' ')
            for I in range(self.nstate):
                print('%5.3f ' % (0.5 * (1.0 - cis_D[I].Z[A])), end=' ')
            print('')
        print('')

    # => CIS Considerations (Classical) <= #

    @staticmethod
    def compute_cis_hamiltonian(hamiltonian):
        """ Ordered [0, A] - the singles are in reverse lexical ordering to quasar Hilbert space order """
        Eref = 0.0
        Eref += 1.0 * np.sum(hamiltonian.Z) 
        Eref += 0.5 * np.sum(hamiltonian.ZZ) 
        H = np.diag(Eref*np.ones(hamiltonian.N+1))
        for A in range(hamiltonian.N):
            H[A+1,A+1] -= 2.0 * hamiltonian.Z[A] 
            H[A+1,A+1] -= 1.0 * np.sum(hamiltonian.ZZ[A,:])
            H[A+1,A+1] -= 1.0 * np.sum(hamiltonian.ZZ[:,A])
        for A in range(hamiltonian.N):
            H[0,A+1] += 1.0 * hamiltonian.X[A] 
            H[0,A+1] += 0.5 * np.sum(hamiltonian.XZ[A,:])
            H[0,A+1] += 0.5 * np.sum(hamiltonian.ZX[:,A])
            H[A+1,0] += 1.0 * hamiltonian.X[A] 
            H[A+1,0] += 0.5 * np.sum(hamiltonian.XZ[A,:])
            H[A+1,0] += 0.5 * np.sum(hamiltonian.ZX[:,A])
        for A in range(hamiltonian.N):
            for B in range(hamiltonian.N): 
                H[A+1,B+1] += 1.0 * hamiltonian.XX[A,B]
        return H 

    @staticmethod
    def compute_cis_hamiltonian_grad(
        hamiltonian,
        D,
        ):

        G = AIEMPauli.zeros_like(hamiltonian)
        for A in range(hamiltonian.N):
            for B in range(hamiltonian.N):
                G.XX[A,B] += 1.0 * D[A+1, B+1]
        for A in range(hamiltonian.N):
            G.X[A] += 1.0 * D[0,A+1]
            G.XZ[A,:] += 0.5 * D[0,A+1]
            G.ZX[:,A] += 0.5 * D[0,A+1]
            G.X[A] += 1.0 * D[A+1,0]
            G.XZ[A,:] += 0.5 * D[A+1,0]
            G.ZX[:,A] += 0.5 * D[A+1,0]
        for A in range(hamiltonian.N):
            G.Z[A] -= 2.0 * D[A+1,A+1]
            G.ZZ[A,:] -= 1.0 * D[A+1,A+1]
            G.ZZ[:,A] -= 1.0 * D[A+1,A+1]
        G.Z[:] += 1.0 * np.sum(np.diag(D))
        G.ZZ[:,:] += 0.5 * np.sum(np.diag(D))
        # print(np.sum(np.diag(D))) # Seems to be 0.0

        # Restricted two-body derivatives
        G.XX *= hamiltonian.connectivity
        G.XZ *= hamiltonian.connectivity
        G.ZX *= hamiltonian.connectivity
        G.ZZ *= hamiltonian.connectivity

        # Factors of 1/2 for off-diagonal get deferred to later
        G.XX *= 2.0
        G.XZ *= 2.0
        G.ZX *= 2.0
        G.ZZ *= 2.0

        return G

    @staticmethod
    def solve_cis(
        hamiltonian,
        nstate,
        ):

        H = AIEM.compute_cis_hamiltonian(hamiltonian=hamiltonian)
        E, C = np.linalg.eigh(H)
        return E[:nstate], C[:,:nstate]
    
    # => CIS Angles <= #

    @staticmethod
    def compute_cis_angles(cs):
    
        # > New MC-VQE-Grad Algorithm < #

        N = len(cs) - 1

        vs = np.zeros((N+1,))
        for L in range(N+1):
            vs[L] = np.sum(cs[L:]**2)

        thetas = np.zeros((N,))
        for D in range(N):
            thetas[D] = np.arccos(cs[D] / np.sqrt(vs[D]))
        if cs[-1] < 0.0: thetas[-1] *= -1.0

        return thetas

    @staticmethod
    def compute_cis_angles_jacobian(cs):

        N = len(cs) - 1

        vs = np.zeros((N+1,))
        for L in range(N+1):
            vs[L] = np.sum(cs[L:]**2)

        J = np.zeros((N,N+1))
        for D in range(N):
            J[D,D] = - 1.0 / np.sqrt(vs[D+1])
            for K in range(D,N+1): 
                J[D,K] += cs[D] * cs[K] / (np.sqrt(vs[D+1]) * vs[D])
        if cs[-1] < 0.0: J[-1,:] *= -1.0

        return J

    @staticmethod
    def contract_cis_angles_jacobian(cs, ds):

        J = AIEM.compute_cis_angles_jacobian(cs=cs)
        return np.dot(J.T, ds)

    # => CIS State Preparation Quantum Circuit <= #
            
    @staticmethod
    def build_cis_circuit_mark1(thetas):
        N = len(thetas)
        circuit = quasar.Circuit(N=N)
        circuit.add_gate(T=0, key=0, gate=quasar.Gate.Ry(theta=thetas[0]))
        for I, theta in enumerate(thetas[1:]):
            circuit.add_gate(T=2*I, key=(I+1), gate=quasar.Gate.Ry(theta=-theta/2.0)) 
            circuit.add_gate(T=2*I+1, key=(I, I+1), gate=quasar.Gate.CZ)
            circuit.add_gate(T=2*I+2, key=(I+1), gate=quasar.Gate.Ry(theta=+theta/2.0)) 
        circuit2 = quasar.Circuit(N=N)
        T = 0
        for A in range(N-2, -1, -1):
            for B in range(N-1, A, -1):
                circuit2.add_gate(T=T, key=(B,A), gate=quasar.Gate.CNOT)
                T += 1
        return quasar.Circuit.concatenate([circuit, circuit2])

    @staticmethod
    def build_cis_circuit_mark2(thetas):
        N = len(thetas)
        circuit = quasar.Circuit(N=N)
        circuit.add_gate(T=0, key=0, gate=quasar.Gate.Ry(theta=thetas[0]))
        for I, theta in enumerate(thetas[1:]):
            circuit.add_gate(T=2*I, key=(I+1), gate=quasar.Gate.Ry(theta=-theta/2.0)) 
            circuit.add_gate(T=2*I+1, key=(I, I+1), gate=quasar.Gate.CZ)
            circuit.add_gate(T=2*I+2, key=(I+1), gate=quasar.Gate.Ry(theta=+theta/2.0)) 
            circuit.add_gate(T=2*I+(3 if I+2 == N else 4), key=(I+1, I), gate=quasar.Gate.CX)
        return circuit

    # => CIS Verification <= #

    def verify_cis(
        self,
        backend,
        nmeasurement,
        ):

        Es = []
        Ds = []
        for I, circuit in enumerate(self.cis_circuits):
            E, D = Collocation.compute_energy_and_pauli_dm(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=self.hamiltonian_pauli,
                circuit=circuit,
                )
            Es.append(E)
            Ds.append(D)
        return Es, Ds
            
    # => VQE Entangler Circuit Recipes <= #

    @staticmethod
    def build_vqe_circuit_mark1(hamiltonian, nonredundant=True):
        
        """ From https://arxiv.org/pdf/1203.0722.pdf """

        # Validity checks
        if hamiltonian.N % 2: 
            raise RuntimeError('Currently only set up for N even')
        if not hamiltonian.is_linear and not hamiltonian.is_cyclic:
            raise RuntimeError('Hamiltonian must be linear or cyclic')

        # 2-body circuit (even)
        circuit_even = quasar.Circuit(N=hamiltonian.N)
        for A in range(hamiltonian.N):
            if A % 2: continue
            B = A + 1
            circuit_even.add_gate(T=0,  key=A, gate=quasar.Gate.Ry(theta=0.0))
            circuit_even.add_gate(T=0,  key=B, gate=quasar.Gate.Ry(theta=0.0))
            # circuit_even.add_gate(T=1,  key=(A,B), gate=quasar.Gate.CNOT)
            circuit_even.add_gate(T=1,  key=(A,B), gate=quasar.Gate.CZ)
            circuit_even.add_gate(T=2,  key=A, gate=quasar.Gate.Ry(theta=0.0))
            circuit_even.add_gate(T=2,  key=B, gate=quasar.Gate.Ry(theta=0.0))
            # circuit_even.add_gate(T=3,  key=(A,B), gate=quasar.Gate.CNOT)
            circuit_even.add_gate(T=3,  key=(A,B), gate=quasar.Gate.CZ)
            circuit_even.add_gate(T=4,  key=A, gate=quasar.Gate.Ry(theta=0.0))
            circuit_even.add_gate(T=4,  key=B, gate=quasar.Gate.Ry(theta=0.0))

        # 2-body circuit (odd)
        circuit_odd = quasar.Circuit(N=hamiltonian.N)
        for A in range(hamiltonian.N):
            if (A + 1) % 2: continue
            B = A + 1
            # Handle or delete cyclic term
            if A + 1 == hamiltonian.N:
                if hamiltonian.is_cyclic and hamiltonian.N > 2:
                    B = 0
                else:
                    continue 
            circuit_odd.add_gate(T=0,  key=A, gate=quasar.Gate.Ry(theta=0.0))
            circuit_odd.add_gate(T=0,  key=B, gate=quasar.Gate.Ry(theta=0.0))
            # circuit_odd.add_gate(T=1,  key=(A,B), gate=quasar.Gate.CNOT)
            circuit_odd.add_gate(T=1,  key=(A,B), gate=quasar.Gate.CZ)
            circuit_odd.add_gate(T=2,  key=A, gate=quasar.Gate.Ry(theta=0.0))
            circuit_odd.add_gate(T=2,  key=B, gate=quasar.Gate.Ry(theta=0.0))
            # circuit_odd.add_gate(T=3,  key=(A,B), gate=quasar.Gate.CNOT)
            circuit_odd.add_gate(T=3,  key=(A,B), gate=quasar.Gate.CZ)
            circuit_odd.add_gate(T=4,  key=A, gate=quasar.Gate.Ry(theta=0.0))
            circuit_odd.add_gate(T=4,  key=B, gate=quasar.Gate.Ry(theta=0.0))

        # Remove redundant Ry gates if requested
        if nonredundant and hamiltonian.N > 2:
            circuit_odd = circuit_odd.subset(Ts=range(1,5))

        circuit = quasar.Circuit.concatenate([circuit_even, circuit_odd])

        return circuit

    @staticmethod
    def build_vqe_circuit_mark2(hamiltonian, nonredundant=True):
        
        """ From https://arxiv.org/pdf/1203.0722.pdf """

        # Validity checks
        if hamiltonian.N % 2: 
            raise RuntimeError('Currently only set up for N even')
        if not hamiltonian.is_linear and not hamiltonian.is_cyclic:
            raise RuntimeError('Hamiltonian must be linear or cyclic')

        # 2-body circuit (even)
        circuit_even = quasar.Circuit(N=hamiltonian.N)
        for A in range(hamiltonian.N):
            if A % 2: continue
            B = A + 1
            circuit_even.add_gate(T=0,  key=A, gate=quasar.Gate.Ry(theta=0.0))
            circuit_even.add_gate(T=0,  key=B, gate=quasar.Gate.Ry(theta=0.0))
            # circuit_even.add_gate(T=1,  key=(A,B), gate=quasar.Gate.CNOT)
            circuit_even.add_gate(T=1,  key=(A,B), gate=quasar.Gate.CZ)
            circuit_even.add_gate(T=2,  key=A, gate=quasar.Gate.Ry(theta=0.0))
            circuit_even.add_gate(T=2,  key=B, gate=quasar.Gate.Ry(theta=0.0))

        # 2-body circuit (odd)
        circuit_odd = quasar.Circuit(N=hamiltonian.N)
        for A in range(hamiltonian.N):
            if (A + 1) % 2: continue
            B = A + 1
            # Handle or delete cyclic term
            if A + 1 == hamiltonian.N:
                if hamiltonian.is_cyclic and hamiltonian.N > 2:
                    B = 0
                else:
                    continue 
            circuit_odd.add_gate(T=0,  key=A, gate=quasar.Gate.Ry(theta=0.0))
            circuit_odd.add_gate(T=0,  key=B, gate=quasar.Gate.Ry(theta=0.0))
            # circuit_odd.add_gate(T=1,  key=(A,B), gate=quasar.Gate.CNOT)
            circuit_odd.add_gate(T=1,  key=(A,B), gate=quasar.Gate.CZ)
            circuit_odd.add_gate(T=2,  key=A, gate=quasar.Gate.Ry(theta=0.0))
            circuit_odd.add_gate(T=2,  key=B, gate=quasar.Gate.Ry(theta=0.0))

        # Remove redundant Ry gates if requested
        if nonredundant and hamiltonian.N > 2:
            circuit_odd = circuit_odd.subset(Ts=range(1,3))

        circuit = quasar.Circuit.concatenate([circuit_even, circuit_odd])

        return circuit

    # => MC-VQE Subspace Hamiltonian <= #

    def subspace_eigenproblem(self):

        # Subspace hamiltonian
        H, D = AIEM.compute_subspace_hamiltonian(
            backend=self.backend,
            nmeasurement=self.nmeasurement_subspace,
            hamiltonian=self.hamiltonian_pauli,
            vqe_circuit=self.vqe_circuit,
            Cs=self.cis_C,
            cis_circuit_function=self.cis_circuit_function,
            )

        # Subspace eigensolve
        E, V = np.linalg.eigh(H)
        C = np.dot(self.cis_C, V)

        # Attribute assignment
        self.vqe_E = E
        self.vqe_C = C
        self.vqe_V = V
        self.vqe_H = H
        self.vqe_D = D

    @staticmethod
    def compute_subspace_hamiltonian(
        backend,
        nmeasurement,
        hamiltonian,
        vqe_circuit,
        Cs,
        cis_circuit_function,
        ):
    
        H = np.zeros((Cs.shape[1],)*2)
        D = {}
        for I in range(Cs.shape[1]):
            C = Cs[:,I]
            thetas = AIEM.compute_cis_angles(cs=C)
            cis_circuit = cis_circuit_function(thetas=thetas)
            circuit = quasar.Circuit.concatenate([cis_circuit, vqe_circuit])
            E, D2 = Collocation.compute_energy_and_pauli_dm( 
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                circuit=circuit,
                ) 
            H[I,I] = E
            D[I,I] = D2
        for I in range(Cs.shape[1]):
            for J in range(I):
                Cp = (Cs[:,I] + Cs[:,J]) / np.sqrt(2.0)
                Cm = (Cs[:,I] - Cs[:,J]) / np.sqrt(2.0)
                thetasp = AIEM.compute_cis_angles(cs=Cp)
                cis_circuitp = cis_circuit_function(thetas=thetasp)
                circuitp = quasar.Circuit.concatenate([cis_circuitp, vqe_circuit])
                Ep, Dp = Collocation.compute_energy_and_pauli_dm( 
                    backend=backend,
                    nmeasurement=nmeasurement,
                    hamiltonian=hamiltonian,
                    circuit=circuitp,
                    ) 
                thetasm = AIEM.compute_cis_angles(cs=Cm)
                cis_circuitm = cis_circuit_function(thetas=thetasm)
                circuitm = quasar.Circuit.concatenate([cis_circuitm, vqe_circuit])
                Em, Dm = Collocation.compute_energy_and_pauli_dm( 
                    backend=backend,
                    nmeasurement=nmeasurement,
                    hamiltonian=hamiltonian,
                    circuit=circuitm,
                    ) 
                H[I,J] = H[J,I] = 0.5 * (Ep - Em)
                D[I,J] = D[J,I] = 0.5 * (Dp - Dm)
        return H, D

    # => Rotation of VQE Pauli DMs (expensive) to VQE eigenbasis <= #

    def rotate_vqe_dms(self):

        self.vqe_D2 = {}
        for I in range(self.nstate):
            for J in range(self.nstate):
                G = np.outer(self.vqe_V[:,I], self.vqe_V[:,J])
                D = Pauli.zeros_like(self.hamiltonian_pauli)
                for I2 in range(self.nstate):
                    for J2 in range(self.nstate):
                        D += G[I2, J2] * self.vqe_D[I2, J2]
                self.vqe_D2[I, J] = D

    # => FCI Utility (Classical) <= #

    @staticmethod
    def compute_fci_hamiltonian(
        hamiltonian,
        ):

        """ Compute the explicit monomer-basis Hamiltonian

        Returns:
            (np.ndarray of shape (2**N,)*2 and dtype of np.float64) - real,
                symmetric monomer-basis Hamiltonian, normal ordered
                (vacuum energy is not included).
        """
    
        N = hamiltonian.N
        H = np.zeros((2**N,)*2)
    
        # One-body term
        for A in range(N):
            A2 = N - A - 1 # Backward ordering in QIS
            pA = [I for I in range(2**N) if not I & (1 << A2)] # |0_A> (+1)
            nA = [I for I in range(2**N) if I & (1 << A2)]     # |1_A> (-1)
            # X
            H[pA,nA] += hamiltonian.X[A]
            H[nA,pA] += hamiltonian.X[A]
            # Z
            H[pA,pA] += hamiltonian.Z[A]
            H[nA,nA] -= hamiltonian.Z[A]
        
        # Two-body term
        for A, B in hamiltonian.ABs:
            if A > B: continue
            A2 = N - A - 1 # Backward ordering in QIS
            B2 = N - B - 1 # Backward ordering in QIS
            pApB = [I for I in range(2**N) if (not I & (1 << A2)) and (not I & (1 << B2))]
            pAnB = [I for I in range(2**N) if (not I & (1 << A2)) and (I & (1 << B2))]
            nApB = [I for I in range(2**N) if (I & (1 << A2)) and (not I & (1 << B2))]
            nAnB = [I for I in range(2**N) if (I & (1 << A2)) and (I & (1 << B2))]
            # XX
            H[pApB,nAnB] += hamiltonian.XX[A,B]
            H[pAnB,nApB] += hamiltonian.XX[A,B]
            H[nApB,pAnB] += hamiltonian.XX[A,B]
            H[nAnB,pApB] += hamiltonian.XX[A,B]
            # XZ
            H[pApB,nApB] += hamiltonian.XZ[A,B]
            H[nApB,pApB] += hamiltonian.XZ[A,B]
            H[pAnB,nAnB] -= hamiltonian.XZ[A,B]
            H[nAnB,pAnB] -= hamiltonian.XZ[A,B]
            # ZX
            H[pApB,pAnB] += hamiltonian.ZX[A,B]
            H[pAnB,pApB] += hamiltonian.ZX[A,B]
            H[nApB,nAnB] -= hamiltonian.ZX[A,B]
            H[nAnB,nApB] -= hamiltonian.ZX[A,B]
            # ZZ
            H[pApB,pApB] += hamiltonian.ZZ[A,B]
            H[pAnB,pAnB] -= hamiltonian.ZZ[A,B]
            H[nApB,nApB] -= hamiltonian.ZZ[A,B]
            H[nAnB,nAnB] += hamiltonian.ZZ[A,B]
            
        return H

    @staticmethod
    def compute_fci_sigma(
        hamiltonian,
        wfn,
        ):

        N = hamiltonian.N
        sigma = np.zeros_like(wfn)

        # One-body term
        for A in range(N):
            A2 = N - A - 1 # Backward ordering in QIS
            pA = [I for I in range(2**N) if not I & (1 << A2)] # |0_A> (+1)
            nA = [I for I in range(2**N) if I & (1 << A2)]     # |1_A> (-1)
            # X
            sigma[pA] += hamiltonian.X[A] * wfn[nA]
            sigma[nA] += hamiltonian.X[A] * wfn[pA]
            # Z
            sigma[pA] += hamiltonian.Z[A] * wfn[pA]
            sigma[nA] -= hamiltonian.Z[A] * wfn[nA]

        # Two-body term
        for A, B in hamiltonian.ABs:
            if A > B: continue
            A2 = N - A - 1 # Backward ordering in QIS
            B2 = N - B - 1 # Backward ordering in QIS
            pApB = [I for I in range(2**N) if (not I & (1 << A2)) and (not I & (1 << B2))]
            pAnB = [I for I in range(2**N) if (not I & (1 << A2)) and (I & (1 << B2))]
            nApB = [I for I in range(2**N) if (I & (1 << A2)) and (not I & (1 << B2))]
            nAnB = [I for I in range(2**N) if (I & (1 << A2)) and (I & (1 << B2))]
            # XX
            sigma[pApB] += hamiltonian.XX[A,B] * wfn[nAnB]
            sigma[pAnB] += hamiltonian.XX[A,B] * wfn[nApB]
            sigma[nApB] += hamiltonian.XX[A,B] * wfn[pAnB]
            sigma[nAnB] += hamiltonian.XX[A,B] * wfn[pApB]
            # XZ
            sigma[pApB] += hamiltonian.XZ[A,B] * wfn[nApB]
            sigma[nApB] += hamiltonian.XZ[A,B] * wfn[pApB]
            sigma[pAnB] -= hamiltonian.XZ[A,B] * wfn[nAnB]
            sigma[nAnB] -= hamiltonian.XZ[A,B] * wfn[pAnB]
            # ZX
            sigma[pApB] += hamiltonian.ZX[A,B] * wfn[pAnB]
            sigma[pAnB] += hamiltonian.ZX[A,B] * wfn[pApB]
            sigma[nApB] -= hamiltonian.ZX[A,B] * wfn[nAnB]
            sigma[nAnB] -= hamiltonian.ZX[A,B] * wfn[nApB]
            # ZZ
            sigma[pApB] += hamiltonian.ZZ[A,B] * wfn[pApB]
            sigma[pAnB] -= hamiltonian.ZZ[A,B] * wfn[pAnB]
            sigma[nApB] -= hamiltonian.ZZ[A,B] * wfn[nApB]
            sigma[nAnB] += hamiltonian.ZZ[A,B] * wfn[nAnB]

        return sigma

    # => FCI Diagonalization Utility <= #

    @staticmethod
    def compute_fci(
        hamiltonian,
        nstate,
        crossover=14,
        ):
        
        if hamiltonian.N < crossover: return AIEM.compute_fci_explicit(hamiltonian=hamiltonian, nstate=nstate)
        else: return AIEM.compute_fci_davidson(hamiltonian=hamiltonian, nstate=nstate)

    @staticmethod
    def compute_fci_explicit(
        hamiltonian,
        nstate,
        ):

        Hfci = AIEM.compute_fci_hamiltonian(hamiltonian=hamiltonian) 
        Efci, Vfci = np.linalg.eigh(Hfci)

        return Efci[:nstate], Vfci[:,:nstate]

    @staticmethod
    def compute_fci_davidson(
        hamiltonian,
        nstate,
        r_convergence=1.0E-7,
        norm_cutoff=1.0E-6,
        maxiter=200,
        subspace_M=10,
        ):

        # People might not have this
        import lightspeed as ls

        print('=> Davidson: <=\n')
        
        # CIS Guess
        guesses = []
        ref = np.zeros((2**hamiltonian.N,))
        ref[0] = 1.0
        guesses.append(ref)
        for A in range(hamiltonian.N):
            wfn = np.zeros((2**hamiltonian.N,))
            wfn[1 << A] = 1.0
            guesses.append(wfn)

        # => Davidson <= #

        dav = ls.Davidson(
            hamiltonian.N+1,
            subspace_M*(hamiltonian.N+1), 
            r_convergence,
            norm_cutoff,
            )
        print(dav)

        bs = [x for x in guesses]

        for iteration in range(maxiter):

            sigmas = [AIEM.compute_fci_sigma(hamiltonian=hamiltonian, wfn=b) for b in bs]

            Rs, Es = dav.add_vectors(
                [ls.Tensor.array(x) for x in bs],
                [ls.Tensor.array(x) for x in sigmas],
                )
            
            print('%4d: %11.3E' % (iteration, dav.max_rnorm))

            if dav.is_converged:
                converged = True
                break

            # No preconditioning
            Ds = Rs
        
            bs = dav.add_preconditioned(
                [ls.Tensor.array(x) for x in Ds],
                )

        if converged:
            print('\nDavidson Converged\n')
        else:
            print('\nDavidson Failed\n')

        wfns = np.array([np.array(ls.Storage.from_storage(x, bs[0])) for x in dav.evecs]).T
        Es = np.array([x for x in dav.evals])

        print('=> End Davidson: <=\n')

        return Es, wfns

    # => Explicit Wavefunction Simulation (Quasar) <= #

    def compute_fci_wavefunction(self, I=0):
        return self.fci_C[:,I]

    def compute_vqe_wavefunction(self, I=0):
        circuit = quasar.Circuit.concatenate([self.vqe_circuits[I], self.vqe_circuit]).compressed()
        wfn = circuit.simulate()
        return wfn

    def compute_cis_wavefunction(self, I=0):
        circuit = self.cis_circuits[I].compressed()
        wfn = circuit.simulate()
        return wfn

    # => Explicit Wavefunction Overlaps <= #

    @memoized_property
    def fci_cis_overlaps(self):
        fci_wfns = np.array([self.compute_fci_wavefunction(I) for I in range(self.nstate)])
        cis_wfns = np.array([self.compute_cis_wavefunction(I) for I in range(self.nstate)])
        return np.dot(fci_wfns.conj(), cis_wfns.T).real

    @memoized_property
    def fci_vqe_overlaps(self):
        fci_wfns = np.array([self.compute_fci_wavefunction(I) for I in range(self.nstate)])
        vqe_wfns = np.array([self.compute_vqe_wavefunction(I) for I in range(self.nstate)])
        return np.dot(fci_wfns.conj(), vqe_wfns.T).real

    @memoized_property
    def vqe_cis_overlaps(self):
        vqe_wfns = np.array([self.compute_vqe_wavefunction(I) for I in range(self.nstate)])
        cis_wfns = np.array([self.compute_cis_wavefunction(I) for I in range(self.nstate)])
        return np.dot(vqe_wfns.conj(), cis_wfns.T).real

    # => Density Matrix Wrappers (AIEMPauli Basis) <= #

    def compute_fci_dm(self, I=0, relaxed=False):

        fci_Cs = [self.fci_C[:, I]]
        fci_ws = [1.0]

        # Relaxed/unrelaxed are identical
        return AIEM.compute_fci_unrelaxed_dm(
            hamiltonian=self.aiem_hamiltonian_pauli,
            fci_Cs=fci_Cs,
            fci_ws=fci_ws,
            )

    def compute_fci_tdm(self, I=0, J=1, relaxed=False):

        if I == J: raise RuntimeError('Can only compute tdm for I != J')

        fci_Cp = (self.fci_C[:, I] + self.fci_C[:, J]) / np.sqrt(2.0)
        fci_Cm = (self.fci_C[:, I] - self.fci_C[:, J]) / np.sqrt(2.0)
        fci_Cs = [fci_Cp, fci_Cm]
        fci_ws = [0.5, -0.5]

        # Relaxed/unrelaxed are identical
        return AIEM.compute_fci_unrelaxed_dm(
            hamiltonian=self.aiem_hamiltonian_pauli,
            fci_Cs=fci_Cs,
            fci_ws=fci_ws,
            )

    def compute_cis_dm(self, I=0, relaxed=False):

        cis_Cs = [self.cis_C[:, I]]
        cis_ws = [1.0]

        # Relaxed/unrelaxed are identical
        return AIEM.compute_cis_unrelaxed_dm(
            hamiltonian=self.aiem_hamiltonian_pauli,
            cis_Cs=cis_Cs,
            cis_ws=cis_ws,
            cis_circuit_function=self.cis_circuit_function,
            )

    def compute_cis_tdm(self, I=0, J=1, relaxed=False):

        if I == J: raise RuntimeError('Can only compute tdm for I != J')

        cis_Cp = (self.cis_C[:, I] + self.cis_C[:, J]) / np.sqrt(2.0)
        cis_Cm = (self.cis_C[:, I] - self.cis_C[:, J]) / np.sqrt(2.0)
        cis_Cs = [cis_Cp, cis_Cm]
        cis_ws = [0.5, -0.5]

        # Relaxed/unrelaxed are identical
        return AIEM.compute_cis_unrelaxed_dm(
            hamiltonian=self.aiem_hamiltonian_pauli,
            cis_Cs=cis_Cs,
            cis_ws=cis_ws,
            cis_circuit_function=self.cis_circuit_function,
            )

    def compute_vqe_dm(self, I=0, relaxed=False):

        if relaxed: raise NotImplemented

        return AIEMUtil.pauli_to_aiem_pauli(self.vqe_D2[I,I])

    def compute_vqe_tdm(self, I=0, J=1, relaxed=False):

        if I == J: raise RuntimeError('Can only compute tdm for I != J')

        if relaxed: raise NotImplemented

        return AIEMUtil.pauli_to_aiem_pauli(self.vqe_D2[I,J])

    # => Unrelaxed Density Matrices <= #        

    @staticmethod
    def compute_fci_unrelaxed_dm(
        hamiltonian,
        fci_Cs,
        fci_ws,
        ):

        pauli_dm = AIEMPauli.zeros_like(hamiltonian)
        for wfn, w in zip(fci_Cs, fci_ws):
            pauli_dm.E += 1.0 * w
            for A in range(pauli_dm.N):
                D = quasar.Circuit.compute_pauli_1(wfn=wfn, A=A)
                pauli_dm.X[A] += D[1] * w
                pauli_dm.Z[A] += D[3] * w
            for A, B in pauli_dm.ABs:
                if A > B: continue
                D = quasar.Circuit.compute_pauli_2(wfn=wfn, A=A, B=B)
                pauli_dm.XX[A,B] += D[1,1] * w
                pauli_dm.XX[B,A] += D[1,1] * w
                pauli_dm.XZ[A,B] += D[1,3] * w
                pauli_dm.ZX[B,A] += D[1,3] * w
                pauli_dm.ZX[A,B] += D[3,1] * w
                pauli_dm.XZ[B,A] += D[3,1] * w
                pauli_dm.ZZ[A,B] += D[3,3] * w
                pauli_dm.ZZ[B,A] += D[3,3] * w
        return pauli_dm

    @staticmethod
    def compute_cis_unrelaxed_dm(
        hamiltonian,
        cis_Cs,
        cis_ws,
        cis_circuit_function,
        ):

        pauli_dm = AIEMPauli.zeros_like(hamiltonian)
        for C, w in zip(cis_Cs, cis_ws):
            thetas = AIEM.compute_cis_angles(cs=C)
            cis_circuit = cis_circuit_function(thetas=thetas)
            circuit = cis_circuit.compressed()
            wfn = circuit.simulate()
            pauli_dm.E += 1.0 * w
            for A in range(pauli_dm.N):
                D = quasar.Circuit.compute_pauli_1(wfn=wfn, A=A)
                pauli_dm.X[A] += D[1] * w
                pauli_dm.Z[A] += D[3] * w
            for A, B in pauli_dm.ABs:
                if A > B: continue
                D = quasar.Circuit.compute_pauli_2(wfn=wfn, A=A, B=B)
                pauli_dm.XX[A,B] += D[1,1] * w
                pauli_dm.XX[B,A] += D[1,1] * w
                pauli_dm.XZ[A,B] += D[1,3] * w
                pauli_dm.ZX[B,A] += D[1,3] * w
                pauli_dm.ZX[A,B] += D[3,1] * w
                pauli_dm.XZ[B,A] += D[3,1] * w
                pauli_dm.ZZ[A,B] += D[3,3] * w
                pauli_dm.ZZ[B,A] += D[3,3] * w
        return pauli_dm
        
