import numpy as np
from ..quasar import quasar
from ..util import Options
from ..util import memoized_property
from ..core import Backend
from ..core import QuasarSimulatorBackend
from ..core import Collocation
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
            key='shots',
            value=None,
            allowed_types=[int],
            doc='Number of shots per observable, or None for infinite sampling')

        # > Problem Description < #

        opt.add_option(
            key='nstate',
            required=True,
            allowed_types=[int],
            doc='Number of states to target')
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

        # > Quantum Circuit Selection < #

        opt.add_option(
            key='cis_circuit_type',
            value='mark1',
            required=True,
            allowed_types=[str],
            allowed_values=['mark1', 'mark2'],
            doc='CIS state preparation circuit recipe')
    
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
    def shots(self):
        return self.options['shots']

    @property
    def N(self):
        return self.aiem_monomer.N

    @property
    def ncis(self):
        return self.N + 1 

    @property
    def nstate(self):
        return self.options['nstate']    

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
        return AIEMUtil.aiem_pauli_to_pauli(self.aiem_hamiltonian_pauli)

    @property
    def cis_circuit_type(self):
        return self.options['cis_circuit_type']

    @property
    def cis_circuit_function(self):
        if self.cis_circuit_type == 'mark1': return AIEM.build_cis_circuit_mark1
        elif self.cis_circuit_type == 'mark2': return AIEM.build_cis_circuit_mark2
        else: raise RuntimeError('Unknown cis_circuit_type: %s' % self.cis_circuit_type)

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
            print('  %-7s = %s' % ('Backend', self.backend))
            print('  %-7s = %s' % ('Shots', self.shots))
            print('')

            print('AIEM Problem:') 
            print(' %-12s = %d' % ('N', self.N))
            print(' %-12s = %d' % ('Ncis', self.ncis))
            print(' %-12s = %d' % ('Nstate', self.nstate))
            print(' %-12s = %s' % ('Connectivity', self.aiem_monomer.connectivity_str))
            print('')

            print('CIS Circuits:')
            print('  %16s = %s' % ('cis_circuit_type', self.cis_circuit_type))
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

        self.cis_circuits = [self.cis_circuit_function(
            thetas=thetas,
            ) for thetas in self.cis_angles]

        # > CIS Circuit Verification < #

        if self.print_level:
            print('CIS Verification:\n')
            print('%-5s: %11s' % ('State', 'Match'))
            Ecis2 = self.verify_cis(
                backend=self.backend_quasar_simulator,
                shots=None,
                )[0]
            for I in range(self.nstate):
                dEcis = Ecis2[I] - self.cis_E[I] - self.aiem_hamiltonian_pauli.E
                print('%-5d: %11.3E' % (I, dEcis))
            print('')

        # > Trailer < #

        if self.print_level:
            print('  Viper:    "That\'s pretty arrogant, considering the company you\'re in."')
            print('  Maverick: "Yes, sir."')
            print('  Viper:    "I like that in a pilot."')
            print('')
            print('==> End MC-VQE+AIEM <==\n')

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
        shots,
        ):

        Es = []
        Ds = []
        for I, circuit in enumerate(self.cis_circuits):
            E, D = Collocation.compute_energy_and_pauli_dm(
                backend=backend,
                shots=shots,
                hamiltonian=self.hamiltonian_pauli,
                circuit=circuit,
                )
            D2 = AIEMUtil.pauli_to_aiem_pauli(D)
            print(D2)
            Es.append(E)
            Ds.append(D)
        return Es, Ds
            

