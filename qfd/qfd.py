import numpy as np
import quasar
from quasar import Options
from quasar import memoized_property
from .aiem_data import AIEMMonomer
from .aiem_data import AIEMMonomerGrad
from .aiem_data import AIEMPauli
from .aiem_data import AIEMUtil

class QFD(object):

    @staticmethod

    def default_options():
        
        if hasattr(QFD, '_default_options'): return QFD._default_options.copy()
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
            value=quasar.QuasarSimulatorBackend(),
            required=True,
            allowed_types=[quasar.Backend],
            doc='Quantum simulator or hardware backend')
        opt.add_option(
            key='nmeasurement',
            value=None,
            allowed_types=[int],
            doc='Number of measurements per observable, or None for infinite sampling (backend must support statevector simulation)')

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

        # > QFD Options < #

        opt.add_option(
            key='kappa_method',
            value='gorshgorin',
            allowed_types=[str],
            allowed_values=['gorshgorin', 'evals',],
            doc='Method to determine kappa scaling parameter')
        opt.add_option(
            key='kmax',
            value=1,
            allowed_types=[int],
            doc='Number of k points to include in QFD expansion')

        QFD._default_options = opt
        return QFD._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ QFD initialization - no computational effort performed. 
        """
        
        self.options = options

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return QFD(QFD.default_options().set_values(kwargs))

    @property
    def print_level(self):
        return self.options['print_level']

    @property
    def backend(self):
        return self.options['backend']

    @memoized_property
    def backend_quasar_simulator(self):
        return quasar.QuasarSimulatorBackend()

    @property
    def nmeasurement(self):
        return self.options['nmeasurement']

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
        return AIEMUtil.monomer_to_pauli_hamiltonian(monomer=self.aiem_monomer)

    @memoized_property
    def aiem_mu_pauli(self):
        return AIEMUtil.monomer_to_pauli_dipole(self.aiem_monomer)

    @memoized_property
    def hamiltonian_pauli(self):
        """ Normal Ordered (self energy E removed) """
        return AIEMUtil.aiem_pauli_to_pauli(self.aiem_hamiltonian_pauli, self_energy=False)

    @memoized_property
    def hamiltonian_matrix(self):
        """ Normal Ordered Hamiltonian in computational basis """
        return self.hamiltonian_pauli.compute_hilbert_matrix(dtype=np.float64) 

    @memoized_property
    def hamiltonian_evals(self):
        return np.linalg.eigh(self.hamiltonian_matrix)[0]

    @memoized_property
    def hamiltonian_diags(self):
        return np.diag(self.hamiltonian_matrix)

    @memoized_property
    def hamiltonian_gorshgorin_disks(self):
        return np.sum(np.abs(self.hamiltonian_matrix - np.diag(self.hamiltonian_diags)), 1)
        print(np.diag((self.hamiltonian_matrix - np.diag(self.hamiltonian_diags))))

    @memoized_property
    def kappa_evals(self):
        return np.max(self.hamiltonian_evals) - np.min(self.hamiltonian_evals)

    @memoized_property
    def kappa_gorshgorin(self):
        return self.hamiltonian_diags[-1] - self.hamiltonian_diags[0] + self.hamiltonian_gorshgorin_disks[-1] + self.hamiltonian_gorshgorin_disks[0]        

    @property
    def kmax(self):
        return self.options['kmask']

    def compute_energy(
        self,
        ):

        # > Header < #
    
        if self.print_level:
            print('==> QFD+AIEM <==\n')
    
            print('Quantum Resources:')
            print('  %-14s = %s' % ('Backend', self.backend))
            print('  %-14s = %s' % ('Shots', self.nmeasurement))
            print('')

            print('AIEM Problem:') 
            print('  %-12s = %d' % ('N', self.N))
            print('  %-12s = %d' % ('Ncis', self.ncis))
            print('  %-12s = %d' % ('Nstate', self.nstate))
            print('  %-12s = %s' % ('Connectivity', self.aiem_monomer.connectivity_str))
            print('')
    
        # => CIS States <= #

        # > CIS States < #

        self.cis_E, self.cis_C = QFD.solve_cis(
            hamiltonian=self.aiem_hamiltonian_pauli,
            nstate=self.nstate,
            )

        # > CIS Angles < #        

        self.cis_angles = [QFD.compute_cis_angles(
            cs=self.cis_C[:,T],
            ) for T in range(self.cis_C.shape[1])]

        # > CIS Circuits < #

        self.cis_circuits = [QFD.build_cis_circuit(
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

        # > CIS Statevectors < #

        self.cis_statevectors = [circuit.simulate() for circuit in self.cis_circuits]

        # > CIS Oscillator Strengths < #

        cis_O = []
        for J in range(1, self.nstate):
            T_pauli = self.compute_cis_tdm(I=0, J=J)
            X = AIEMUtil.pauli_energy(pauli_hamiltonian=self.aiem_mu_pauli[0], pauli_dm=T_pauli)
            Y = AIEMUtil.pauli_energy(pauli_hamiltonian=self.aiem_mu_pauli[1], pauli_dm=T_pauli)
            Z = AIEMUtil.pauli_energy(pauli_hamiltonian=self.aiem_mu_pauli[2], pauli_dm=T_pauli)
            O = 2.0 / 3.0 * (self.cis_E[J] - self.cis_E[0]) * (X**2 + Y**2 + Z**2)
            cis_O.append(O)
        self.cis_O = np.array(cis_O)

        # => FCI States <= #

        # > FCI States < #

        self.fci_E, self.fci_C = QFD.compute_fci(
            hamiltonian=self.aiem_hamiltonian_pauli,
            nstate=self.nstate,
            )
        self.fci_statevectors = self.fci_C

        # > FCI Oscillator Strengths < #        

        fci_O = []
        for J in range(1, self.nstate):
            T_pauli = self.compute_fci_tdm(I=0, J=J)
            X = AIEMUtil.pauli_energy(pauli_hamiltonian=self.aiem_mu_pauli[0], pauli_dm=T_pauli)
            Y = AIEMUtil.pauli_energy(pauli_hamiltonian=self.aiem_mu_pauli[1], pauli_dm=T_pauli)
            Z = AIEMUtil.pauli_energy(pauli_hamiltonian=self.aiem_mu_pauli[2], pauli_dm=T_pauli)
            O = 2.0 / 3.0 * (self.fci_E[J] - self.fci_E[0]) * (X**2 + Y**2 + Z**2)
            fci_O.append(O)
        self.fci_O = np.array(fci_O)

        # => Kappa Determination <= #

        if self.options['kappa_method'] == 'evals':
            self.kappa = self.kappa_evals
        elif self.options['kappa_method'] == 'gorshgorin':
            self.kappa = self.kappa_gorshgorin
        else:
            raise RuntimeError('Unknown kappa method: %s' % self.option['kappa_method'])

        if self.print_level:
            print('Kappa Spectral Range:') 
            print('  Kappa method = %11s' % (self.options['kappa_method']))
            print('  Kappa        = %11.3E' % (self.kappa))
            print('  Kappa evals  = %11.3E' % (self.kappa_evals))
            print('  Kappa ratio  = %11.3E' % (self.kappa / self.kappa_evals))
            print('')

        # => 

        

        # > Trailer < #

        if self.print_level:
            print("  Han Solo: Flying through hyperspace ain't like dusting crops, boy!")
            print('')
            print('==> QFD+AIEM <==\n')

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
    def solve_cis(
        hamiltonian,
        nstate,
        ):

        H = QFD.compute_cis_hamiltonian(hamiltonian=hamiltonian)
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

        J = QFD.compute_cis_angles_jacobian(cs=cs)
        return np.dot(J.T, ds)

    # => CIS State Preparation Quantum Circuit <= #

    @staticmethod
    def build_cis_circuit(thetas):
        N = len(thetas)
        circuit = quasar.Circuit(N=N)
        circuit.add_gate(time=0, qubits=0, gate=quasar.Gate.Ry(theta=thetas[0]))
        for I, theta in enumerate(thetas[1:]):
            circuit.add_gate(time=3*I,   qubits=(I+1), gate=quasar.Gate.Ry(theta=-theta/2.0)) 
            circuit.add_gate(time=3*I+1, qubits=(I, I+1), gate=quasar.Gate.CZ)
            circuit.add_gate(time=3*I+2, qubits=(I+1), gate=quasar.Gate.Ry(theta=+theta/2.0)) 
            circuit.add_gate(time=3*I+3, qubits=(I+1, I), gate=quasar.Gate.CX)
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
            E, D = quasar.run_observable_expectation_value_and_pauli(
                backend=backend,
                circuit=circuit,
                pauli=self.hamiltonian_pauli,
                nmeasurement=nmeasurement,
                )
            Es.append(E)
            Ds.append(D)
        return Es, Ds
            
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
        
        if hamiltonian.N < crossover: return QFD.compute_fci_explicit(hamiltonian=hamiltonian, nstate=nstate)
        else: return QFD.compute_fci_davidson(hamiltonian=hamiltonian, nstate=nstate)

    @staticmethod
    def compute_fci_explicit(
        hamiltonian,
        nstate,
        ):

        Hfci = QFD.compute_fci_hamiltonian(hamiltonian=hamiltonian) 
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

            sigmas = [QFD.compute_fci_sigma(hamiltonian=hamiltonian, wfn=b) for b in bs]

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

    # => Density Matrix Wrappers (AIEMPauli Basis) <= #

    def compute_fci_dm(self, I=0, relaxed=False):

        fci_Cs = [self.fci_C[:, I]]
        fci_ws = [1.0]

        # Relaxed/unrelaxed are identical
        return QFD.compute_fci_unrelaxed_dm(
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
        return QFD.compute_fci_unrelaxed_dm(
            hamiltonian=self.aiem_hamiltonian_pauli,
            fci_Cs=fci_Cs,
            fci_ws=fci_ws,
            )

    def compute_cis_dm(self, I=0, relaxed=False):

        cis_Cs = [self.cis_C[:, I]]
        cis_ws = [1.0]

        # Relaxed/unrelaxed are identical
        return QFD.compute_cis_unrelaxed_dm(
            hamiltonian=self.aiem_hamiltonian_pauli,
            cis_Cs=cis_Cs,
            cis_ws=cis_ws,
            )

    def compute_cis_tdm(self, I=0, J=1, relaxed=False):

        if I == J: raise RuntimeError('Can only compute tdm for I != J')

        cis_Cp = (self.cis_C[:, I] + self.cis_C[:, J]) / np.sqrt(2.0)
        cis_Cm = (self.cis_C[:, I] - self.cis_C[:, J]) / np.sqrt(2.0)
        cis_Cs = [cis_Cp, cis_Cm]
        cis_ws = [0.5, -0.5]

        # Relaxed/unrelaxed are identical
        return QFD.compute_cis_unrelaxed_dm(
            hamiltonian=self.aiem_hamiltonian_pauli,
            cis_Cs=cis_Cs,
            cis_ws=cis_ws,
            )

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
        ):

        pauli_dm = AIEMPauli.zeros_like(hamiltonian)
        for C, w in zip(cis_Cs, cis_ws):
            thetas = QFD.compute_cis_angles(cs=C)
            cis_circuit = QFD.build_cis_circuit(thetas=thetas)
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
        
