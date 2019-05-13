import time
import itertools
import numpy as np
from ..util import options    
from . import collocation
from .parameters import IdentityParameterGroup

# ==> Jacobi Optimization Utility Classes <== #

class Jacobi(object):

    @staticmethod
    def quad_x(D=1):
    
        return np.array(np.meshgrid(
            *[[-np.pi / 4.0, 0.0, +np.pi / 4.0]]*D,
            indexing='ij',
            ))
    
    @staticmethod
    def quad_transfer(D=1):
    
        T1 = np.array([
            [1, 0, -1,],
            [1, 1,  0,],
            [1, 0, +1,],
            ], dtype=np.float)
    
        T = np.copy(T1)
        for D2 in range(1,D):
            T = np.kron(T, T1)
        T *= 0.5**D
    
        return T
    
    @staticmethod
    def quad_transfer_inv(D=1):
    
        T1inv = np.array([
            [+1, 0, +1,],
            [-1, 2, -1,],
            [-1, 0, +1,],
            ], dtype=np.float)
    
        Tinv = np.copy(T1inv)
        for D2 in range(1,D):
            Tinv = np.kron(Tinv, T1inv)
        Tinv *= 0.5**D
    
        return Tinv
    
    @staticmethod
    def quad_coefs(O):
    
        return np.reshape(np.dot(Jacobi.quad_transfer_inv(D=O.ndim), O.ravel()), O.shape)
    
    @staticmethod
    def quad_collocation(
        coefs,
        thetas,
        ):
    
        if thetas.ndim < 2: raise RuntimeError('thetas.ndim < 2')
        if thetas.shape[0] != coefs.ndim: raise RuntimeError('thetas.shape[0] != coefs.ndim')
    
        # NOTE: Special case (bare coefficent)
        if coefs.ndim == 0: return coefs
    
        bs = []
        for theta in thetas:
            bs.append([
                np.ones_like(theta),
                np.cos(2.0 * theta),
                np.sin(2.0 * theta),  
                ])
    
        O = np.zeros_like(thetas[0])
        for Js in itertools.product(range(3), repeat=coefs.ndim):
            R = np.ones_like(thetas[0])
            for J2, J in enumerate(Js):
                R *= bs[J2][J]
            O += coefs[Js] * R
    
        return O
    
    @staticmethod
    def quad_gradient(
        coefs,
        thetas,
        ):
    
        if thetas.ndim < 2: raise RuntimeError('thetas.ndim < 2')
        if thetas.shape[0] != coefs.ndim: raise RuntimeError('thetas.shape[0] != coefs.ndim')
    
        G = np.zeros((coefs.ndim,) + thetas[0].shape)
        for k in range(coefs.ndim):
            notk = [_ for _ in range(coefs.ndim) if _ != k] 
            thetask = thetas[k]
            thetas2 = thetas[notk]
            coefs_b = np.take(coefs, 1, k)
            coefs_c = np.take(coefs, 2, k)
            Ob = Jacobi.quad_collocation(coefs=coefs_b, thetas=thetas2)
            Oc = Jacobi.quad_collocation(coefs=coefs_c, thetas=thetas2)
            G[k] = -2.0 * np.sin(2.0 * thetask) * Ob + 2.0 * np.cos(2.0 * thetask) * Oc
        return G
    
    @staticmethod
    def quad_hessian(
        coefs,
        thetas,
        ):
    
        if thetas.ndim < 2: raise RuntimeError('thetas.ndim < 2')
        if thetas.shape[0] != coefs.ndim: raise RuntimeError('thetas.shape[0] != coefs.ndim')
    
        H = np.zeros((coefs.ndim,)*2 + thetas[0].shape)
        for k in range(coefs.ndim):
            notk = [_ for _ in range(coefs.ndim) if _ != k] 
            thetask = thetas[k]
            thetas2 = thetas[notk]
            coefs_b = np.take(coefs, 1, k)
            coefs_c = np.take(coefs, 2, k)
            Ob = Jacobi.quad_collocation(coefs=coefs_b, thetas=thetas2)
            Oc = Jacobi.quad_collocation(coefs=coefs_c, thetas=thetas2)
            if len(notk):
                Gb = Jacobi.quad_gradient(coefs=coefs_b, thetas=thetas2)
                Gc = Jacobi.quad_gradient(coefs=coefs_c, thetas=thetas2)
                H[k,notk] = -2.0 * np.sin(2.0 * thetask) * Gb + 2.0 * np.cos(2.0 * thetask) * Gc
            H[k, k] = - 4.0 * np.cos(2.0 * thetask) * Ob - 4.0 * np.sin(2.0 * thetask) * Oc
        return H 
    
    # => Optimization <= #
    
    @staticmethod
    def optimize_jacobi_1(
        coefs,
        theta0=None,
        n=100,
        d=0,
        ):
    
        if theta0 is None:
            theta0 = np.zeros((coefs.ndim,))
    
        theta = np.copy(theta0)
        thetas = [theta0]
        for iteration in range(n):
            k = (iteration + d) % coefs.ndim
            theta_2 = np.array([theta for k2, theta in enumerate(theta) if k2 != k])
            theta_2 = np.reshape(theta_2, theta_2.shape + (1,))
            coefs_b = np.take(coefs, 1, k)
            coefs_c = np.take(coefs, 2, k)
            Ob = Jacobi.quad_collocation(coefs=coefs_b, thetas=theta_2)
            Oc = Jacobi.quad_collocation(coefs=coefs_c, thetas=theta_2)
            theta[k] = 0.5 * np.arctan2(-Oc, -Ob)
            thetas.append(np.copy(theta))
        thetas = np.array(thetas)
        return thetas.T
    
    @staticmethod
    def optimize_jacobi_1_best(
        coefs,
        theta0=None,
        n=100,
        ):
    
        thetas = []
        for d in range(coefs.ndim):
            thetas.append(Jacobi.optimize_jacobi_1(
                coefs=coefs,
                theta0=theta0,  
                n=n,    
                d=d,
                ))
        Os = np.array([Jacobi.quad_collocation(coefs=coefs, thetas=theta2[:,-2:-1]) for theta2 in thetas])
        return thetas[np.argmin(Os)] 
    
    @staticmethod
    def optimize_nr(
        coefs,
        theta0=None,
        n=20,
        ): 
    
        if theta0 is None:
            theta0 = np.zeros((coefs.ndim,))
    
        theta = np.copy(theta0)
        thetas = [theta0]
        for iteration in range(n):
            theta2 = np.reshape(theta, theta.shape + (1,))
            G = Jacobi.quad_gradient(coefs=coefs, thetas=theta2)[:,0]
            H = Jacobi.quad_hessian(coefs=coefs, thetas=theta2)[:,:,0]
            h, U = np.linalg.eigh(H)
            hinv = 1.0 / h
            hinv[h < 1.0E-10] = 0.0
            if any(h < 1.0E-10): print("Negative")
            hinv = np.diag(hinv)
            theta = theta - np.dot(U, np.dot(hinv, np.dot(U.T, G)))
            thetas.append(np.copy(theta))
        thetas = np.array(thetas)
        return thetas.T
            
    @staticmethod
    def optimize_hybrid(
        coefs,
        theta0=None,
        njacobi=100,
        nnr=20,
        ):
    
        thetas_jacobi = Jacobi.optimize_jacobi_1_best(coefs=coefs, theta0=theta0, n=njacobi)
        thetas_nr = Jacobi.optimize_nr(coefs=coefs, theta0=thetas_jacobi[:,-1], n=nnr)
        return np.hstack((thetas_jacobi, thetas_nr))    
    
    @staticmethod
    def optimize_hybrid2(
        coefs,
        ntheta=8,
        njacobi=100,
        nnr=20,
        ):
    
        thetas = np.meshgrid(
            *[np.linspace(0.0, np.pi, ntheta) - 0.5 * np.pi]*coefs.ndim,
            indexing='ij',
            )
        thetas = np.array(thetas)
    
        Os = Jacobi.quad_collocation(
            coefs=coefs,
            thetas=thetas,
            )
    
        ind = np.unravel_index(np.argmin(Os, axis=None), Os.shape)
        theta0 = np.array([thetas[A][ind] for A in range(len(ind))])
    
        return Jacobi.optimize_hybrid(
            coefs=coefs,
            theta0=theta0,
            njacobi=njacobi,
            nnr=nnr,
            )

class DIIS(object):

    def __init__(
        self,
        M=6,    
        ):

        self.M = M
        self.E = np.zeros((M,M))

        self.state_vecs = []
        self.error_vecs = []

    def extrapolate(
        self,
        state_vec,
        error_vec,
        ):

        if len(self.state_vecs) == self.M:
            I = np.argmax(np.diag(self.E))      
            self.state_vecs[I] = state_vec
            self.error_vecs[I] = error_vec
        else:
            I = len(self.state_vecs)
            self.state_vecs.append(state_vec)
            self.error_vecs.append(error_vec)
        
        for J in range(len(self.state_vecs)):
            self.E[I,J] = self.E[J,I] = np.sum(self.error_vecs[I] * self.error_vecs[J])

        B = np.zeros((len(self.state_vecs)+1,)*2)
        B[:-1,:-1] = self.E[:len(self.state_vecs),:len(self.state_vecs)]
        B[-1,:-1] = 1
        B[:-1,-1] = 1
    
        R = np.zeros((len(self.state_vecs)+1,)*1)
        R[-1] = 1
        
        L = np.linalg.solve(B, R)

        D = L[:-1]

        F = np.zeros_like(self.state_vecs[0])
        for Dval, Fval in zip(D, self.state_vecs):
            F += Dval * Fval

        return F

# ==> VQA Optimizers <== #

class Optimizer(object):

    pass

class BFGSOptimizer(Optimizer):

    @staticmethod
    def default_options():
        
        if hasattr(BFGSOptimizer, '_default_options'): return BFGSOptimizer._default_options.copy()
        opt = options.Options() 

        opt.add_option(
            key='maxiter',
            value=100,
            allowed_types=[int],
            doc='Quantum simulator or hardware backend')
        opt.add_option(
            key='g_convergence',
            value=1.0E-7,
            allowed_types=[float],
            doc='Maximum gradient element criteria for convergence')
            
        BFGSOptimizer._default_options = opt
        return BFGSOptimizer._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ BFGSOptimizer initialization - no computational effort performed. 
        """
        
        self.options = options

        
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return BFGSOptimizer(BFGSOptimizer.default_options().set_values(kwargs))

    def optimize(
        self,
        print_level,
        backend,
        nmeasurement,
        hamiltonian,
        reference_circuits,
        reference_weights,
        entangler_circuit,
        entangler_circuit_parameter_group,
        guess_params,
        ):

        if print_level:
            print(' > BFGS Optimizer <\n')

        if print_level:
            print('  %-13s = %d' % ('maxiter', self.options['maxiter']))
            print('  %-13s = %.3E' % ('g_convergence', self.options['g_convergence']))
            print('')

        entangler_circuit2 = entangler_circuit.copy()
        
        entangler_history = []
        entangler_history.append(np.array(guess_params))
    
        def sa_energy_helper(Z):
            Z2 = entangler_circuit_parameter_group.compute_raw(Z)
            entangler_circuit2.set_param_values(Z2)
            return collocation.Collocation.compute_sa_energy_and_pauli_dm(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                circuit=entangler_circuit2,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                )[0]
    
        def sa_gradient_helper(Z):
            Z2 = entangler_circuit_parameter_group.compute_raw(Z)
            entangler_circuit2.set_param_values(Z2)
            return collocation.Collocation.compute_sa_gradient(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                circuit=entangler_circuit2,
                parameter_group=entangler_circuit_parameter_group,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                )

        def sa_history_helper(Z):
            entangler_history.append(np.copy(Z))

        Z = np.array(guess_params)

        import scipy.optimize
        Z, sa_E, opt_data = scipy.optimize.fmin_l_bfgs_b(
            x0=Z,
            func=sa_energy_helper,
            fprime=sa_gradient_helper,
            iprint=print_level > 0,   
            maxiter=self.options['maxiter'],
            pgtol=self.options['g_convergence'],
            factr=1.0E-6, # Prevent energy difference from being used as stopping criteria
            callback=sa_history_helper,
            )
        Z = np.array(Z)
        Z2 = entangler_circuit_parameter_group.compute_raw(Z)
        entangler_circuit2.set_param_values(Z2)

        if print_level:
            print(' > End BFGS Optimizer <\n')

        return Z, entangler_circuit2, np.array(entangler_history)
        
class PowellOptimizer(Optimizer):

    @staticmethod
    def default_options():
        
        if hasattr(PowellOptimizer, '_default_options'): return PowellOptimizer._default_options.copy()
        opt = options.Options() 

        opt.add_option(
            key='maxiter',
            value=100,
            allowed_types=[int],
            doc='Quantum simulator or hardware backend')
        opt.add_option(
            key='ftol',
            value=1.0E-16,
            allowed_types=[float],
            doc='ftol criteria for stopping Powell iterations')
        opt.add_option(
            key='xtol',
            value=1.0E-6,
            allowed_types=[float],
            doc='xtol criteria for stopping Powell linesearches')
            
        PowellOptimizer._default_options = opt
        return PowellOptimizer._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ PowellOptimizer initialization - no computational effort performed. 
        """
        
        self.options = options

    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return PowellOptimizer(PowellOptimizer.default_options().set_values(kwargs))

    def optimize(
        self,
        print_level,
        backend,
        nmeasurement,
        hamiltonian,
        reference_circuits,
        reference_weights,
        entangler_circuit,
        entangler_circuit_parameter_group,
        guess_params,
        ):

        if print_level:
            print(' > Powell Optimizer <\n')

        if print_level:
            print('  %-13s = %d' % ('maxiter', self.options['maxiter']))
            print('  %-13s = %.3E' % ('ftol', self.options['ftol']))
            print('  %-13s = %.3E' % ('xtol', self.options['xtol']))
            print('')

        entangler_circuit2 = entangler_circuit.copy()
        
        entangler_history = []
        entangler_history.append(np.array(guess_params))
    
        def sa_energy_helper(Z):
            Z2 = entangler_circuit_parameter_group.compute_raw(Z)
            entangler_circuit2.set_param_values(Z2)
            return collocation.Collocation.compute_sa_energy_and_pauli_dm(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                circuit=entangler_circuit2,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                )[0]

        def sa_history_helper(Z):
            entangler_history.append(np.copy(Z))

        Z = np.array(guess_params)

        import scipy.optimize
        result = scipy.optimize.minimize(
            method='powell',
            x0=Z,
            fun=sa_energy_helper,
            options={
                'maxiter' : self.options['maxiter'] , 
                'disp' : True , 
                'ftol' : 1.0E-16, 
                'xtol' : 1.0E-10,
                },
            callback=sa_history_helper,
            )
        Z = np.array(result.x)
        Z2 = entangler_circuit_parameter_group.compute_raw(Z)
        entangler_circuit2.set_param_values(Z2)

        if print_level:
            print(' > End Powell Optimizer <\n')

        return Z, entangler_circuit2, np.array(entangler_history)


class JacobiOptimizer(Optimizer):

    @staticmethod
    def default_options():
        
        if hasattr(JacobiOptimizer, '_default_options'): return JacobiOptimizer._default_options.copy()
        opt = options.Options() 

        opt.add_option(
            key='maxiter',
            value=100,
            allowed_types=[int],
            doc='Quantum simulator or hardware backend')
        opt.add_option(
            key='g_convergence',
            value=1.0E-7,
            allowed_types=[float],
            doc='Maximum gradient element criteria for convergence')
        opt.add_option(
            key='jacobi_tasks',
            required=False,
            allowed_types=[list],
            doc='List of tuples of parameters to optimize (1st priority)')
        opt.add_option(
            key='jacobi_level',
            value=1,
            required=False,
            allowed_types=[int, str],
            allowed_values=[1, 2, 'gen'],
            doc='Standard Jacobi level (2nd priority) or "gen" to use jacobi_tasks')
        opt.add_option(
            key='jacobi_randomize',
            value=False,
            allowed_types=[bool],
            doc='Randomize Jacobi pivot order on each iteration?')
        opt.add_option(
            key='diis_type',
            value='anderson',
            allowed_types=[str],
            allowed_values=['anderson', 'pulay'],
            doc='Type of DIIS algorithm')
        opt.add_option(
            key='diis_max_vecs',
            value=6,
            allowed_types=[int],
            doc='Maximum size of DIIS history')
        opt.add_option(
            key='diis_flush_vecs',
            value=20,
            allowed_types=[int],
            doc='Number of iterations before DIIS flush')
            
        JacobiOptimizer._default_options = opt
        return JacobiOptimizer._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ JacobiOptimizer initialization - no computational effort performed. 
        """
        
        self.options = options

        
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return JacobiOptimizer(JacobiOptimizer.default_options().set_values(kwargs))

    def optimize(
        self,
        print_level,
        backend,
        nmeasurement,
        hamiltonian,
        reference_circuits,
        reference_weights,
        entangler_circuit,
        entangler_circuit_parameter_group,
        guess_params,
        ):

        if not isinstance(entangler_circuit_parameter_group, IdentityParameterGroup):
            raise RuntimeError('JacobiOptimizer can only be used with VQE-like entanglers with IdentityParameterGroup')

        if print_level:
            print(' > Jacobi Optimizer <\n')

        entangler_circuit2 = entangler_circuit.copy()
        entangler_circuit2.set_param_values(guess_params)
        Told = np.array(entangler_circuit2.param_values)

        entangler_history = []
        entangler_history.append(np.array(entangler_circuit.param_values))

        diis = DIIS(M=self.options['diis_max_vecs'])

        if self.options['jacobi_level'] == 1:
            jacobi_tasks2=[(x,) for x in range(entangler_circuit2.nparam)]
        elif self.options['jacobi_level'] == 2:
            jacobi_tasks2=[(x,y) for x in range(entangler_circuit2.nparam) for y in range(x)]
        elif self.options['jacobi_level'] == 'gen':
            jacobi_tasks2 = self.options['jacobi_tasks'].copy()
        else:
            raise RuntimeError('Unknown Jacobi level: %r' % (jacobi_level))

        start = time.time()
        Eold = 0.0
        converged = False
        if print_level: print('MC-VQE Jacobi Iterations:\n')
        if print_level:
            print('Jacobi Level       = %r' % self.options['jacobi_level'])
            print('Jacobi Randomize   = %r' % self.options['jacobi_randomize'])
            print('DIIS Type          = %s' % self.options['diis_type'])
            print('DIIS Max Vectors   = %d' % self.options['diis_max_vecs'])
            print('DIIS Flush Vectors = %d' % self.options['diis_flush_vecs'])
            print('')
        if print_level:
            print('Jacobi Tasks:')
            for task in jacobi_tasks2:
                print(task)
            print('')
        if print_level: print('%4s: %24s %11s %11s %8s' % ('Iter', 'Energy', 'dE', 'dG', 'Time[s]'))
        for iteration in range(self.options['maxiter']):

            # SA Energy
            E = collocation.Collocation.compute_sa_energy_and_pauli_dm(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                circuit=entangler_circuit2,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                )[0]

            # SA Energy Gradient (to check convergence)
            G = collocation.Collocation.compute_sa_gradient(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                circuit=entangler_circuit2,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                parameter_group=entangler_circuit_parameter_group,
                )

            # Convergence Characteristics
            dE = E - Eold
            Eold = E
            dG = np.max(np.abs(G))

            # Print Iterative Trace
            stop = time.time()
            if print_level: print('%4d: %24.16E %11.3E %11.3E %8.3f' % (iteration, E, dE, dG, stop-start))
            start = stop

            # Check convergence
            if dG < self.options['g_convergence']:
                converged = True
                break

            # DIIS flush
            if iteration > 0 and iteration % self.options['diis_flush_vecs'] == 0:
                diis = DIIS(M=self.options['diis_max_vecs'])

            # DIIS
            T = np.array(entangler_circuit2.param_values)
            dT = T - Told
            if self.options['diis_type'] == 'anderson':
                if (iteration % self.options['diis_flush_vecs']) > 0: T = diis.extrapolate(T, dT)
            elif self.options['diis_type'] == 'pulay':
                T = diis.extrapolate(T, G)
            else:
                raise RuntimeError('Unknown diis type: %s' % diis_type)
            Told = T.copy()
            entangler_circuit2.set_param_values(T)
    
            # Randomization
            if self.options['jacobi_randomize']:
                np.random.shuffle(jacobi_tasks2)

            # Jacobi sweep
            entangler_circuit2 = JacobiOptimizer.sa_energy_jacobi_sweep_gen(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                entangler_circuit=entangler_circuit2,
                jacobi_tasks=jacobi_tasks2,
                )
                
            # History save
            entangler_history.append(np.array(entangler_circuit2.param_values))

        if print_level:
            print('')
            if converged:
                print('Jacobi Iterations Converged\n')
            else:
                print('Jacobi Iterations Failed\n')

        return np.array(entangler_circuit2.param_values), entangler_circuit2, np.array(entangler_history)

    @staticmethod
    def sa_energy_jacobi_sweep_gen(
        backend,
        nmeasurement,
        hamiltonian,
        reference_circuits,
        reference_weights,
        entangler_circuit,
        jacobi_tasks,
        ): 

        entangler_circuit2 = entangler_circuit.copy()
        for task in jacobi_tasks:
            thetas = entangler_circuit2.param_values.copy()
            # Quadrature
            Es = np.zeros((3,)*len(task))
            for Js in itertools.product(range(3), repeat=len(task)):
                thetas2 = thetas.copy()
                for A2, A in enumerate(task):
                    thetas2[A] += (Js[A2] - 1) * np.pi / 4.0
                entangler_circuit2.set_param_values(thetas2)
                Es[Js] = collocation.Collocation.compute_sa_energy_and_pauli_dm(
                    backend=backend,
                    nmeasurement=nmeasurement,
                    hamiltonian=hamiltonian,
                    circuit=entangler_circuit2,
                    reference_circuits=reference_circuits,
                    reference_weights=reference_weights,
                    )[0]
            # Tomography fitting
            coefs = Jacobi.quad_coefs(Es) 
            # Classical parameter optimization
            # thetas_h = Jacobi.optimize_hybrid2(coefs=coefs, njacobi=100, nnr=20, ntheta=8)
            thetas_h = Jacobi.optimize_jacobi_1(coefs=coefs, n=1000)
            theta_opt = thetas_h[:,-1]
            thetas2 = thetas.copy()
            for A2, A in enumerate(task):
                thetas2[A] += theta_opt[A2]
            entangler_circuit2.set_param_values(thetas2)

        return entangler_circuit2

    @staticmethod
    def build_jacobi_tasks(
        entangler_circuit,
        bandwidth=1,
        is_cyclic=False,
        ):

        jacobi_tasks = []
        param_keys = entangler_circuit.param_keys
        for A, keyA in enumerate(param_keys):
            A2 = keyA[1][0]
            for B, keyB in enumerate(param_keys):
                B2 = keyB[1][0]
                if A <= B: continue
                if is_cyclic:
                    distance = min(
                        (B2 - A2) % entangler_circuit.N,
                        (A2 - B2) % entangler_circuit.N,
                        )
                else:
                    distance = abs(B2 - A2)
                assert(distance >= 0)
                if distance <= bandwidth:
                    jacobi_tasks.append((A,B))
        return jacobi_tasks

