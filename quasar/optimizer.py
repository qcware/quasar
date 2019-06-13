import numpy as np
import time
from .options import Options
from .parameters import IdentityParameterGroup
from .derivatives import run_ensemble_observable_expectation_value
from .derivatives import run_ensemble_observable_expectation_value_gradient
from .tomography import run_ensemble_observable_expectation_value_tomography

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
        opt = Options() 

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
            return run_ensemble_observable_expectation_value(
                backend=backend,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                circuit=entangler_circuit2,
                pauli=hamiltonian,
                nmeasurement=nmeasurement,
                )
    
        def sa_gradient_helper(Z):
            Z2 = entangler_circuit_parameter_group.compute_raw(Z)
            entangler_circuit2.set_param_values(Z2)
            G1 = run_ensemble_observable_expectation_value_gradient(
                backend=backend,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                circuit=entangler_circuit2,
                pauli=hamiltonian,
                nmeasurement=nmeasurement,
                )
            return entangler_circuit_parameter_group.compute_chain_rule1(Z, G1)

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
        opt = Options() 

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
            return run_ensemble_observable_expectation_value(
                backend=backend,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                circuit=entangler_circuit2,
                pauli=hamiltonian,
                nmeasurement=nmeasurement,
                )

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
        opt = Options() 

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
            E = run_ensemble_observable_expectation_value(
                backend=backend,
                pauli=hamiltonian,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                circuit=entangler_circuit2,
                nmeasurement=nmeasurement,
                )

            # SA Energy Gradient (to check convergence)
            G = run_ensemble_observable_expectation_value_gradient(
                backend=backend,
                pauli=hamiltonian,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                circuit=entangler_circuit2,
                nmeasurement=nmeasurement,
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
            tomography = run_ensemble_observable_expectation_value_tomography(
                    backend=backend,
                    reference_circuits=reference_circuits,
                    reference_weights=reference_weights,
                    circuit=entangler_circuit2,
                    pauli=hamiltonian,
                    nmeasurement=nmeasurement,
                    param_indices=task)
            entangler_circuit2.set_param_values(
                tomography.optimize(),
                param_indices=task,
                )
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

