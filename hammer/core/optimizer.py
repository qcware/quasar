import numpy as np
from . import options    
from . import collocation

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
        shots,
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
                shots=shots,
                hamiltonian=hamiltonian,
                circuit=entangler_circuit2,
                parameter_group=entangler_circuit_parameter_group,
                reference_circuits=reference_circuits,
                reference_weights=reference_weights,
                )[0]
    
        def sa_gradient_helper(Z):
            Z2 = entangler_circuit_parameter_group.compute_raw(Z)
            entangler_circuit2.set_param_values(Z2)
            return collocation.Collocation.compute_sa_gradient(
                backend=backend,
                shots=shots,
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
        Z, vqe_sa_E, opt_data = scipy.optimize.fmin_l_bfgs_b(
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
        shots,
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
                shots=shots,
                hamiltonian=hamiltonian,
                circuit=entangler_circuit2,
                parameter_group=entangler_circuit_parameter_group,
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
