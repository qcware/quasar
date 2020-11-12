import numpy as np
import time
from .options import Options

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
        observable,
        guess_parameters,
        print_level=1,
        ):

        if print_level:
            print(' > BFGS Optimizer <\n')

        if print_level:
            print('  %-13s = %d' % ('maxiter', self.options['maxiter']))
            print('  %-13s = %.3E' % ('g_convergence', self.options['g_convergence']))
            print('')

        Z = np.array(guess_parameters)

        import scipy.optimize
        Z, sa_E, opt_data = scipy.optimize.fmin_l_bfgs_b(
            x0=Z,
            func=observable.run_observable,
            fprime=observable.run_observable_gradient,
            iprint=print_level > 0,   
            maxiter=self.options['maxiter'],
            pgtol=self.options['g_convergence'],
            factr=1.0E-6, # Prevent energy difference from being used as stopping criteria
            )
        Z = np.array(Z)

        if print_level:
            print(' > End BFGS Optimizer <\n')

        return Z
