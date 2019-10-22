from .run import run_pauli_expectation
from .resolution import build_quasar_circuit
from .circuit import Circuit
import numpy as np
import itertools

class Tomography(object):

    def __init__(self):
        raise NotImplementedError

    @property
    def nparam(self):
        raise NotImplementedError
        

    def compute_observable_expectation_value(
        self,
        params, 
        ):
        raise NotImplementedError

    def compute_observable_expectation_value_gradient(
        self,
        params,
        ):

        raise NotImplementedError

    def compute_observable_expectation_value_hessian(
        self,
        params,
        ):

        raise NotImplementedError

class RotationTomography(Tomography):

    def __init__(
        self,
        coefs,
        ):

        self.coefs = coefs

    @property
    def nparam(self):
        return self.coefs.ndim
        
    def compute_observable_expectation_value(
        self,
        params,
        ):
    
        if params.ndim < 2: raise RuntimeError('params.ndim < 2')
        if params.shape[0] != self.nparam: raise RuntimeError('params.shape[0] != self.nparam')
    
        # NOTE: Special case (bare coefficent)
        if self.coefs.ndim == 0: return self.coefs
    
        bs = []
        for theta in params:
            bs.append([
                np.ones_like(theta),
                np.cos(2.0 * theta),
                np.sin(2.0 * theta),  
                ])
    
        O = np.zeros_like(params[0])
        for Js in itertools.product(range(3), repeat=self.nparam):
            R = np.ones_like(params[0])
            for J2, J in enumerate(Js):
                R *= bs[J2][J]
            O += self.coefs[Js] * R
    
        return O
    
    def compute_observable_expectation_value_gradient(
        self,
        params,
        ):
    
        if params.ndim < 2: raise RuntimeError('params.ndim < 2')
        if params.shape[0] != self.nparam: raise RuntimeError('params.shape[0] != self.nparam')
    
        G = np.zeros((self.nparam,) + params[0].shape)
        for k in range(self.nparam):
            notk = [_ for _ in range(self.nparam) if _ != k] 
            paramsk = params[k]
            params2 = params[notk]
            coefs_b = np.take(self.coefs, 1, k)
            coefs_c = np.take(self.coefs, 2, k)
            tomography_b = RotationTomography(coefs_b)
            tomography_c = RotationTomography(coefs_c)
            Ob = tomography_b.compute_observable_expectation_value(params2)
            Oc = tomography_c.compute_observable_expectation_value(params2)
            G[k] = -2.0 * np.sin(2.0 * paramsk) * Ob + 2.0 * np.cos(2.0 * paramsk) * Oc
        return G
    
    def compute_observable_expectation_value_hessian(
        self,
        params,
        ):
    
        if params.ndim < 2: raise RuntimeError('params.ndim < 2')
        if params.shape[0] != self.nparam: raise RuntimeError('params.shape[0] != self.nparam')
    
        H = np.zeros((self.nparam,)*2 + params[0].shape)
        for k in range(self.nparam):
            notk = [_ for _ in range(self.nparam) if _ != k] 
            paramsk = params[k]
            params2 = params[notk]
            coefs_b = np.take(self.coefs, 1, k)
            coefs_c = np.take(self.coefs, 2, k)
            tomography_b = RotationTomography(coefs_b)
            tomography_c = RotationTomography(coefs_c)
            Ob = tomography_b.compute_observable_expectation_value(params2)
            Oc = tomography_c.compute_observable_expectation_value(params2)
            if len(notk):
                Gb = tomography_b.compute_observable_expectation_value_gradient(params2)
                Gc = tomography_c.compute_observable_expectation_value_gradient(params2)
                H[k,notk] = -2.0 * np.sin(2.0 * paramsk) * Gb + 2.0 * np.cos(2.0 * paramsk) * Gc
            H[k, k] = - 4.0 * np.cos(2.0 * paramsk) * Ob - 4.0 * np.sin(2.0 * paramsk) * Oc
        return H 

    # > Tomography quadrature utility < #

    @staticmethod
    def quad_x(D=1):
    
        return np.array(np.meshgrid(
            *[[-np.pi / 3.0, 0.0, +np.pi / 3.0]]*D,
            indexing='ij',
            ))
    
    @staticmethod
    def quad_transfer(D=1):
    
        T1 = np.array([
            [1.0, -0.5, -np.sqrt(3.0)/2.0,],
            [1.0,  1.0,               0.0,],
            [1.0, -0.5, +np.sqrt(3.0)/2.0,],
            ], dtype=np.float)
    
        T = np.copy(T1)
        for D2 in range(1,D):
            T = np.kron(T, T1)
        T *= 0.5**D
    
        return T
    
    @staticmethod
    def quad_transfer_inv(D=1):
    
        T1inv = np.array([
            [         1.0,  1.0,           1.0,],
            [        -1.0,  2.0,          -1.0,],
            [-np.sqrt(3.0), 0.0, +np.sqrt(3.0),],
            ], dtype=np.float) / 3.0
    
        Tinv = np.copy(T1inv)
        for D2 in range(1,D):
            Tinv = np.kron(Tinv, T1inv)
    
        return Tinv
    
    @staticmethod
    def quad_coefs(O):
    
        return np.reshape(np.dot(RotationTomography.quad_transfer_inv(D=O.ndim), O.ravel()), O.shape)

    # => Optimization <= #
    
    def optimize_jacobi_1(
        self,
        theta0=None,
        n=100,
        d=0,
        ):
    
        if theta0 is None:
            theta0 = np.zeros((self.nparam,))
    
        theta = np.copy(theta0)
        thetas = [theta0]
        for iteration in range(n):
            k = (iteration + d) % self.nparam
            theta_2 = np.array([theta for k2, theta in enumerate(theta) if k2 != k])
            theta_2 = np.reshape(theta_2, theta_2.shape + (1,))
            coefs_b = np.take(self.coefs, 1, k)
            coefs_c = np.take(self.coefs, 2, k)
            tomography_b = RotationTomography(coefs_b)
            tomography_c = RotationTomography(coefs_c)
            Ob = tomography_b.compute_observable_expectation_value(theta_2)
            Oc = tomography_c.compute_observable_expectation_value(theta_2)
            theta[k] = 0.5 * np.arctan2(-Oc, -Ob)
            thetas.append(np.copy(theta))
        thetas = np.array(thetas)
        return thetas.T
    
    def optimize_jacobi_1_best(
        self,
        theta0=None,
        n=100,
        ):
    
        thetas = []
        for d in range(self.nparam):
            thetas.append(self.optimize_jacobi_1(
                theta0=theta0,  
                n=n,    
                d=d,
                ))
        Os = np.array([self.compute_observable_expectation_value(theta2[:,-2:-1]) for theta2 in thetas])
        return thetas[np.argmin(Os)] 

    def optimize(self):
        return self.optimize_jacobi_1_best()[:,-1]

def run_observable_expectation_value_tomography(
    backend,
    circuit,
    pauli,
    nmeasurement=None,
    param_indices=None,
    **kwargs):
    
    # No dropthrough - always need quasar.Circuit to manipulate
    circuit = build_quasar_circuit(circuit).copy()
    param_values = circuit.param_values

    # Default to doing tomography over all parameters (NOTE: This costs 3**nparam pauli expectation values)
    if param_indices is None:
        param_indices = tuple(range(circuit.nparam))

    # Check that the tomography formula is known for these parameters (i.e., Rx, Ry, Rz gates)
    param_keys = circuit.param_keys
    for param_index in param_indices:
        key = param_keys[param_index]
        time, qubits, name = key
        gate = circuit.gates[(time, qubits)]
        if not gate.name in ('Rx', 'Ry', 'Rz'): 
            raise RuntimeError('Unknown tomography rule: presently can only tomography Rx, Ry, Rz gates: %s' % gate)

    # The tomography quadrature grid
    T = RotationTomography.quad_x(len(param_indices))
    O = np.zeros_like(T[0])
    npoint = T.size // T.shape[0]
    for I in range(npoint):
        param_values2 = param_values.copy()
        for param_index, T2 in zip(param_indices, T):
            param_values2[param_index] = T2.ravel()[I]
        circuit.set_param_values(param_values2)
        O.ravel()[I] = run_pauli_expectation(backend, circuit, pauli, nmeasurement, **kwargs).dot(pauli).real # TODO: do we need this to be real
    
    # Tomography fitting
    coefs = RotationTomography.quad_coefs(O) 

    # Finished RotationTomography object
    return RotationTomography(coefs=coefs) 

def run_ensemble_observable_expectation_value_tomography(
    backend,
    reference_circuits,
    reference_weights,
    circuit,
    pauli,
    nmeasurement=None,
    param_indices=None,
    **kwargs):
    
    # No dropthrough - always need quasar.Circuit to manipulate
    reference_circuits = [build_quasar_circuit(_) for _ in reference_circuits]
    circuit = build_quasar_circuit(circuit).copy()
    param_values = circuit.param_values

    # Default to doing tomography over all parameters (NOTE: This costs 3**nparam pauli expectation values)
    if param_indices is None:
        param_indices = tuple(range(circuit.nparam))

    # Check that the tomography formula is known for these parameters (i.e., Rx, Ry, Rz gates)
    param_keys = circuit.param_keys
    for param_index in param_indices:
        key = param_keys[param_index]
        time, qubits, name = key
        gate = circuit.gates[(time, qubits)]
        if not gate.name in ('Rx', 'Ry', 'Rz'): 
            raise RuntimeError('Unknown tomography rule: presently can only tomography Rx, Ry, Rz gates: %s' % gate)

    # The tomography quadrature grid
    T = RotationTomography.quad_x(len(param_indices))
    O = np.zeros_like(T[0])
    npoint = T.size // T.shape[0]
    for I in range(npoint):
        param_values2 = param_values.copy()
        for param_index, T2 in zip(param_indices, T):
            param_values2[param_index] = T2.ravel()[I]
        circuit.set_param_values(param_values2)
        Oval = 0.0
        for ref, w in zip(reference_circuits, reference_weights):
            circuit2 = Circuit.concatenate([ref, circuit])
            Oval += w * run_pauli_expectation(backend, circuit2, pauli, nmeasurement, **kwargs).dot(pauli).real # TODO: do we need this to be real
        O.ravel()[I] = Oval
    
    # Tomography fitting
    coefs = RotationTomography.quad_coefs(O) 

    # Finished RotationTomography object
    return RotationTomography(coefs=coefs) 



