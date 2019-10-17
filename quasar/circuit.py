import numpy as np
import sortedcontainers # SortedSet, SortedDict
import collections      # OrderedDict
from .algebra import Algebra

""" Quasar: an ultralight python-3.X quantum simulator package

Note on Qubit Order:

We use the standard QIS qubit order of Nielsen and Chuang, where the qubits are
ordered from left to right in the ket, i.e., |0123>. For instance, the circuit:

T   : |0|
         
|0> : -H-
         
|1> : ---
         
|2> : ---
         
|3> : ---

T   : |0|

Produces the state (|0000> + |1000>) / sqrt(2), which appears in the simulated
state vector as:

[0.70710678 0.         0.         0.         0.         0.
 0.         0.         0.70710678 0.         0.         0.
 0.         0.         0.         0.        ]

E.g., the 0-th (|0000>) and 8-th (|1000>) coefficient are set.

This ordering is used in many places in QIS, e.g., Cirq, but the opposite
ordering is also sometimes seen, e.g., in Qiskit.
"""
# ==> Matrix class <== #

class Matrix(object):

    """ Class Matrix holds several common matrices encountered in quantum circuits.

    These matrices are stored in np.ndarray with dtype=np.complex128.

    The naming/ordering of the matrices in Quasar follows that of Nielsen and
    Chuang, *except* that rotation matrices are specfied in full turns:

        Rz(theta) = exp(-i*theta*Z)
    
    whereas Nielsen and Chuang define these in half turns:

        Rz^NC(theta) = exp(-i*theta*Z/2)
    """

    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    """ The 1-qubit I (identity) matrix """

    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    """ The 1-qubit X (NOT) matrix """

    Y = np.array([[0.0, -1.0j], [+1.0j, 0.0]], dtype=np.complex128)
    """ The 1-qubit Y matrix """

    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    """ The 1-qubit Z matrix """

    S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    """ The 1-qubit S (Phase) matrix """

    ST = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
    """ The 1-qubit S^+ (Phase dagger) matrix """

    T = np.array([[1.0, 0.0], [0.0, np.exp(np.pi/4.0*1.j)]], dtype=np.complex128)
    """ The 1-qubit T (sqrt-S) matrix """

    TT = np.array([[1.0, 0.0], [0.0, np.exp(-np.pi/4.0*1.j)]], dtype=np.complex128)
    """ The 1-qubit T (sqrt-S-dagger) matrix """

    H = 1.0 / np.sqrt(2.0) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    """ The 1-qubit H (Hadamard) matrix """

    # exp(+i (pi/4) * X) : Z -> Y basis transformation
    Rx2 = 1.0 / np.sqrt(2.0) * np.array([[1.0, +1.0j], [+1.0j, 1.0]], dtype=np.complex128)
    """ The 1-qubit Z -> Y basis transformation matrix (a specific Rx matrix) """

    Rx2T = 1.0 / np.sqrt(2.0) * np.array([[1.0, -1.0j], [-1.0j, 1.0]], dtype=np.complex128)
    """ The 1-qubit Y -> Z basis transformation matrix (a specific Rx matrix) """

    II = np.kron(I, I)
    """ The 2-qubit I \otimes I matrix """
    
    IX = np.kron(I, X)
    """ The 2-qubit I \otimes X matrix """

    IY = np.kron(I, Y)
    """ The 2-qubit I \otimes Y matrix """

    IZ = np.kron(I, Z)
    """ The 2-qubit I \otimes Z matrix """

    XI = np.kron(X, I)
    """ The 2-qubit X \otimes I matrix """

    XX = np.kron(X, X)
    """ The 2-qubit X \otimes X matrix """

    XY = np.kron(X, Y)
    """ The 2-qubit X \otimes Y matrix """

    XZ = np.kron(X, Z)
    """ The 2-qubit X \otimes Z matrix """

    YI = np.kron(Y, I)
    """ The 2-qubit Y \otimes I matrix """

    YX = np.kron(Y, X)
    """ The 2-qubit Y \otimes X matrix """

    YY = np.kron(Y, Y)
    """ The 2-qubit Y \otimes Y matrix """

    YZ = np.kron(Y, Z)
    """ The 2-qubit Y \otimes Z matrix """

    ZI = np.kron(Z, I)
    """ The 2-qubit Z \otimes I matrix """

    ZX = np.kron(Z, X)
    """ The 2-qubit Z \otimes X matrix """

    ZY = np.kron(Z, Y)
    """ The 2-qubit Z \otimes Y matrix """

    ZZ = np.kron(Z, Z)
    """ The 2-qubit Z \otimes Z matrix """

    CX = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.complex128)
    """ The 2-qubit CX (controlled-X) matrix """

    CY = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0j],
        [0.0, 0.0, +1.0j, 0.0],
        ], dtype=np.complex128)
    """ The 2-qubit CY (controlled-Y) matrix """

    CZ = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        ], dtype=np.complex128)
    """ The 2-qubit CZ (controlled-Z) matrix """

    CS = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0j],
        ], dtype=np.complex128)
    """ The 2-qubit CS (controlled-S) matrix """

    CST = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0j],
        ], dtype=np.complex128)
    """ The 2-qubit CS^+ (controlled-S-dagger) matrix """

    SWAP = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.complex128)
    """ The 2-qubit SWAP matrix """

    # Toffoli
    CCX = np.eye(8, dtype=np.complex128)
    """ The 3-qubit CCX (Toffoli) matrix """
    CCX[6,6] = 0.0
    CCX[7,7] = 0.0
    CCX[6,7] = 1.0
    CCX[7,6] = 1.0

    # Fredkin
    CSWAP = np.eye(8, dtype=np.complex128)
    """ The 3-qubit CSWAP (Fredkin) matrix """
    CSWAP[5,5] = 0.0
    CSWAP[6,6] = 0.0
    CSWAP[5,6] = 1.0
    CSWAP[6,5] = 1.0

    @staticmethod
    def Rx(theta=0.0):
        """ The 1-qubit Rx (rotation about X) matrix

        Defined as,

            U = exp(-i*theta*X)

        Params:
            theta (float) - rotation angle.
        Returns:
            (np.ndarray) - Rx matrix for the specified value of theta.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -1.j*s], [-1.j*s, c]], dtype=np.complex128)

    @staticmethod
    def Ry(theta=0.0):
        """ The 1-qubit Ry (rotation about Y) matrix

        Defined as,

            U = exp(-i*theta*Y)

        Params:
            theta (float) - rotation angle.
        Returns:
            (np.ndarray) - Ry matrix for the specified value of theta.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [+s, c]], dtype=np.complex128)

    @staticmethod
    def Rz(theta=0.0):
        """ The 1-qubit Rz (rotation about Z) matrix

        Defined as,

            U = exp(-i*theta*Z)

        Params:
            theta (float) - rotation angle.
        Returns:
            (np.ndarray) - Rz matrix for the specified value of theta.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c-1.j*s, 0.0], [0.0, c+1.j*s]], dtype=np.complex128)

    @staticmethod
    def u1(lam=0.0):
        return np.array([[1.0, 0.0], [0.0, np.exp(+1.j*lam)]], dtype=np.complex128)

    @staticmethod
    def u2(phi=0.0, lam=0.0):
        return np.array([[1.0, -np.exp(+1.j*lam)], [+np.exp(+1.j*phi), np.exp(+1.j*(phi+lam))]], dtype=np.complex128) / np.sqrt(2.0)

    @staticmethod
    def u3(theta=0.0, phi=0.0, lam=0.0):
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        return np.array([
            [c, -np.exp(+1.j*lam)*s],
            [+np.exp(+1.j*phi)*s, np.exp(+1.j*(phi+lam))*c],
            ], dtype=np.complex128)
    
    @staticmethod
    def R_ion(theta=0.0, phi=0.0):
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        fm = np.exp(-1.j * phi)
        fp = np.exp(+1.j * phi)
        return np.array([
            [c, -1.j * fm * s],
            [-1.j * fp * s, c],
            ], dtype=np.complex128)
    
    @staticmethod
    def Rz_ion(theta=0.0):
        fm = np.exp(-0.5j * theta)
        fp = np.exp(+0.5j * theta)
        return np.array([
            [fm, 0.0],
            [0.0, fp],
            ], dtype=np.complex128)

    @staticmethod
    def XX_ion(chi=0.0):
        c = np.cos(chi)
        s = np.sin(chi)
        return np.array([
            [c, 0.0, 0.0, -1.j * s],
            [0.0, c, -1.j * s, 0.0],
            [0.0, -1.j * s, c, 0.0],
            [-1.j * s, 0.0, 0.0, c],
            ], dtype=np.complex128)
    
# ==> Gate Class <== #

class Gate(object):

    def __init__(
        self,
        nqubit,
        operator_function,
        parameters,
        name,
        ascii_symbols,
        involuntary=False,
        dagger_function=None,
        ):
        
        self.nqubit = nqubit 
        self.operator_function = operator_function
        self.parameters = parameters
        self.name = name
        self.ascii_symbols = ascii_symbols
        self.involuntary = involuntary
        self.dagger_function = dagger_function

        # Validity checks
        if not isinstance(self.nqubit, int): raise RuntimeError('nqubit must be int')
        if self.nqubit <= 0: raise RuntimeError('nqubit <= 0') 
        if self.operator.shape != (2**self.nqubit,)*2: raise RuntimeError('U must be shape (2**nqubit,)*2')
        if not isinstance(self.parameters, collections.OrderedDict): raise RuntimeError('parameters must be collections.OrderedDict')
        if not all(isinstance(_, str) for _ in list(self.parameters.keys())): raise RuntimeError('parameters keys must all be str')
        if not all(isinstance(_, float) for _ in list(self.parameters.values())): raise RuntimeError('parameters values must all be float')
        if not isinstance(self.name, str): raise RuntimeError('name must be str')
        if not isinstance(self.ascii_symbols, list): raise RuntimeError('ascii_symbols must be list')
        if len(self.ascii_symbols) != self.nqubit: raise RuntimeError('len(ascii_symbols) != nqubit')
        if not all(isinstance(_, str) for _ in self.ascii_symbols): raise RuntimeError('ascii_symbols must all be str')
        if not isinstance(self.involuntary, bool): raise RuntimeError('involuntary must be bool')
        
    @property
    def ntime(self):
        return 1

    @property
    def is_composite(self):
        return False

    @property
    def is_controlled(self):
        return False

    def __str__(self):
        """ String representation of this Gate (self.name) """
        return self.name

    @property
    def operator(self): 
        """ The (2**N,)*2 operator (unitary) matrix underlying this Gate. 

        The action of the gate on a given state is given graphically as,

        |\Psi> -G- |\Psi'>

        and mathematically as,

        |\Psi_I'> = \sum_J U_IJ |\Psi_J>

        Returns:
            (np.ndarray of shape (2**N,)*2) - the operator (unitary) matrix
                underlying this gate, built from the current parameter state.
        """
        return self.operator_function(self.parameters)

    # > Equivalence < #

    def test_operator_equivalence(
        gate1,
        gate2,
        operator_tolerance=1.0E-12,
        ):
    
        """ Test if the operator matrices of two gates are numerically
            equivalent to within a maximum absolute deviation of
            operator_tolerance.

            Note that the gates might still have different recipes, but produce
            the same operator. Therefore, this definition should be considered
            to be an intermediate level of equivalence.

        Params:
            gate1 (Gate) - first gate to compare
            gate2 (Gate) - second gate to compare
            operator_tolerance (float) - maximum absolute deviation threshold
                for declaring Gate operator matrices to be identical.
        Returns:
            (bool) - True if the gates are equivalent under the definition
                above, else False.
        """

        return np.max(np.abs(gate1.operator - gate2.operator)) < operator_tolerance

    # > Copying < #
    
    def copy(self):
        """ Make a deep copy of the current Gate. 
        
        Returns:
            (Gate) - a copy of this Gate whose parameters may be modified
                without modifying the parameters of self.
        """
        return Gate(
            nqubit=self.nqubit, 
            operator_function=self.operator_function, 
            parameters=self.parameters.copy(), 
            name=self.name,  
            ascii_symbols=self.ascii_symbols.copy(),
            involuntary=self.involuntary,
            dagger_function=self.dagger_function,
            )

    # > Adjoint < #

    def dagger(self):
        if self.involuntary:
            return self.copy()
        elif self.dagger_function:
            return self.dagger_function(self.parameters)
        else:
            return Gate(
                nqubit=self.nqubit,
                operator_function=lambda parameters : self.operator_function(parameters).T.conj(),
                parameters=self.parameters.copy(), 
                name=self.name+'^+',
                ascii_symbols=[symbol + ('' if symbol in ['@', 'O'] else '^+') for symbol in self.ascii_symbols],
                involuntary=self.involuntary,
                dagger_function=self.dagger_function,
                )

    # > Explosion Utility < #

    def exploded_gates(self):
        return { (0, tuple(range(self.nqubit))) : self }

    # > Parameter Access < #

    @property
    def nparameter(self):
        """ Total number of parameters in this Gate """
        return len(self.parameters)

    def set_parameter(self, key, value):
        """ Set the value of a parameter of this Gate. 

        Params:
            key (str) - the key of the parameter
            value (float) - the value of the parameter
        Result:
            self.parameters[key] = value. If the Gate does not have a parameter
                corresponding to key, a RuntimeError is thrown.
        """
        if key not in self.parameters: raise RuntimeError('Key %s is not in parameters' % key)
        self.parameters[key] = value

    def set_parameters(self, parameters):
        """ Set the values of multiple parameters of this Gate.

        Params:
            parameters (dict of str : float) -  dict of parameter values
        Result:
            self.parameters is updated with the contents of parameters by
                calling self.set_parameter for each key/value pair.
        """
        for key, value in parameters.items():
            self.set_parameter(key=value, parameter=value)

    def apply_to_statevector(
        self,
        statevector1,
        statevector2,
        qubits,
        dtype=np.complex128,
        ):

        """ Apply this gate to statevector1, acting on qubit indices in qubits,
            and return the result, along with a scratch statevector. Ideally,
            no statevector allocations will be performed in the course of this
            operation - a scratch statevector is provided as input to help with
            this.

        Params:
            statevector1 (np.ndarray of shape 2**K) - input statevector
            statevector2 (np.ndarray of shape 2**K) - scratch statevector
            qubits (iterable of int of size self.nqubit) - qubit indices to
                apply this gate to. 
            dtype (real or complex dtype) - the dtype to perform the
                computation at. The gate operator will be cast to this dtype.
                Note that using real dtypes (float64 or float32) can reduce
                storage and runtime, but the imaginary parts of the input wfn
                and all gate unitary operators will be discarded without
                checking. In these cases, the user is responsible for ensuring
                that the circuit works on O(2^N) rather than U(2^N) and that
                the output is valid.
        Result:
            Either or both of statevector1 and statevector2 may be modified.
            One of them is modified to contain the resultant statevector, and
            then this output statevector and the new scratch statevector are
            returned.
        Returns:
            output, scratch (np.ndarray of shape 2**K) - output statevector,
                then scratch statevector.
        """

        if self.nqubit != len(qubits): raise RuntimeError('self.nqubit != len(qubits)')

        operator = np.array(self.operator, dtype=dtype)

        if self.nqubit == 1:
            return Algebra.apply_operator_1(
                statevector1=statevector1,
                statevector2=statevector2,
                operator=operator,
                A=qubits[0],
                )
        elif self.nqubit == 2:
            return Algebra.apply_operator_2(
                statevector1=statevector1,
                statevector2=statevector2,
                operator=operator,
                A=qubits[0],
                B=qubits[1],
                )
        elif self.nqubit == 3:
            return Algebra.apply_operator_3(
                statevector1=statevector1,
                statevector2=statevector2,
                operator=operator,
                A=qubits[0],
                B=qubits[1],
                C=qubits[2],
                )
        else:
            return Algebra.apply_operator_n(
                statevector1=statevector1,
                statevector2=statevector2,
                operator=operator,
                qubits=qubits,
                )

# > Explicit 1-body gates < #

Gate.I = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.I,
    parameters=collections.OrderedDict(),
    name='I',
    ascii_symbols=['I'],
    involuntary=True,
    )
""" I (identity) gate """

Gate.X = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.X,
    parameters=collections.OrderedDict(),
    name='X',
    ascii_symbols=['X'],
    involuntary=True,
    )
""" X (NOT) gate """

Gate.Y = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.Y,
    parameters=collections.OrderedDict(),
    name='Y',
    ascii_symbols=['Y'],
    involuntary=True,
    )
""" Y gate """

Gate.Z = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.Z,
    parameters=collections.OrderedDict(),
    name='Z',
    ascii_symbols=['Z'],
    involuntary=True,
    )
""" Z gate """

Gate.H = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.H,
    parameters=collections.OrderedDict(),
    name='H',
    ascii_symbols=['H'],
    involuntary=True,
    )
""" H (Hadamard) gate """

Gate.S = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.S,
    parameters=collections.OrderedDict(),
    name='S',
    ascii_symbols=['S'],
    dagger_function = lambda parameters : Gate.ST,
    )
""" S gate """

Gate.ST = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.ST,
    parameters=collections.OrderedDict(),
    name='S^+',
    ascii_symbols=['S^+'],
    dagger_function = lambda parameters : Gate.S,
    )
""" S^+ gate """

Gate.T = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.T,
    name='T',
    parameters=collections.OrderedDict(),
    ascii_symbols=['T'],
    dagger_function = lambda parameters : Gate.TT,
    )
""" T gate """

Gate.TT = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.TT,
    name='T^+',
    parameters=collections.OrderedDict(),
    ascii_symbols=['T^+'],
    dagger_function = lambda parameters : Gate.T,
    )
""" T^+ gate """

Gate.Rx2 = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.Rx2,
    parameters=collections.OrderedDict(),
    name='Rx2',
    ascii_symbols=['Rx2'],
    dagger_function = lambda parameters : Gate.Rx2T,
    )
""" Rx2 gate """

Gate.Rx2T = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.Rx2T,
    parameters=collections.OrderedDict(),
    name='Rx2T',
    ascii_symbols=['Rx2T'],
    dagger_function = lambda parameters : Gate.Rx2,
    )
""" Rx2T gate """

# > Explicit 2-body gates < #

Gate.CX = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.CX,
    parameters=collections.OrderedDict(),
    name='CX',
    ascii_symbols=['@', 'X'],
    involuntary=True,
    )
""" CX (CNOT) gate """
Gate.CY = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.CY,
    parameters=collections.OrderedDict(),
    name='CY',
    ascii_symbols=['@', 'Y'],
    involuntary=True,
    )
""" CY gate """
Gate.CZ = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.CZ,
    parameters=collections.OrderedDict(),
    name='CZ',
    ascii_symbols=['@', 'Z'],
    involuntary=True,
    )
""" CZ gate """
Gate.CS = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.CS,
    parameters=collections.OrderedDict(),
    name='CS',
    ascii_symbols=['@', 'S'],
    dagger_function = lambda parameters : Gate.CST,
    )
""" CS gate """
Gate.CST = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.CST,
    parameters=collections.OrderedDict(),
    name='CS^+',
    ascii_symbols=['@', 'S^+'],
    dagger_function = lambda parameters : Gate.CS,
    )
""" CS^+ gate """
Gate.SWAP = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.SWAP,
    parameters=collections.OrderedDict(),
    name='SWAP',
    ascii_symbols=['X', 'X'],
    involuntary=True,
    )
""" SWAP gate """

# > Explicit 3-body gates < #

Gate.CCX = Gate(
    nqubit=3,
    operator_function = lambda parameters: Matrix.CCX,
    parameters=collections.OrderedDict(),
    name='CCX',
    ascii_symbols=['@', '@', 'X'],
    involuntary=True,
    )
""" CCX (Toffoli gate) """
Gate.CSWAP = Gate(
    nqubit=3,
    operator_function = lambda parameters: Matrix.CSWAP,
    parameters=collections.OrderedDict(),
    name='CSWAP',
    ascii_symbols=['@', 'X', 'X'],
    involuntary=True,
    )
""" CSWAP (Toffoli gate) """

# > Parametrized 1-body gates < #

@staticmethod
def _GateRx(theta=0.0):

    """ Rx (theta) = exp(-i * theta * x) """
    
    def operator_function(parameters):
        theta = parameters['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -1.j*s], [-1.j*s, c]], dtype=np.complex128)
    
    return Gate(
        nqubit=1,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('theta', theta)]),
        name='Rx',
        ascii_symbols=['Rx'],
        dagger_function=lambda parameters : Gate.Rx(**{ k : -v for k, v in parameters.items()})
        )
    
@staticmethod
def _GateRy(theta=0.0):

    """ Ry (theta) = exp(-i * theta * Y) """
    
    def operator_function(parameters):
        theta = parameters['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [+s, c]], dtype=np.complex128)

    return Gate(
        nqubit=1,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('theta', theta)]),
        name='Ry',
        ascii_symbols=['Ry'],
        dagger_function=lambda parameters : Gate.Ry(**{ k : -v for k, v in parameters.items()})
        )
    
@staticmethod
def _GateRz(theta=0.0):

    """ Rz (theta) = exp(-i * theta * Z) """
    
    def operator_function(parameters):
        theta = parameters['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c-1.j*s, 0.0], [0.0, c+1.j*s]], dtype=np.complex128)

    return Gate(
        nqubit=1,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('theta', theta)]),
        name='Rz',
        ascii_symbols=['Rz'],
        dagger_function=lambda parameters : Gate.Rz(**{ k : -v for k, v in parameters.items()})
        )
    
Gate.Rx = _GateRx
Gate.Ry = _GateRy
Gate.Rz = _GateRz

@staticmethod
def _Gateu1(lam=0.0):

    def operator_function(parameters):
        return Matrix.u1(lam=parameters['lam'])

    return Gate(
        nqubit=1,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('lam', lam)]),
        name='u1',
        ascii_symbols=['u1'],
        dagger_function=lambda parameters : Gate.u1(**{ k : -v for k, v in parameters.items()})
        )

@staticmethod
def _Gateu2(phi=0.0, lam=0.0):

    def operator_function(parameters):
        return Matrix.u2(phi=parameters['phi'], lam=parameters['lam'])

    return Gate(
        nqubit=1,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('phi', phi), ('lam', lam)]),
        name='u2',
        ascii_symbols=['u2'],
        dagger_function=lambda parameters : Gate.u2(**{ k : -v for k, v in parameters.items()})
        )

@staticmethod
def _Gateu3(theta=0.0, phi=0.0, lam=0.0):

    def operator_function(parameters):
        return Matrix.u3(theta=parameters['theta'], phi=parameters['phi'], lam=parameters['lam'])

    return Gate(
        nqubit=1,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('theta', theta), ('phi', phi), ('lam', lam)]),
        name='u3',
        ascii_symbols=['u3'],
        dagger_function=lambda parameters : Gate.u3(**{ k : -v for k, v in parameters.items()})
        )

Gate.u1 = _Gateu1
Gate.u2 = _Gateu2
Gate.u3 = _Gateu3

# > Parametrized 2-body gates < #

@staticmethod
def _GateSO4(A=0.0, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0):
    
    def operator_function(parameters):
        A = parameters['A']
        B = parameters['B']
        C = parameters['C']
        D = parameters['D']
        E = parameters['E']
        F = parameters['F']
        X = np.array([
            [0.0, +A,  +B,  +C],
            [-A, 0.0,  +D,  +E],
            [-B,  -D, 0.0,  +F],
            [-C,  -E,  -F, 0.0],
            ])
        import scipy.linalg
        U = scipy.linalg.expm(X)
        return np.array(U, dtype=np.complex128)

    return Gate(
        nqubit=2,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('A', A), ('B', B), ('C', C), ('D', D), ('E', E), ('F', F)]),
        name='SO4',
        ascii_symbols=['SO4A', 'SO4B'],
        dagger_function=lambda parameters : Gate.SO4(**{ k : -v for k, v in parameters.items()})
        )

Gate.SO4 = _GateSO4

@staticmethod
def _GateSO42(thetaIY=0.0, thetaYI=0.0, thetaXY=0.0, thetaYX=0.0, thetaZY=0.0, thetaYZ=0.0):
    
    def operator_function(parameters):
        A = -(parameters['thetaIY'] + parameters['thetaZY'])
        F = -(parameters['thetaIY'] - parameters['thetaZY'])
        C = -(parameters['thetaYX'] + parameters['thetaXY'])
        D = -(parameters['thetaYX'] - parameters['thetaXY'])
        B = -(parameters['thetaYI'] + parameters['thetaYZ'])
        E = -(parameters['thetaYI'] - parameters['thetaYZ'])
        X = np.array([
            [0.0, +A,  +B,  +C],
            [-A, 0.0,  +D,  +E],
            [-B,  -D, 0.0,  +F],
            [-C,  -E,  -F, 0.0],
            ])
        import scipy.linalg
        U = scipy.linalg.expm(X)
        return np.array(U, dtype=np.complex128)

    return Gate(
        nqubit=2,
        operator_function=operator_function,
        parameters=collections.OrderedDict([
            ('thetaIY' , thetaIY),
            ('thetaYI' , thetaYI),
            ('thetaXY' , thetaXY),
            ('thetaYX' , thetaYX),
            ('thetaZY' , thetaZY),
            ('thetaYZ' , thetaYZ),
        ]),
        name='SO42',
        ascii_symbols=['SO42A', 'SO42B'],
        dagger_function=lambda parameters : Gate.SO42(**{ k : -v for k, v in parameters.items()})
        )

Gate.SO42 = _GateSO42

@staticmethod
def _CF(theta=0.0):

    """ Controlled F gate """
    
    def operator_function(parameters):
        theta = parameters['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0,  +c,  +s],
            [0.0, 0.0,  +s,  -c],
            ], dtype=np.complex128)
    
    return Gate(
        nqubit=2,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('theta', theta)]),
        name='CF',
        ascii_symbols=['@', 'F'],
        dagger_function=lambda parameters : Gate.CF(**{ k : -v for k, v in parameters.items()})
        )

Gate.CF = _CF

# > Ion trap gates < #

def _GateR_ion(theta=0.0, phi=0.0):

    def operator_function(parameters):
        return Matrix.R_ion(theta=parameters['theta'], phi=parameters['phi'])
    
    return Gate(
        nqubit=1,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('theta', theta), ('phi', phi)]),
        name='R_ion',
        ascii_symbols=['R'],
        dagger_function=lambda parameters : Gate.R_ion(**{ k : -v for k, v in parameters.items()})
        )

def _GateRx_ion(theta=0.0):

    def operator_function(parameters):
        return Matrix.R_ion(theta=parameters['theta'], phi=0.0)
    
    return Gate(
        nqubit=1,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('theta', theta)]),
        name='Rx_ion',
        ascii_symbols=['Rx'],
        dagger_function=lambda parameters : Gate.Rx_ion(**{ k : -v for k, v in parameters.items()})
        )

def _GateRy_ion(theta=0.0):

    def operator_function(parameters):
        return Matrix.R_ion(theta=parameters['theta'], phi=np.pi/2.0)
    
    return Gate(
        nqubit=1,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('theta', theta)]),
        name='Ry_ion',
        ascii_symbols=['Ry'],
        dagger_function=lambda parameters : Gate.Ry_ion(**{ k : -v for k, v in parameters.items()})
        )

def _GateRz_ion(theta=0.0):

    def operator_function(parameters):
        return Matrix.Rz_ion(theta=parameters['theta'])
    
    return Gate(
        nqubit=1,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('theta', theta)]),
        name='Rz_ion',
        ascii_symbols=['Rz'],
        dagger_function=lambda parameters : Gate.Rz_ion(**{ k : -v for k, v in parameters.items()})
        )

def _GateXX_ion(chi=0.0):

    def operator_function(parameters):
        return Matrix.XX_ion(chi=parameters['chi'])
    
    return Gate(
        nqubit=2,
        operator_function=operator_function,
        parameters=collections.OrderedDict([('chi', chi)]),
        name='XX_ion',
        ascii_symbols=['XX', 'XX'],
        dagger_function=lambda parameters : Gate.XX_ion(**{ k : -v for k, v in parameters.items()})
        )

Gate.R_ion = _GateR_ion
Gate.Rx_ion = _GateRx_ion
Gate.Ry_ion = _GateRy_ion
Gate.Rz_ion = _GateRz_ion
Gate.XX_ion = _GateXX_ion

# > Special explicit gates < #

@staticmethod
def _GateU1(U):

    """ An explicit 1-body gate that is specified by the user. """

    return Gate(
        nqubit=1,
        operator_function = lambda parameters : U,
        parameters=collections.OrderedDict(),
        name='U1',
        ascii_symbols=['U1'],
        )

@staticmethod
def _GateU2(U):

    """ An explicit 2-body gate that is specified by the user. """

    return Gate(
        nqubit=2,
        operator_function = lambda parameters : U,
        parameters=collections.OrderedDict(),
        name='U2',
        ascii_symbols=['U2A', 'U2B'],
        )

Gate.U1 = _GateU1
Gate.U2 = _GateU2

class CompositeGate(Gate):

    def __init__(
        self,
        circuit,
        name=None,
        ascii_symbols=None,
        ):

        self.circuit = circuit
        self.name = 'CG' if name is None else name
        self.ascii_symbols = ['CG'] * self.circuit.nqubit if ascii_symbols is None else ascii_symbols

    @property
    def ntime(self):
        return self.circuit.ntime

    @property
    def nqubit(self):
        return self.circuit.nqubit 

    @property
    def is_composite(self):
        return True

    @property
    def is_controlled(self):
        return self.circuit.is_controlled

    @property
    def operator_function(self):
        def Ufun(parameters):
            U = np.zeros((2**self.nqubit,)*2, dtype=np.complex128)
            statevector1 = np.zeros((2**self.nqubit), dtype=np.complex128)
            statevector2 = np.zeros((2**self.nqubit), dtype=np.complex128)
            qubits = list(range(self.nqubit))
            for i in range(2**self.nqubit):
                statevector1[...] = 0.0
                statevector1[i] = 1.0
                statevector1, statevector2 = self.apply_to_statevector(
                    statevector1=statevector1,
                    statevector2=statevector2,
                    qubits=qubits,
                    dtype=np.complex128,
                    )
                U[:, i] = statevector1
            return U
        return Ufun

    def apply_to_statevector(
        self,
        statevector1,
        statevector2,
        qubits,
        dtype=np.complex128,
        ):

        return self.circuit.apply_to_statevector(
            statevector1=statevector1,
            statevector2=statevector2,
            qubits=qubits,  
            dtype=dtype,
            )

    @property
    def parameters(self):
        return self.circuit.parameters

    def set_parameter(self, key, value):
        self.circuit.set_parameter(key, value)

    def set_parameters(self, parameters):
        self.circuit.set_parameters(parameters)

    def copy(self):
        return CompositeGate(
            circuit=self.circuit.copy(), 
            name=self.name,
            ascii_symbols=self.ascii_symbols.copy(),
            )

    def dagger(self):
        return CompositeGate(
            circuit=self.circuit.dagger(),
            name=self.name+'^+',
            ascii_symbols=[symbol + ('' if symbol in ['@', 'O'] else '^+') for symbol in self.ascii_symbols],
            )

    def exploded_gates(self):
        gates = {}
        for key, gate in self.circuit.explode(copy=False).gates.items():
            times, qubits = key
            time2 = times[0] - self.circuit.min_time
            qubits2 = tuple(_ - self.circuit.min_qubit for _ in qubits)
            gates[(time2, qubits2)] = gate
        return gates

class ControlledGate(Gate):

    def __init__(
        self,
        gate,
        controls=None,
        ):

        if controls is None: controls = [True]

        self.gate = gate
        self.controls = controls

        if not isinstance(self.controls, list): raise RuntimeError('controls must be list')
        if not all(isinstance(_, bool) for _ in self.controls): raise RuntimeError('controls must be list of bool')

    @property
    def ncontrol(self):
        return len(self.controls)

    @property
    def ntime(self):
        return self.gate.ntime

    @property
    def nqubit(self):
        return self.ncontrol + self.gate.nqubit

    @property
    def is_composite(self):
        return self.gate.is_composite

    @property
    def is_controlled(self):
        return True

    @property
    def operator_function(self):
        def cU(params):        
            start = sum([2**(self.nqubit - index - 1) for index, control in enumerate(self.controls) if control] + [0])
            end = start + 2**self.gate.nqubit
            U = np.eye(2**self.nqubit, dtype=np.complex128)
            U[start:end, start:end] = self.gate.operator
            return U
        return cU

    @property
    def name(self):
        return ''.join(['c' if control else 'o' for control in self.controls]) + self.gate.name

    @property
    def ascii_symbols(self):
        return ['@' if control else 'O' for control in self.controls] + self.gate.ascii_symbols

    @property
    def parameters(self):
        return self.gate.parameters

    def set_parameter(self, key, value):
        self.gate.set_parameter(key, value)

    def set_parameters(self, parameters):
        self.gate.set_parameters(parameters)

    def copy(self):
        return ControlledGate(
            gate=self.gate.copy(), 
            controls=self.controls.copy(),
            )

    def dagger(self):
        return ControlledGate(
            gate=self.gate.dagger(),
            controls=self.controls.copy(),
            )

    def exploded_gates(self):
        gates = {}
        for key, gate in self.gate.exploded_gates().items():
            time, qubits = key
            qubits2 = tuple(list(range(self.ncontrol)) + [_ + self.ncontrol for _ in qubits])
            gates[(time, qubits2)] = ControlledGate(gate, self.controls)
        return gates 

class Circuit(object):

    # => Initializer <= #

    def __init__(
        self,
        ):

        self.gates = sortedcontainers.SortedDict()
    
        # Memoization of occupied time/qubit indices
        self.times = sortedcontainers.SortedSet()
        self.qubits = sortedcontainers.SortedSet()
        self.times_and_qubits = sortedcontainers.SortedSet()

        # Memory of last qubits/times key used in add_gate
        self.last_qubits = None
        self.last_times = None

    # => Simple Circuit Attributes <= #

    @property
    def ngate(self):
        """ The total number of gates in the circuit. """
        return len(self.gates)

    @property
    def ngate1(self):
        """ The total number of 1-qubit gates in the circuit. """
        return self.ngate_nqubit(nqubit=1)

    @property
    def ngate2(self):
        """ The total number of 2-qubit gates in the circuit. """
        return self.ngate_nqubit(nqubit=2)

    @property
    def ngate3(self):
        """ The total number of 3-qubit gates in the circuit. """
        return self.ngate_nqubit(nqubit=3)

    @property
    def ngate4(self):
        """ The total number of 4-qubit gates in the circuit. """
        return self.ngate_nqubit(nqubit=4)

    def ngate_nqubit(self, nqubit):
        """ The total number of nqubit-qubit gates in the circuit. 

        Params:
            nqubit (int) - number of qubits to screen on.
        """
        return sum(1 for gate in self.gates.values() if gate.nqubit == nqubit)

    @property
    def max_gate_nqubit(self):
        """ Maximum number of qubits in any gate in the circuit. """
        return max(gate.nqubit for gate in self.gates.values()) if self.ngate else 0
    
    @property
    def max_gate_ntime(self):
        """ Maximum number of times in any gate in the circuit. """
        return max(gate.ntime for gate in self.gates.values()) if self.ngate else 0
    
    @property
    def min_time(self):
        """ The minimum occupied time index (or 0 if no occupied times) """
        return self.times[0] if len(self.times) else 0
    
    @property
    def max_time(self):
        """ The maximum occupied time index (or -1 if no occupied times) """
        return self.times[-1] if len(self.times) else -1

    @property
    def ntime(self):
        """ The total number of time indices in the circuit (including empty time indices). """
        return self.times[-1] - self.times[0] + 1 if len(self.times) else 0

    @property
    def ntime_sparse(self):
        """ The total number of occupied time indices in the circuit (excluding empty time indices). """
        return len(self.times)

    @property
    def min_qubit(self):
        """ The minimum occupied qubit index (or 0 if no occupied qubits) """
        return self.qubits[0] if len(self.qubits) else 0
    
    @property
    def max_qubit(self):
        """ The maximum occupied qubit index (or -1 if no occupied qubits) """
        return self.qubits[-1] if len(self.qubits) else -1

    @property
    def nqubit(self):
        """ The total number of qubit indices in the circuit (including empty qubit indices). """
        return self.qubits[-1] - self.qubits[0] + 1 if len(self.qubits) else 0

    @property
    def nqubit_sparse(self):
        """ The total number of occupied qubit indices in the circuit (excluding empty qubit indices). """
        return len(self.qubits)

    @property
    def is_composite(self):
        """ Does this circuit contain any CompositeGate objects? """
        return any(gate.is_composite for gate in self.gates.values())

    @property
    def is_controlled(self):
        """ Does this circuit contain any ControlledGate objects? """
        return any(gate.is_controlled for gate in self.gates.values())

    # => Circuit Equivalence <= #

    @staticmethod
    def test_equivalence(
        circuit1,
        circuit2,
        operator_tolerance=1.0E-12,
        ):

        """ Test logical circuit equivalence at the level of geographic
            locations of Gate objects and operator equivalence of Gate objects.

            Note that this can be conceptually considered to be an intermediate
            level definition of equivalence. At the lowest level (not this
            case), one could define equivalence to be in terms of the overall
            unitary matrix of the circuits - many different gate layouts and
            definitions would provide equivalence under this definitions. At
            the highest level (not this case), one could define equivalence to
            require identical geographic locations of Gate object, and
            identical Gate objects (e.g., in terms of
            Gate/ControlledGate/CompositeGate class, parameters, names, etc). 
            Here, we define circuit equivalence to the intermediate level of
            identical geographic locations of Gate objects, and numerically
            identical Gate operators, but do not check the specific recipe of
            each Gate's definition.

        Params:
            circuit1 (Circuit) - first circuit to compare
            circuit2 (Circuit) - second circuit to compare
            operator_tolerance (float) - maximum absolute deviation threshold
                for declaring Gate operator matrices to be identical.
        Returns:
            (bool) - True if the circuits are equivalent under the definition
                above, else False.
        """

        # Check that keys are geographically the same 
        if circuit1.gates.keys() != circuit2.gates.keys(): 
            return False

        # Check that the operators of the gates are numerically the same
        for gate1, gate2 in zip(circuit1.gates.values(), circuit2.gates.values()):
            if not Gate.test_operator_equivalence(gate1, gate2, operator_tolerance=1.0E-12): 
                return False

        return True
    
    # => Gate Addition <= #

    def add_gate(
        self,
        gate,
        qubits,
        times=None, 
        time_start=None, 
        time_placement='early',
        copy=True,
        name=None,
        ascii_symbols=None,
        ):

        """ Add a gate to self at specified qubits and times, updating self. The
            qubits to add gate to are always explicitly specified. The times to
            add the gate to may be explicitly specified in the times argumet
            (1st priority), or a recipe for determining the time placement can
            be specified using the time_placement argument (2nd priority).

        Params:
            qate (Gate or Circuit) - the gate to add into self. If gate is a
                Circuit, gate will be cast to a CompositeGate and then added
                into self.
            qubits (int or tuple of int) - ordered qubit indices in self to add the
                qubit indices of circuit into. If a single int is provided (for
                one-qubit gate addition), it is converted to a tuple with a
                single int entry.
            times (int or tuple of int or None) - time moments in self to add
                the gate into. If None, the time_start argument will be
                considered next.
            time_start (int or None) - starting time moment in self to add the
                gate into (often used with ntime > 1 gates). If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
            copy (bool) - copy the gate or not?
            name (str) - name of gate for use in CompositeGate (None indicates
                default name)
            ascii_symbols (list of str or None) - ASCII symbols for use in
                CompositeGate (None indicates default symbols)
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid. The
                `last_qubits` and `last_times` attribute of self is set to the
                qubits and times key of this call to `add_gate`.
        Returns:
            self - for chaining
        """

        # If gate is Circuit, make it a CompositeGate
        gate = CompositeGate(gate, name, ascii_symbols) if isinstance(gate, Circuit) else gate

        # Make qubits a tuple regardless of input
        qubits = (qubits,) if isinstance(qubits, int) else qubits
        # Make times a tuple (or None) regardless of input
        times = (times,) if isinstance(times, int) else times
        
        # Qubit validation
        if len(set(qubits)) != len(qubits):
            raise RuntimeError("Qubit list must not contain repeated indices.")

        # Time determination by rules
        if times is None:
            if time_start is not None:
                times = tuple(range(time_start, time_start + gate.ntime))
            elif time_placement == 'early':
                timemax = self.min_time - 1
                for time, qubit in self.times_and_qubits:
                    if qubit in qubits:
                        timemax = max(timemax, time)
                times = tuple(range(timemax + 1, timemax + 1 + gate.ntime))
            elif time_placement == 'late':
                timemax = self.max_time
                if any((timemax, qubit) in self.times_and_qubits for qubit in qubits):
                    timemax += 1
                times = tuple(range(timemax, timemax + gate.ntime))
            elif time_placement == 'next':
                times = tuple(range(self.max_time + 1, self.max_time + 1 + gate.ntime))
            else:
                raise RuntimeError('Unknown time_placement: %s. Allowed values are early, late, next' % time_placement)

        # Check that qubits makes sense for gate.nqubit
        if len(qubits) != gate.nqubit: raise RuntimeError('%d qubit entries provided for %d-qubit gate' % (len(qubits), gate.nqubit))
        # Check that times makes sense for gate.ntime
        if len(times) != gate.ntime: raise RuntimeError('%d time entries provided for %d-time gate' % (len(times), gate.ntime))
        # Check that the times are sequential and contiguous
        if len(times) > 1 and times != tuple(range(times[0], times[-1]+1)): raise RuntimeError('times are not sequential: %r' % times)
        # Check that the requested circuit locations are open
        for qubit in qubits:
            for time in times:
                if (time, qubit) in self.times_and_qubits:
                    raise RuntimeError('time=%d, qubit=%d circuit location is already occupied' % (time,qubit))

        # Add gate to circuit
        self.gates[(times, qubits)] = gate.copy() if copy else gate
        for qubit in qubits: self.qubits.add(qubit)
        for time in times: self.times.add(time)
        for qubit in qubits:
            for time in times:
                self.times_and_qubits.add((time, qubit))

        # Mark qubits/times key in case user wants to know
        self.last_qubits = tuple(qubits)
        self.last_times = tuple(times)

        return self

    def add_controlled_gate(
        self,
        gate,
        qubits,
        controls=None,
        name=None,
        ascii_symbols=None,
        **kwargs):

        gate = CompositeGate(gate, name, ascii_symbols) if isinstance(gate, Circuit) else gate
        gate = ControlledGate(gate, controls=controls)
        return self.add_gate(gate=gate, qubits=qubits, **kwargs) 

    def add_gates(
        self,
        circuit,
        qubits,
        times=None,
        time_start=None,
        time_placement='early',
        copy=True,
        ):

        """ Add the gates of another circuit to self at specified qubits and
            times, updating self. Essentially a composite version of add_gate.
            The qubits to add circuit to are always explicitly specified. The
            times to add circuit to may be explicitly specified in the times
            argument (1st priority), the starting time moment may be explicitly
            specified and then the circuit added in a time-contiguous manner
            from that point using the time_start argument (2nd priority), or a
            recipe for determining the time-contiguous placement can be
            specified using the time_placement argument (3rd priority).

        Params:
            circuit (Circuit) - the circuit containing the gates to add into
                self. 
            qubits (tuple of int) - ordered qubit indices in self to add the
                qubit indices of circuit into.
            times (tuple of int) - ordered time moments in self to add the time
                moments of circuit into. If None, the time argument will be
                considered next.
            time_start (int) - starting time moment in self to add the time moments
                of circuit into. If None, the time_placement argument will be
                considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine starting time moment in self to add the time moments
                of circuit into. The rules are:
                    'early' - start adding the circuit as early as possible,
                        just after any existing gates on self's qubit wires.
                    'late' - start adding the circuit in the last open time
                        moment in self, unless a conflict arises, in which
                        case, start adding the circuit in the next (new) time
                        moment.
                    'next' - start adding the circuit in the next (new) time
                        moment.
            copy (bool) - copy Gate elements to remove parameter dependencies
                between circuit and updated self (True - default) or not
                (False). 
        Result:
            self is updated with the added gates from circuit. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        # Make qubits a tuple regardless of input
        qubits = (qubits,) if isinstance(qubits, int) else qubits
        # Also make times a tuple if int
        times = (times,) if isinstance(times, int) else times

        # circuit validation
        if circuit.nqubit != len(qubits):
            raise RuntimeError("len(qubits) must be equal to the number of registers in circuit.")
        # circuit validation
        
        if times is None:
            if time_start is not None:
                times = list(range(time_start,time_start+circuit.ntime))
            else:
                if time_placement == 'early':
                    leads = [circuit.ntime] * circuit.nqubit
                    for time2, qubit2 in circuit.times_and_qubits:
                        leads[qubit2 - circuit.min_qubit] = min(leads[qubit2 - circuit.min_qubit], time2 - circuit.min_time)
                    timemax = -1
                    for time2, qubit2 in self.times_and_qubits:
                        if qubit2 in qubits:
                            timemax = max(timemax, time2 - leads[qubits.index(qubit2)])
                    timemax += 1
                    times = list(range(timemax, timemax+circuit.ntime))
                elif time_placement == 'late':
                    timemax = self.max_time
                    if any((timemax, qubit) in self.times_and_qubits for qubit in qubits):
                        timemax += 1 
                    times = list(range(timemax, timemax+circuit.ntime))
                elif time_placement == 'next':
                    times = list(range(self.max_time+1, self.max_time+1+circuit.ntime))
                else:
                    raise RuntimeError('Unknown time_placement: %s. Allowed values are early, late, next' % time_placement)

        if len(qubits) != circuit.nqubit: raise RuntimeError('len(qubits) != circuit.nqubit')
        if len(times) != circuit.ntime: raise RuntimeError('len(times) != circuit.ntime')

        circuit.slice(
            qubits=list(range(circuit.min_qubit, circuit.max_qubit+1)),
            qubits_to=qubits,
            times=list(range(circuit.min_time, circuit.max_time+1)),
            times_to=times,
            copy=copy,
            circuit_to=self,
            )

        return self

    def gate(
        self,
        qubits,
        times,
        ):

        # OK

        # Make qubits a tuple regardless of input
        qubits = (qubits,) if isinstance(qubits, int) else qubits
        # Make times a tuple regardless of input
        times = (times,) if isinstance(times, int) else times

        return self.gates[(times, qubits)]

    def remove_gate(
        self,
        qubits,
        times,
        ):

        # OK

        # Make qubits a tuple regardless of input
        qubits = (qubits,) if isinstance(qubits, int) else qubits
        # Make times a tuple regardless of input
        times = (times,) if isinstance(times, int) else times

        # Print sensible error message if key is invalid
        if (times, qubits) not in self.gates:
            raise RuntimeError('Key is not in circuit: (times=%r, qubits=%r)' % (times, qubits))

        # Delete the gate
        del self.gates[(times, qubits)]

        # Rebuild the indexing arrays
        self.qubits.clear() 
        self.times.clear() 
        self.times_and_qubits.clear()
        for key, gate in self.gates.items():
            times2, qubits2 = key
            for qubit in qubits2:
                self.qubits.add(qubit)
            for time in times2:
                self.times.add(time)
            for qubit in qubits2:
                for time in times2:
                    self.times_and_qubits.add((time, qubit))

        # If the user deleted the Gate entered in the last add_gate call, flush
        # the last_qubits/last_times history
        if qubits == self.last_qubits and times == self.last_times:
            self.last_qubits = None 
            self.last_times = None
                    
        return self

    def replace_gate(
        self,
        gate,
        qubits,
        times,
        name=None,
        ascii_symbols=None,
        ):

        # If gate is Circuit, make it a CompositeGate
        gate = CompositeGate(gate, name, ascii_symbols) if isinstance(gate, Circuit) else gate

        # Make qubits a tuple regardless of input
        qubits = (qubits,) if isinstance(qubits, int) else qubits
        # Make times a tuple regardless of input
        times = (times,) if isinstance(times, int) else times

        # Print sensible error message if key is invalid
        if (times, qubits) not in self.gates:
            raise RuntimeError('Key is not in circuit: (times=%r, qubits=%r)' % (times, qubits))
        
        # Check that qubits makes sense for gate.nqubit
        if len(qubits) != gate.nqubit: raise RuntimeError('%d qubit entries provided for %d-qubit gate' % (len(qubits), gate.nqubit))
        # Check that times makes sense for gate.ntime
        if len(times) != gate.ntime: raise RuntimeError('%d time entries provided for %d-time gate' % (len(times), gate.ntime))

        # Replace the gate
        self.gates[(times, qubits)] = gate 

        return self

    # => Slicing and Dicing <= #

    def copy(
        self,
        ):
    
        # OK

        """ Return a copy of circuit self so that parameter modifications in
            the copy do not affect self.

        Returns:
            (Circuit) - copy of self with all Gate objects copied deeply enough
                to remove parameter dependencies between self and returned
                Circuit.
        """

        circuit = Circuit()
        for key, gate in self.gates.items():
            times, qubits = key
            circuit.add_gate(times=times, qubits=qubits, gate=gate.copy())
        return circuit

    def slice(
        self,
        qubits=None,
        times=None,
        qubits_to=None,
        times_to=None,
        circuit_to=None,
        copy=True,
        ):
        
        # OK

        # (Rule 0): Seems bad to have a target but no source (TODO: think about this convention)
        if qubits_to is not None and qubits is None: raise RuntimeError('qubits_to is not None but qubits is None')
        if times_to is not None and times is None: raise RuntimeError('times_to is not None but times is None')

        # (Rule 1): Default to leaving dimensions untouched if not specified by the user
        if qubits is None: 
            qubits = self.qubits
            qubits_to = self.qubits
        if times is None: 
            times = self.times
            times_to = self.times
        
        # (Rule 2): Default to compressing explicitly sliced dimensions to 0, 1, ...
        if qubits_to is None: qubits_to = list(range(len(qubits)))
        if times_to is None: times_to = list(range(len(times)))

        # Validity checks
        if len(qubits) != len(qubits_to): raise RuntimeError('len(qubits) != len(qubits_to)') 
        if len(times) != len(times_to): raise RuntimeError('len(times) != len(times_to)') 

        # Map of qubit -> qubit_to (similar for time)
        qubit_map = { v : k for k, v in zip(qubits_to, qubits) }
        time_map = { v : k for k, v in zip(times_to, times) }

        # Circuit construction
        circuit_to = Circuit() if circuit_to is None else circuit_to
        for key, gate in self.gates.items():
            times2, qubits2 = key
            # Check if the gate is in the slice
            if any(time not in times for time in times2): continue
            if any(qubit not in qubits for qubit in qubits2): continue
            # New times (sorted by convention for multi-time gates)
            times3 = tuple(sorted(time_map[time] for time in times2))
            # New qubits
            qubits3 = tuple(qubit_map[qubit] for qubit in qubits2)
            # Gate addition
            circuit_to.add_gate(
                gate=gate,
                qubits=qubits3,
                times=times3,
                copy=copy,
                )
    
        return circuit_to
         
    @staticmethod
    def join_in_time(
        circuits,
        copy=True,
        ):

        # OK

        circuit1 = Circuit()
        for circuit in circuits:
            circuit.slice(
                times=list(range(circuit.min_time, circuit.max_time+1)),
                times_to=list(range(circuit1.ntime,circuit1.ntime+circuit.ntime)),
                circuit_to=circuit1,
                copy=copy,
                )
        return circuit1

    @staticmethod
    def join_in_qubits(
        circuits,
        copy=True,
        ):

        # OK

        circuit1 = Circuit()
        for circuit in circuits:
            circuit.slice(
                qubits=list(range(circuit.min_qubit, circuit.max_qubit+1)),
                qubits_to=list(range(circuit1.nqubit,circuit1.nqubit+circuit.nqubit)),
                circuit_to=circuit1,
                copy=copy,
                )
        return circuit1

    def reverse(
        self,
        copy=True,
        ):

        # OK

        return self.slice(
            times=list(reversed(self.times)),
            times_to=self.times,
            copy=copy,
            )

    def dagger(
        self,
        ):

        # OK

        circuit1 = self.reverse()
        circuit2 = Circuit()
        for key, gate in circuit1.gates.items():
            times, qubits = key
            circuit2.add_gate(
                gate=gate.dagger(),
                times=times,
                qubits=qubits,
                )
    
        return circuit2

    def sparse(
        self,
        sparse_in_qubits=True,
        sparse_in_time=True,
        copy=True,
        ):

        # OK

        return self.slice(
            qubits=self.qubits if sparse_in_qubits else None,
            times=self.times if sparse_in_time else None,
            copy=copy,
            )

    def center(
        self,
        center_in_qubits=True,
        center_in_times=True,
        origin_in_qubits=0,
        origin_in_time=0,
        copy=True,
        ):

        # OK

        return self.slice(
            qubits=self.qubits if center_in_qubits else None,
            qubits_to=[qubit - self.min_qubit + origin_in_qubits for qubit in self.qubits] if center_in_qubits else None,
            times=self.times if center_in_times else None,
            times_to=[time - self.min_time + origin_in_time for time in self.times] if center_in_times else None,
            copy=copy,
            )

    def explode(
        self,
        copy=True,
        ):

        circuit = Circuit()
        for key, gate in self.gates.items():
            times, qubits = key
            for key2, subgate in gate.exploded_gates().items():
                time2, qubits2 = key2
                qubits3 = tuple(qubits[_] for _ in qubits2)
                circuit.add_gate(gate=subgate, qubits=qubits3, copy=copy)
        return circuit

    def serialize_in_time(
        self,
        origin_in_time=0,
        copy=True,
        ):

        circuit = Circuit()
        for key, gate in self.gates.items():
            times, qubits = key
            circuit.add_gate(gate=gate, qubits=qubits, time_placement='next')
        return circuit
            
    # => Parameter Access/Manipulation <= #

    @property
    def nparameter(self):
        return len(self.parameter_keys)
    
    @property
    def parameters(self):
        parameters = collections.OrderedDict() 
        for key, gate in self.gates.items():
            times, qubits = key
            for key2, value in gate.parameters.items():
                parameters[(times, qubits, key2)] = value
        return parameters
    
    @property
    def parameter_keys(self):
        return list(self.parameters.keys())
        
    @property
    def parameter_values(self):
        return list(self.parameters.values())

    @property
    def parameter_gate_keys(self):
        return [(key[0], key[1]) for key in self.parameter_keys]

    @property
    def parameter_indices(self):
        """ A map from all circuit Gate keys to parameter indices. 

        Useful as a utility to determine the absolute parameter indices of a
        Gate, given knowledge of its Gate key.
        
        Returns:
            (OrderedDict of Gate key : tuple of int) - map from all circuit
                Gate keys to absolute parameter indices. For each Gate key, a
                tuple of absolute parameter indices is supplied - there may be
                no parameter indices, one parameter index, or multiple
                parameter indices in each value, depending on the number of
                parameters of the underlying Gate.
        """
        index_map = collections.OrderedDict()
        index = 0
        for key, gate in self.gates.items():
            index_map[key] = tuple(range(index, index + gate.nparameter))
            index += gate.nparameter
        return index_map

    @property
    def parameter_str(self):
        """ A human-readable string describing the circuit coordinates,
            parameter names, gate names, and values of all mutable parameters in
            this circuit.
        
        Returns:
            (str) - human-readable string describing parameters in order
                specified by param_keys.
        """ 
        s = ''
        s += '%-5s %-10s %-10s %-10s %-10s: %9s\n' % ('Index', 'Time', 'Qubits', 'Name', 'Gate', 'Value')
        I = 0
        for k, v in self.parameters.items():
            times, qubits, key2 = k
            gate = self.gates[(times, qubits)]
            if isinstance(key2, str):
                s += '%-5d %-10s %-10s %-10s %-10s: %9.6f\n' % (I, times, qubits, key2, gate.name, v)
            else:
                s += '%-5d %-10s %-10s %-10s %-10s:\n' % (I, times, qubits, '', gate.name)
                while True:
                    times, qubits, key2 = key2
                    if isinstance(key2, str):
                        s += '%-5s %-10s %-10s %-10s %-10s: %9.6f\n' % ('->', times, qubits, key2, gate.name, v)
                        break
                    else:
                        s += '%-5s %-10s %-10s %-10s %-10s:\n' % ('->', times, qubits, '', gate.name)
            I += 1
        return s

    def set_parameter(
        self,
        key,
        value,
        ):

        times, qubits, key2 = key
        self.gates[(times, qubits)].set_parameter(key=key2, value=value)
        return self

    def set_parameters(
        self, 
        parameters,
        ):
    
        for key, value in parameters.items():
            times, qubits, key2 = key
            self.gates[(times, qubits)].set_parameter(key=key2, value=value)
        return self

    def set_parameter_values(
        self,
        parameter_values,
        parameter_indices=None,
        ):

        parameter_keys = self.parameter_keys
        
        if parameter_indices is None:
            parameter_indices = list(range(len(parameter_keys)))

        for index, value in zip(parameter_indices, parameter_values):
            times, qubits, key2 = parameter_keys[index]
            self.gates[(times, qubits)].set_parameter(key=key2, value=value)
        
        return self

    # => ASCII Circuit Diagrams <= #

    def __str__(
        self,
        ):

        """ String representation of this Circuit (an ASCII circuit diagram). """
        return self.ascii_diagram(time_lines='both')

    def ascii_diagram(
        self,
        time_lines='both',
        ):

        """ Return a simple ASCII string diagram of the circuit.

        Params:
            time_lines (str) - specification of time lines:
                "both" - time lines on top and bottom (default)
                "top" - time lines on top 
                "bottom" - time lines on bottom
                "neither" - no time lines
        Returns:
            (str) - the ASCII string diagram
        """

        # Left side states
        Wd = max(len(str(_)) for _ in range(self.min_qubit, self.max_qubit+1)) if self.nqubit else 0
        lstick = '%-*s : |\n' % (1+Wd, 'T')
        for qubit in range(self.min_qubit, self.max_qubit+1): 
            lstick += '%*s\n' % (5+Wd, ' ')
            lstick += 'q%-*d : -\n' % (Wd, qubit)

        # Build moment strings
        moments = []
        for time in range(self.min_time, self.max_time+1):
            moments.append(self.ascii_diagram_time(
                time=time,
                adjust_for_time=False if time_lines=='neither' else True,
                ))

        # Unite strings
        lines = lstick.split('\n')
        for moment in moments:
            for i, tok in enumerate(moment.split('\n')):
                lines[i] += tok
        # Time on top and bottom
        lines.append(lines[0])

        # Adjust for time lines
        if time_lines == 'both':
            pass
        elif time_lines == 'top':
            lines = lines[:-2]
        elif time_lines == 'bottom':
            lines = lines[2:]
        elif time_lines == 'neither':
            lines = lines[2:-2]
        else:
            raise RuntimeError('Invalid time_lines argument: %s' % time_lines)
        
        strval = '\n'.join(lines)

        return strval

    def ascii_diagram_time(
        self,
        time,
        adjust_for_time=True,
        ):

        """ Return an ASCII string diagram for a given time moment time.

        Users should not generally call this utility routine - see
        ascii_diagram instead.

        Params:
            time (int) - time moment to diagram
            adjust_for_time (bool) - add space adjustments for the length of
                time lines.
        Returns:
            (str) - ASCII diagram for the given time moment.
        """

        # Subset for time, including multi-time gates
        gates = { key[1] : gate for key, gate in self.gates.items() if time in key[0] }

        # list (total seconds) of dict of A -> gate symbol
        seconds = [{}]
        # list (total seconds) of dict of A -> interstitial symbol
        seconds2 = [{}]
        for qubits, gate in gates.items():
            # Find the first second this gate fits within (or add a new one)
            for idx, second in enumerate(seconds):
                fit = not any(A in second for A in range(min(qubits), max(qubits)+1))
                if fit:
                    break
            if not fit:
                seconds.append({})
                seconds2.append({})
                idx += 1
            # Put the gate into that second
            for A in range(min(qubits), max(qubits)+1):
                # Gate symbol
                if A in qubits:
                    Aind = [Aind for Aind, B in enumerate(qubits) if A == B][0]
                    seconds[idx][A] = gate.ascii_symbols[Aind] 
                else:
                    seconds[idx][A] = '|'
                # Gate connector
                if A != min(qubits):
                    seconds2[idx][A] = '|'

        # + [1] for the - null character
        wseconds = [max([len(v) for k, v in second.items()] + [1]) for second in seconds]
        wtot = sum(wseconds)    

        # Adjust widths for T field
        Tsymb = '%d' % time
        if adjust_for_time:
            if wtot < len(Tsymb): wseconds[0] += len(Tsymb) - wtot
            wtot = sum(wseconds)    

        # TODO: Print CompositeGate like
        # 
        # T   : |-1|0|1|2|3|4|5  |6  |7  |
        #                                 
        # q-1 : -H------------------------
        #                                 
        # q0  : ------@---@-O-A---A---A---
        #             |   | | |===|===|   
        # q1  : ------X-@-X-@-|-B-|-B-|-B-
        #               |   | |=|=|=|=|=| 
        # q2  : --------X---X-A-|-A-|-A-|-
        #                       |===|===| 
        # q3  : ----------------B---B---B-
        # 
        # T   : |-1|0|1|2|3|4|5  |6  |7  |
        
        Is = ['' for A in range(self.nqubit)]
        Qs = ['' for A in range(self.nqubit)]
        for second, second2, wsecond in zip(seconds, seconds2, wseconds):
            for Aind, A in enumerate(range(self.min_qubit, self.max_qubit+1)):
                Isymb = second2.get(A, ' ')
                IwR = wsecond - len(Isymb)
                Is[Aind] += Isymb + ' ' * IwR + ' '
                Qsymb = second.get(A, '-')
                QwR = wsecond - len(Qsymb)
                Qs[Aind] += Qsymb + '-' * QwR + '-'

        strval = Tsymb + ' ' * (wtot + len(wseconds) - len(Tsymb) - 1) + '|\n' 
        for I, Q in zip(Is, Qs):
            strval += I + '\n'
            strval += Q + '\n'

        return strval

        
    # => Gate Addition Sugar <= #

    # TODO: Fix docs in sugar

    def I(
        self,
        qubit,
        **kwargs):
        
        """ Add an I (Identity) gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.I,
            qubits=(qubit,),
            **kwargs)

    def X(
        self,
        qubit,
        **kwargs):
        
        """ Add an X gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.X,
            qubits=(qubit,),
            **kwargs)

    def Y(
        self,
        qubit,
        **kwargs):
        
        """ Add an Y gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.Y,
            qubits=(qubit,),
            **kwargs)

    def Z(
        self,
        qubit,
        **kwargs):
        
        """ Add an Z gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.Z,
            qubits=(qubit,),
            **kwargs)

    def H(
        self,
        qubit,
        **kwargs):
        
        """ Add an H (Hadamard) gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.H,
            qubits=(qubit,),
            **kwargs)

    def S(
        self,
        qubit,
        **kwargs):
        
        """ Add an S gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.S,
            qubits=(qubit,),
            **kwargs)

    def ST(
        self,
        qubit,
        **kwargs):
        
        """ Add an S^+ gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.ST,
            qubits=(qubit,),
            **kwargs)

    def T(
        self,
        qubit,
        **kwargs):
        
        """ Add a T gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.T,
            qubits=(qubit,),
            **kwargs)

    def TT(
        self,
        qubit,
        **kwargs):
        
        """ Add a T^+ gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.TT,
            qubits=(qubit,),
            **kwargs)

    def Rx2(
        self,
        qubit,
        **kwargs):
        
        """ Add an Rx2 (Z -> Y basis) gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.Rx2,
            qubits=(qubit,),
            **kwargs)

    def Rx2T(
        self,
        qubit,
        **kwargs):
        
        """ Add an Rx2T (Y -> Z basis) gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """

        return self.add_gate(
            gate=Gate.Rx2T,
            qubits=(qubit,),
            **kwargs)

    def CX(
        self,
        qubitA,
        qubitB,
        **kwargs):

        """ Add a CX gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.CX,
            qubits=(qubitA, qubitB),
            **kwargs)

    def CY(
        self,
        qubitA,
        qubitB,
        **kwargs):

        """ Add a CY gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.CY,
            qubits=(qubitA, qubitB),
            **kwargs)

    def CZ(
        self,
        qubitA,
        qubitB,
        **kwargs):

        """ Add a CZ gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.CZ,
            qubits=(qubitA, qubitB),
            **kwargs)

    def CS(
        self,
        qubitA,
        qubitB,
        **kwargs):

        """ Add a CS gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.CS,
            qubits=(qubitA, qubitB),
            **kwargs)

    def CST(
        self,
        qubitA,
        qubitB,
        **kwargs):

        """ Add a CS^+ gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.CST,
            qubits=(qubitA, qubitB),
            **kwargs)

    def SWAP(
        self,
        qubitA,
        qubitB,
        **kwargs):

        """ Add a SWAP gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.SWAP,
            qubits=(qubitA, qubitB),
            **kwargs)

    def CCX(
        self,
        qubitA,
        qubitB,
        qubitC,
        **kwargs):

        """ Add a CCX gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control1 qubit index in self to add the gate into.
            qubitB (int) - control2 qubit index in self to add the gate into.
            qubitC (int) - target qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.CCX,
            qubits=(qubitA, qubitB, qubitC),
            **kwargs)

    def CSWAP(
        self,
        qubitA,
        qubitB,
        qubitC,
        **kwargs):

        """ Add a CSWAP gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - swap1 qubit index in self to add the gate into.
            qubitC (int) - swap2 qubit index in self to add the gate into.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.CSWAP,
            qubits=(qubitA, qubitB, qubitC),
            **kwargs)

    def Rx(
        self,
        qubit,
        theta=0.0,
        **kwargs):

        """ Add an Rx (X-rotation) gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            theta (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.Rx(theta=theta),
            qubits=(qubit,),
            **kwargs)

    def Ry(
        self,
        qubit,
        theta=0.0,
        **kwargs):

        """ Add an Ry (Y-rotation) gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            theta (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.Ry(theta=theta),
            qubits=(qubit,),
            **kwargs)

    def Rz(
        self,
        qubit,
        theta=0.0,
        **kwargs):

        """ Add an Rz (Z-rotation) gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            theta (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.Rz(theta=theta),
            qubits=(qubit,),
            **kwargs)

    def u1(
        self,
        qubit,
        lam=0.0,
        **kwargs):

        """ Add a u1 gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            lam (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.u1(lam=lam),
            qubits=(qubit,),
            **kwargs)

    def u2(
        self,
        qubit,
        phi=0.0,
        lam=0.0,
        **kwargs):

        """ Add a u3 gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            phi (float) - the angle parameter of the gate (default - 0.0).
            lam (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.u2(phi=phi, lam=lam),
            qubits=(qubit,),
            **kwargs)

    def u3(
        self,
        qubit,
        theta=0.0,
        phi=0.0,
        lam=0.0,
        **kwargs):

        """ Add a u3 gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            theta (float) - the angle parameter of the gate (default - 0.0).
            phi (float) - the angle parameter of the gate (default - 0.0).
            lam (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.u3(theta=theta, phi=phi, lam=lam),
            qubits=(qubit,),
            **kwargs)

    def SO4(
        self,
        qubitA,
        qubitB,
        A=0.0,
        B=0.0,
        C=0.0,
        D=0.0,
        E=0.0,
        F=0.0,
        **kwargs):

        """ Add an SO4 gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            A (float) - SO4 A parameter (default - 0.0).
            B (float) - SO4 B parameter (default - 0.0).
            C (float) - SO4 C parameter (default - 0.0).
            D (float) - SO4 D parameter (default - 0.0).
            E (float) - SO4 E parameter (default - 0.0).
            F (float) - SO4 F parameter (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.SO4(A=A, B=B, C=C, D=D, E=E, F=F),
            qubits=(qubitA, qubitB),
            **kwargs)

    def SO42(
        self,
        qubitA,
        qubitB,
        thetaIY=0.0,
        thetaYI=0.0,
        thetaYX=0.0,
        thetaXY=0.0,
        thetaZY=0.0,
        thetaYZ=0.0,
        **kwargs):

        """ Add an SO4 gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            thetaIY (float) - SO4 thetaIY parameter (default - 0.0).
            thetaYI (float) - SO4 thetaYI parameter (default - 0.0).
            thetaYX (float) - SO4 thetaYX parameter (default - 0.0).
            thetaXY (float) - SO4 thetaXY parameter (default - 0.0).
            thetaZY (float) - SO4 thetaZY parameter (default - 0.0).
            thetaYZ (float) - SO4 thetaYZ parameter (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.SO42(
                thetaIY=thetaIY, 
                thetaYI=thetaYI, 
                thetaYX=thetaYX, 
                thetaXY=thetaXY, 
                thetaZY=thetaZY, 
                thetaYZ=thetaYZ,
                ),
            qubits=(qubitA, qubitB),
            **kwargs)

    def CF(
        self,
        qubitA,
        qubitB,
        theta=0.0,
        **kwargs):

        """ Add a CF gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            theta (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.CF(theta=theta),
            qubits=(qubitA, qubitB),
            **kwargs)

    def R_ion(
        self,
        qubit,
        theta=0.0,
        phi=0.0,
        **kwargs):

        """ Add an R_ion gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            theta (float) - the angle parameter of the gate (default - 0.0).
            phi (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.R_ion(theta=theta, phi=phi),
            qubits=(qubit,),
            **kwargs)

    def Rx_ion(
        self,
        qubit,
        theta=0.0,
        **kwargs):

        """ Add an Rx_ion gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            theta (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.Rx_ion(theta=theta),
            qubits=(qubit,),
            **kwargs)

    def Ry_ion(
        self,
        qubit,
        theta=0.0,
        **kwargs):

        """ Add an Ry_ion gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            theta (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.Ry_ion(theta=theta),
            qubits=(qubit,),
            **kwargs)

    def Rz_ion(
        self,
        qubit,
        theta=0.0,
        **kwargs):

        """ Add an Rz_ion gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubit (int) - qubit index in self to add the gate into.
            theta (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.Rz_ion(theta=theta),
            qubits=(qubit,),
            **kwargs)

    def XX_ion(
        self,
        qubitA,
        qubitB,
        chi=0.0,
        **kwargs):

        """ Add an XX_ion gate to self at specified qubits and time,
            updating self. The qubits to add gate to are always explicitly
            specified. The time to add gate to may be explicitly specified in
            the time argumet (1st priority), or a recipe for determining the
            time placement can be specified using the time_placement argument
            (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            chi (float) - the angle parameter of the gate (default - 0.0).
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.XX_ion(chi=chi),
            qubits=(qubitA, qubitB),
            **kwargs)

    def U1(
        self,
        qubitA,
        U,
        **kwargs):

        """ Add a U1 gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            U (np.ndarray) - 2 x 2 unitary to construct the U1 gate from.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.U1(U=U),
            qubits=(qubitA,),
            **kwargs)

    def U2(
        self,
        qubitA,
        qubitB,
        U,
        **kwargs):

        """ Add a U2 gate to self at specified qubits and time, updating self.
            The qubits to add gate to are always explicitly specified. The time
            to add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qubitA (int) - control qubit index in self to add the gate into.
            qubitB (int) - target qubit index in self to add the gate into.
            U (np.ndarray) - 4 x 4 unitary to construct the U2 gate from.
            time (int) - time moment in self to add the gate into. If None, the
                time_placement argument will be considered next.
            time_placement (str - 'early', 'late', or 'next') - recipe to
                determine time moment in self to add the gate into. The rules
                are:
                    'early' -  add the gate as early as possible, just after
                        any existing gates on self's qubit wires.
                    'late' - add the gate in the last open time moment in self,
                        unless a conflict arises, in which case, add the gate
                        in the next (new) time moment.
                    'next' - add the gate in the next (new) time moment.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining
        """
        return self.add_gate(
            gate=Gate.U2(U=U),
            qubits=(qubitA, qubitB),
            **kwargs)

    def apply_to_statevector(
        self,
        statevector1,
        statevector2,
        qubits,
        dtype=np.complex128,
        ):

        if self.nqubit != len(qubits): raise RuntimeError('self.nqubit != len(qubits)')

        qubit_map = { qubit2 : qubit for qubit, qubit2 in zip(qubits, range(self.min_qubit, self.min_qubit + self.nqubit)) }

        for key, gate in self.gates.items(): 
            times, qubits2 = key
            gate.apply_to_statevector(
                statevector1=statevector1,
                statevector2=statevector2,
                qubits=tuple(qubit_map[qubit2] for qubit2 in qubits2),
                dtype=dtype,
                )
            statevector1, statevector2 = statevector2, statevector1
        
        return statevector1, statevector2
