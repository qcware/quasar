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
    """ The 1-qubit S^+ (Phase adjoint) matrix """

    T = np.array([[1.0, 0.0], [0.0, np.exp(np.pi/4.0*1.j)]], dtype=np.complex128)
    """ The 1-qubit T (sqrt-S) matrix """

    TT = np.array([[1.0, 0.0], [0.0, np.exp(-np.pi/4.0*1.j)]], dtype=np.complex128)
    """ The 1-qubit T (sqrt-S-adjoint) matrix """

    H = 1.0 / np.sqrt(2.0) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    """ The 1-qubit H (Hadamard) matrix """

    # exp(+i (pi/4) * X) : Z -> Y basis transformation
    Rx2 = 1.0 / np.sqrt(2.0) * np.array([[1.0, +1.0j], [+1.0j, 1.0]], dtype=np.complex128)
    """ The 1-qubit Z -> Y basis transformation matrix (a specific Rx matrix) """

    Rx2T = 1.0 / np.sqrt(2.0) * np.array([[1.0, -1.0j], [-1.0j, 1.0]], dtype=np.complex128)
    """ The 1-qubit Y -> Z basis transformation matrix (a specific Rx matrix) """

    II = np.kron(I, I)
    """ The 2-qubit I \\otimes I matrix """
    
    IX = np.kron(I, X)
    """ The 2-qubit I \\otimes X matrix """

    IY = np.kron(I, Y)
    """ The 2-qubit I \\otimes Y matrix """

    IZ = np.kron(I, Z)
    """ The 2-qubit I \\otimes Z matrix """

    XI = np.kron(X, I)
    """ The 2-qubit X \\otimes I matrix """

    XX = np.kron(X, X)
    """ The 2-qubit X \\otimes X matrix """

    XY = np.kron(X, Y)
    """ The 2-qubit X \\otimes Y matrix """

    XZ = np.kron(X, Z)
    """ The 2-qubit X \\otimes Z matrix """

    YI = np.kron(Y, I)
    """ The 2-qubit Y \\otimes I matrix """

    YX = np.kron(Y, X)
    """ The 2-qubit Y \\otimes X matrix """

    YY = np.kron(Y, Y)
    """ The 2-qubit Y \\otimes Y matrix """

    YZ = np.kron(Y, Z)
    """ The 2-qubit Y \\otimes Z matrix """

    ZI = np.kron(Z, I)
    """ The 2-qubit Z \\otimes I matrix """

    ZX = np.kron(Z, X)
    """ The 2-qubit Z \\otimes X matrix """

    ZY = np.kron(Z, Y)
    """ The 2-qubit Z \\otimes Y matrix """

    ZZ = np.kron(Z, Z)
    """ The 2-qubit Z \\otimes Z matrix """

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
    """ The 2-qubit CS^+ (controlled-S-adjoint) matrix """

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

    """ Class Gate represents a quantum gate operation, i.e., a (usually)
        unitary operator acting on a specific number of qubits.
        
    This specific Gate class represents a "primitive" gate with an explicitly
    defined operator that occupies a single time index. Primitive Gate objects
    do not contain any subsidiary Gate or Circuit objects.

    """
    
    def __init__(
        self,
        nqubit,
        operator_function,
        parameters,
        name,
        ascii_symbols,
        involutary=False,
        adjoint_function=None,
        ):

        """ Gate initializer (verbatim).

        Params/Attributes:
            nqubit (int) - number of qubits.
            operator_function (function of OrderedDict of str : float ->
                np.ndarray of shape (2**nqubit,)*2) - function that yields the
                Gate operator matrix corresponding to the current parameter
                dictionary.
            parameters (OrderedDict of str : float) - parameter declarations
                and initial values for this Gate. Gates without parameters will
                have an empty OrderedDict for this field.
            name (str) - name of Gate. The name of the gate and the number of
                qubits are occasionally used as a shorthand for the Gate
                definition. E.g., all 1-nqubit gates with the name "Ry" are
                assumed to be quasar Ry gates by certain transpilation and
                runtime functions.
            ascii_symbols (list of str of size nqubit) - ASCII symbol to
                display on each qubit index during ASCII circuit printing.
                involutary (bool, default False) - setting involutary to True
                (1st priority) indicates that the gate is its own adjoint,
                meaning that a copy of this Gate can be returned in the
                `adjoint` function.
            adjoint_function (function of OrderedDict of str : float -> Gate or
                None (default)) - setting adjoint function to a non-None type
                provides a custom recipe to return a sensible adjoint Gate for
                this Gate, with the current parameters provided as input
                arguments to the adjoint function.
        """
        
        self.nqubit = nqubit 
        self.operator_function = operator_function
        self.parameters = parameters
        self.name = name
        self.ascii_symbols = ascii_symbols
        self.involutary = involutary
        self.adjoint_function = adjoint_function

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
        if not isinstance(self.involutary, bool): raise RuntimeError('involutary must be bool')
        
    @property
    def ntime(self):
        """ (int) - Number of time indices occupied by this Gate (always 1) """
        return 1

    @property
    def is_composite(self):
        """ (bool) - Is this Gate CompositeGate (containing subgates) (always False) """
        return False

    @property
    def is_controlled(self):
        """ (bool) - Is this Gate ControlledGate (containing controls + Gate) (always False) """
        return False

    def __str__(self):
        """ String representation of this Gate (self.name) """
        return self.name

    @property
    def operator(self): 
        """ The (2**N,)*2 operator (unitary) matrix underlying this Gate. 

        The action of the gate on a given state is given graphically as,

        |\\Psi> -G- |\\Psi'>

        and mathematically as,

        |\\Psi_I'> = \\sum_J U_IJ |\\Psi_J>

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
            involutary=self.involutary,
            adjoint_function=self.adjoint_function,
            )

    # > Adjoint < #

    def adjoint(self):
        """ Make the adjoint of the current Gate (always a copy).

        Returns:
            (Gate) - a Gate representing the adjoint of this Gate. 
                - If self.involutary is True, a copy of self is returned.
                - Else if self.adjoint_function is not None,
                    self.adjoint_function(self.parameters) is called and used to
                    return the desired Gate.
                - Else an extended copy of this Gate with propery dressed
                    adjoint operator_function and '^+' added to name and
                    ascii_symbols is returned (works correctly always, but
                    ^+^+^+ runs build up under repeated calls to adjoint).
        """

        if self.involutary:
            return self.copy()
        elif self.adjoint_function:
            return self.adjoint_function(self.parameters)
        else:
            return Gate(
                nqubit=self.nqubit,
                operator_function=lambda parameters : self.operator_function(parameters).T.conj(),
                parameters=self.parameters.copy(), 
                name=self.name+'^+',
                ascii_symbols=[symbol + ('' if symbol in ['@', 'O'] else '^+') for symbol in self.ascii_symbols],
                involutary=self.involutary,
                adjoint_function=self.adjoint_function,
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
    involutary=True,
    )
""" I (identity) gate """

Gate.X = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.X,
    parameters=collections.OrderedDict(),
    name='X',
    ascii_symbols=['X'],
    involutary=True,
    )
""" X (NOT) gate """

Gate.Y = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.Y,
    parameters=collections.OrderedDict(),
    name='Y',
    ascii_symbols=['Y'],
    involutary=True,
    )
""" Y gate """

Gate.Z = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.Z,
    parameters=collections.OrderedDict(),
    name='Z',
    ascii_symbols=['Z'],
    involutary=True,
    )
""" Z gate """

Gate.H = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.H,
    parameters=collections.OrderedDict(),
    name='H',
    ascii_symbols=['H'],
    involutary=True,
    )
""" H (Hadamard) gate """

Gate.S = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.S,
    parameters=collections.OrderedDict(),
    name='S',
    ascii_symbols=['S'],
    adjoint_function = lambda parameters : Gate.ST,
    )
""" S gate """

Gate.ST = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.ST,
    parameters=collections.OrderedDict(),
    name='S^+',
    ascii_symbols=['S^+'],
    adjoint_function = lambda parameters : Gate.S,
    )
""" S^+ gate """

Gate.T = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.T,
    name='T',
    parameters=collections.OrderedDict(),
    ascii_symbols=['T'],
    adjoint_function = lambda parameters : Gate.TT,
    )
""" T gate """

Gate.TT = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.TT,
    name='T^+',
    parameters=collections.OrderedDict(),
    ascii_symbols=['T^+'],
    adjoint_function = lambda parameters : Gate.T,
    )
""" T^+ gate """

Gate.Rx2 = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.Rx2,
    parameters=collections.OrderedDict(),
    name='Rx2',
    ascii_symbols=['Rx2'],
    adjoint_function = lambda parameters : Gate.Rx2T,
    )
""" Rx2 gate """

Gate.Rx2T = Gate(
    nqubit=1,
    operator_function = lambda parameters : Matrix.Rx2T,
    parameters=collections.OrderedDict(),
    name='Rx2T',
    ascii_symbols=['Rx2T'],
    adjoint_function = lambda parameters : Gate.Rx2,
    )
""" Rx2T gate """

# > Explicit 2-body gates < #

Gate.CX = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.CX,
    parameters=collections.OrderedDict(),
    name='CX',
    ascii_symbols=['@', 'X'],
    involutary=True,
    )
""" CX (CNOT) gate """
Gate.CY = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.CY,
    parameters=collections.OrderedDict(),
    name='CY',
    ascii_symbols=['@', 'Y'],
    involutary=True,
    )
""" CY gate """
Gate.CZ = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.CZ,
    parameters=collections.OrderedDict(),
    name='CZ',
    ascii_symbols=['@', 'Z'],
    involutary=True,
    )
""" CZ gate """
Gate.CS = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.CS,
    parameters=collections.OrderedDict(),
    name='CS',
    ascii_symbols=['@', 'S'],
    adjoint_function = lambda parameters : Gate.CST,
    )
""" CS gate """
Gate.CST = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.CST,
    parameters=collections.OrderedDict(),
    name='CS^+',
    ascii_symbols=['@', 'S^+'],
    adjoint_function = lambda parameters : Gate.CS,
    )
""" CS^+ gate """
Gate.SWAP = Gate(
    nqubit=2,
    operator_function = lambda parameters: Matrix.SWAP,
    parameters=collections.OrderedDict(),
    name='SWAP',
    ascii_symbols=['X', 'X'],
    involutary=True,
    )
""" SWAP gate """

# > Explicit 3-body gates < #

Gate.CCX = Gate(
    nqubit=3,
    operator_function = lambda parameters: Matrix.CCX,
    parameters=collections.OrderedDict(),
    name='CCX',
    ascii_symbols=['@', '@', 'X'],
    involutary=True,
    )
""" CCX (Toffoli gate) """
Gate.CSWAP = Gate(
    nqubit=3,
    operator_function = lambda parameters: Matrix.CSWAP,
    parameters=collections.OrderedDict(),
    name='CSWAP',
    ascii_symbols=['@', 'X', 'X'],
    involutary=True,
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
        adjoint_function=lambda parameters : Gate.Rx(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.Ry(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.Rz(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.u1(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.u2(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.u3(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.SO4(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.SO42(**{ k : -v for k, v in parameters.items()})
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
        involutary=True,
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
        adjoint_function=lambda parameters : Gate.R_ion(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.Rx_ion(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.Ry_ion(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.Rz_ion(**{ k : -v for k, v in parameters.items()})
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
        adjoint_function=lambda parameters : Gate.XX_ion(**{ k : -v for k, v in parameters.items()})
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

    def adjoint(self):
        return CompositeGate(
            circuit=self.circuit.adjoint(),
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

    def adjoint(self):
        return ControlledGate(
            gate=self.gate.adjoint(),
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
        return_key=False,
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
            return_key (bool) - return self for chaining (False - default)
                or (times, qubits) key (True) to determine gate placement.
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid. 
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

        return (tuple(times), tuple(qubits)) if return_key else self

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

    def adjoint(
        self,
        ):

        # OK

        circuit1 = self.reverse()
        circuit2 = Circuit()
        for key, gate in circuit1.gates.items():
            times, qubits = key
            circuit2.add_gate(
                gate=gate.adjoint(),
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
        # return self.ascii_diagram(time_lines='both')
        return self.ascii_diagram2()

    ascii_diagram_max_width = 80
        
    def ascii_diagram2(
        self,
        max_width=None,
        ):

        max_width = Circuit.ascii_diagram_max_width if max_width is None else max_width

        # => Utility Class <= #

        class GatePrintingLayout(object):
            
            def __init__(
                self,
                min_qubit=0,
                max_qubit=-1,
                ascii_symbols=[], # None to indicate vertical connector
                ):
            
                self.min_qubit = min_qubit
                self.max_qubit = max_qubit
                self.ascii_symbols = ascii_symbols
                
            @property
            def ascii_symbol_enumeration(self):
                return [(qubit, symbol) for qubit, symbol in zip(range(self.min_qubit, self.max_qubit+1), self.ascii_symbols)]
                
            @property
            def max_ascii_width(self):
                return max(len(_) for _ in self.ascii_symbols if _ is not None)
            
            @staticmethod
            def interferes(layout1, layout2):
                if layout1.min_qubit > layout2.max_qubit: return False
                if layout1.max_qubit < layout2.min_qubit: return False
                return True
                
            @staticmethod
            def build(qubits, ascii_symbols):
                min_qubit = min(qubits)
                max_qubit = max(qubits)
                nqubit = max_qubit - min_qubit + 1
                ascii_symbols2 = [None] * nqubit
                for qubit, ascii_symbol in zip(qubits, ascii_symbols):
                    ascii_symbols2[qubit - min_qubit] = ascii_symbol
                return GatePrintingLayout(
                    min_qubit=min_qubit,
                    max_qubit=max_qubit,
                    ascii_symbols=ascii_symbols2,
                    )

        # => Logical Layout <= #
            
        # Map of time : list of [list of GatePrintingLayout]
        # Portion in [] represents a relative second within the time index
        layouts = { time : [[]] for time in range(self.min_time, self.max_time+1) }
        # List of (starting second, ending second) for all multi-time gates
        time_connector_keys = []
        # GatePrintingLayouts for all multi-time gates
        time_connector_layouts = []
        for key, gate in self.gates.items():
            times, qubits = key
            layout = GatePrintingLayout.build(qubits, gate.ascii_symbols)
            time_connector_key = []
            for time in times:
                found = False
                for second, layouts2 in enumerate(layouts[time]):
                    if any(GatePrintingLayout.interferes(layout, _) for _ in layouts2): 
                        continue
                    layouts2.append(layout)
                    found = True
                    break
                if not found:
                    second += 1
                    layouts[time].append([layout])
                time_connector_key.append((time, second))
            if len(times) > 1:
                time_connector_keys.append((time_connector_key[0], time_connector_key[-1]))
                time_connector_layouts.append(layout)
                    
        # List of tuple of (time, relative_second) for all absolute seconds
        seconds = []
        for time, second_layouts in layouts.items():
            seconds += [(time, _) for _ in range(len(second_layouts))]
            
        # Reverse map of (time, relative_second) : absolute second
        seconds_map = { v : k for k, v in enumerate(seconds) }
        
        # => ASCII Sizes <= #
        
        # Determine ASCII widths of each second
        seconds_ascii_widths = [max([_.max_ascii_width for _ in layouts[time][second]] + [0]) for time, second in seconds]
        
        # Adjust ASCII widths for time width
        time_widths_header = { time : len(str(time)) for time in range(self.min_time, self.max_time+1)}
        time_widths = { time : 0 for time in range(self.min_time, self.max_time+1)}
        for second_index, key in enumerate(seconds):
            time, second = key
            time_widths[time] += seconds_ascii_widths[second_index]
        for time, second_layouts in layouts.items():
            if len(second_layouts) > 0: time_widths[time] += len(second_layouts) - 1
        time_adjustments = { time : max(time_widths_header[time] - time_widths[time], 0) 
            for time in time_widths.keys()}
        for time, second_layouts in layouts.items():
            if len(second_layouts) == 0: continue
            second = len(second_layouts)-1 # Last second in moment gets adjusted
            second_index = seconds_map[(time, second)]
            seconds_ascii_widths[second_index] += time_adjustments[time]
        
        # Adjust ASCII widths for separation characters
        seconds_ascii_widths = [_ + 1 for _ in seconds_ascii_widths]
        
        # Determine ASCII starts of each second
        seconds_ascii_starts = [0]
        for index, width in enumerate(seconds_ascii_widths):
            seconds_ascii_starts.append(seconds_ascii_starts[index] + width)
        ascii_width = seconds_ascii_starts[-1] # Total width
        seconds_ascii_starts = seconds_ascii_starts[:-1]
        
        # => ASCII Art <= #
        
        nqubit = self.nqubit
        min_qubit = self.min_qubit
        
        wire_lines = [['-'] * ascii_width for _ in range(nqubit)]
        join_lines = [[' '] * ascii_width for _ in range(nqubit)]
        
        for time_connector_key, time_connector_layout in zip(time_connector_keys, time_connector_layouts):
            second1 = time_connector_key[0]
            second2 = time_connector_key[1]
            second1_index = seconds_map[second1]
            second2_index = seconds_map[second2]
            second1_start = seconds_ascii_starts[second1_index]
            second2_start = seconds_ascii_starts[second2_index]
            for qubit, symbol in time_connector_layout.ascii_symbol_enumeration[:-1]:
                join_lines[qubit - min_qubit][second1_start:second2_start] = '*' * (second2_start - second1_start)
            for qubit, symbol in time_connector_layout.ascii_symbol_enumeration:
                if symbol is None: continue
                wire_lines[qubit - min_qubit][second1_start:second2_start] = '=' * (second2_start - second1_start)
        
        for time, second_layouts in layouts.items():
            for second, layouts2 in enumerate(second_layouts):
                second_index = seconds_map[(time, second)]
                second_start = seconds_ascii_starts[second_index]
                for layout in layouts2:
                    for qubit in range(layout.min_qubit, layout.max_qubit+1):
                        wire_lines[qubit - min_qubit][second_start] = '|'
                    for qubit in range(layout.min_qubit, layout.max_qubit):
                        join_lines[qubit - min_qubit][second_start] = '|'
                    for qubit, symbol in layout.ascii_symbol_enumeration:
                        if symbol is None: continue
                        for index, char in enumerate(symbol):
                            wire_lines[qubit - min_qubit][second_start + index] = char      
        
        # => Assembly <= #

        wire_strs = [''.join(_) for _ in wire_lines]
        join_strs = [''.join(_) for _ in join_lines]
        time_str = ''.join(['%-*d|' % (v, k) for k, v in time_widths.items()])
        
        qwidth = max(len(str(index + min_qubit)) for index in range(nqubit)) if nqubit else 0
        
        # Pagination
        fwidth = qwidth + 6 # Width of first column
        effective_width = max_width - (qwidth + 6)
        page_starts = [0]
        for time in range(self.min_time, self.max_time+1):
            second_index = seconds_map[(time, 0)]
            second_ascii_start = seconds_ascii_starts[second_index]
            if second_ascii_start - page_starts[-1] > effective_width:
                second_index2 = seconds_map[(time - 1, 0)]
                second_ascii_start2 = seconds_ascii_starts[second_index2]
                if second_ascii_start2 == page_starts[-1]:
                    raise RuntimeError('time index %d is too large to fit' % time)
                page_starts.append(second_ascii_start2)
        page_starts.append(ascii_width)

        ascii_str = ''
        for page_index in range(len(page_starts) - 1):
            page_start = page_starts[page_index]
            page_stop = page_starts[page_index+1]

            ascii_str += 'T%*s : %s%s\n' % (
                qwidth,
                ' ',
                '|' if page_index == 0 else ' ',
                time_str[page_start:page_stop],
                )

            ascii_str += '\n'
    
            for index, wire_str in enumerate(wire_strs): 
                join_str = join_strs[index]
                ascii_str += 'q%-*d : %s%s\n' % (
                    qwidth,
                    index + min_qubit,
                    '-' if page_index == 0 else ' ',
                    wire_str[page_start:page_stop],
                    )
                ascii_str += ' %-*s   %s%s\n' % (
                    qwidth,
                    ' ',
                    ' ',
                    join_str[page_start:page_stop],
                    )

            ascii_str += 'T%*s : %s%s\n' % (
                qwidth,
                ' ',
                '|' if page_index == 0 else ' ',
                time_str[page_start:page_stop],
                )

            if page_index < len(page_starts) - 2:
                ascii_str += '\n'
        
        return ascii_str

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

    # => Simulation Utility <= #

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
            statevector1, statevector2 = gate.apply_to_statevector(
                statevector1=statevector1,
                statevector2=statevector2,
                qubits=tuple(qubit_map[qubit2] for qubit2 in qubits2),
                dtype=dtype,
                )
        
        return statevector1, statevector2
