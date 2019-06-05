# Quasar: an ultralight python-2.7/python-3.X quantum simulator package
# Copyright (C) 2019 QC Ware Corporation - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Robert Parrish <rob.parrish@qcware.com>, 2019

import numpy as np
import collections
import itertools
from .measurement import Ket, Measurement

""" Quasar: an ultralight python-2.7/python-3.X quantum simulator package

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

# => Matrix class <= #

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
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    Y = np.array([[0.0, -1.0j], [+1.0j, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    T = np.array([[1.0, 0.0], [0.0, np.exp(np.pi/4.0*1.j)]], dtype=np.complex128)
    H = 1.0 / np.sqrt(2.0) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    # exp(+i (pi/4) * X) : Z -> Y basis transformation
    Rx2 = 1.0 / np.sqrt(2.0) * np.array([[1.0, +1.0j], [+1.0j, 1.0]], dtype=np.complex128)
    Rx2T = 1.0 / np.sqrt(2.0) * np.array([[1.0, -1.0j], [-1.0j, 1.0]], dtype=np.complex128)

    II = np.kron(I, I)
    IX = np.kron(I, X)
    IY = np.kron(I, Y)
    IZ = np.kron(I, Z)
    XI = np.kron(X, I)
    XX = np.kron(X, X)
    XY = np.kron(X, Y)
    XZ = np.kron(X, Z)
    YI = np.kron(Y, I)
    YX = np.kron(Y, X)
    YY = np.kron(Y, Y)
    YZ = np.kron(Y, Z)
    ZI = np.kron(Z, I)
    ZX = np.kron(Z, X)
    ZY = np.kron(Z, Y)
    ZZ = np.kron(Z, Z)

    CX = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.complex128)
    CY = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0j],
        [0.0, 0.0, +1.0j, 0.0],
        ], dtype=np.complex128)
    CZ = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        ], dtype=np.complex128)
    CS = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0j],
        ], dtype=np.complex128)
    SWAP = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.complex128)

    @staticmethod
    def Rx(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -1.j*s], [-1.j*s, c]], dtype=np.complex128)

    @staticmethod
    def Ry(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [+s, c]], dtype=np.complex128)

    @staticmethod
    def Rz(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c-1.j*s, 0.0], [0.0, c+1.j*s]], dtype=np.complex128)
    
# => Gate class <= #

class Gate(object):

    """ Class Gate represents a general N-body quantum gate. 

    An N-body quantum gate applies a unitary operator to the state of a subset
    of N qubits, with an implicit identity matrix acting on the remaining
    qubits. The Gate class specifies the (2**N,)*2 unitary matrix U for the N
    active qubits, but does not specify which qubits are active.

    Usually, most users will not initialize their own Gates, but will use gates
    from the standard library, which are defined as Gate class members (for
    parameter-free gates) or Gate class methods (for parameter-including gates).
    Some simple examples include:

    >>> I = Gate.I
    >>> Ry = Gate.Ry(theta=np.pi/4.0)
    >>> SO4 = Gate.SO4(A=0.0, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0)
    >>> CF = Gate.CF(theta=np.pi/3.0)
    """

    def __init__(
        self,
        N,
        Ufun,
        params,
        name,
        ascii_symbols,
        ):

        """ Initializer. Params are set as object attributes.

        Params:
            N (int > 0) - the dimensionality of the quantum gate, e.g, 1 for
                1-body, 2 for 2-body, etc.
            Ufun (function of OrderedDict of str : float -> np.ndarray of shape
                (2**N,)*2) - a function which generates the unitary
                matrix for this gate from the current parameter set.
            params (OrderedDict of str : float) - the dictionary of initial
                gate parameters.
            name (str) - a simple name for the gate, e.g., 'CX'
            ascii_symbols (list of str of len N) - a list of ASCII symbols for
                each active qubit of the gate, for use in generating textual diagrams, e.g.,
                ['@', 'X'] for CX.
        """
        
        self.N = N
        self.Ufun = Ufun
        self.params = params
        self.name = name
        self.ascii_symbols = ascii_symbols

        # Validity checks
        if not isinstance(self.N, int): raise RuntimeError('N must be int')
        if self.N <= 0: raise RuntimeError('N <= 0') 
        if self.U.shape != (2**self.N,)*2: raise RuntimeError('U must be shape (2**N,)*2')
        if not isinstance(self.params, collections.OrderedDict): raise RuntimeError('params must be collections.OrderedDict')
        if not all(isinstance(_, str) for _ in list(self.params.keys())): raise RuntimeError('params keys must all be str')
        if not all(isinstance(_, float) for _ in list(self.params.values())): raise RuntimeError('params values must all be float')
        if not isinstance(self.name, str): raise RuntimeError('name must be str')
        if not isinstance(self.ascii_symbols, list): raise RuntimeError('ascii_symbols must be list')
        if len(self.ascii_symbols) != self.N: raise RuntimeError('len(ascii_symbols) != N')
        if not all(isinstance(_, str) for _ in self.ascii_symbols): raise RuntimeError('ascii_symbols must all be str')

    def __str__(self):
        """ String representation of this Gate (self.name) """
        return self.name
    
    @property
    def U(self): 
        """ The (2**N,)*2 unitary matrix underlying this Gate. 

        The action of the gate on a given state is given graphically as,

        |\Psi> -G- |\Psi'>

        and mathematically as,

        |\Psi_I'> = \sum_J U_IJ |\Psi_J>

        Returns:
            (np.ndarray of shape (2**N,)*2) - the unitary matrix underlying
                this gate, built from the current parameter state.
        """
        return self.Ufun(self.params)

    # > Copying < #
    
    def copy(self):
        """ Make a deep copy of the current Gate. 
        
        Returns:
            (Gate) - a copy of this Gate whose parameters may be modified
                without modifying the parameters of self.
        """
        return Gate(
            N=self.N, 
            Ufun=self.Ufun, 
            params=self.params.copy(), 
            name=self.name,  
            ascii_symbols=self.ascii_symbols,
            )

    # > Parameter Access < #

    def set_param(self, key, param):
        """ Set the value of a parameter of this Gate. 

        Params:
            key (str) - the key of the parameter
            param (float) - the value of the parameter
        Result:
            self.params[key] = param. If the Gate does not have a parameter
                corresponding to key, a RuntimeError is thrown.
        """
        if key not in self.params: raise RuntimeError('Key %s is not in params' % key)
        self.params[key] = param

    def set_params(self, params):
        """ Set the values of multiple parameters of this Gate.

        Params:
            params (dict of str : float) -  dict of param values
        Result:
            self.params is updated with the contents of params by calling
                self.set_param for each key/value pair.
        """
        for key, param in params.items():
            self.set_param(key=key, param=param)

# > Explicit 1-body gates < #

""" I (identity) gate """
Gate.I = Gate(
    N=1,
    Ufun = lambda params : Matrix.I,
    params=collections.OrderedDict(),
    name='I',
    ascii_symbols=['I'],
    )
""" X (NOT) gate """
Gate.X = Gate(
    N=1,
    Ufun = lambda params : Matrix.X,
    params=collections.OrderedDict(),
    name='X',
    ascii_symbols=['X'],
    )
""" Y gate """
Gate.Y = Gate(
    N=1,
    Ufun = lambda params : Matrix.Y,
    params=collections.OrderedDict(),
    name='Y',
    ascii_symbols=['Y'],
    )
""" Z gate """
Gate.Z = Gate(
    N=1,
    Ufun = lambda params : Matrix.Z,
    params=collections.OrderedDict(),
    name='Z',
    ascii_symbols=['Z'],
    )
""" H (Hadamard) gate """
Gate.H = Gate(
    N=1,
    Ufun = lambda params : Matrix.H,
    params=collections.OrderedDict(),
    name='H',
    ascii_symbols=['H'],
    )
""" S gate """
Gate.S = Gate(
    N=1,
    Ufun = lambda params : Matrix.S,
    params=collections.OrderedDict(),
    name='S',
    ascii_symbols=['S'],
    )
""" T gate """
Gate.T = Gate(
    N=1,
    Ufun = lambda params : Matrix.T,
    name='T',
    params=collections.OrderedDict(),
    ascii_symbols=['T'],
    )
""" Rx2 gate """
Gate.Rx2 = Gate(
    N=1,
    Ufun = lambda params : Matrix.Rx2,
    params=collections.OrderedDict(),
    name='Rx2',
    ascii_symbols=['Rx2'],
    )
""" Rx2T gate """
Gate.Rx2T = Gate(
    N=1,
    Ufun = lambda params : Matrix.Rx2T,
    params=collections.OrderedDict(),
    name='Rx2T',
    ascii_symbols=['Rx2T'],
    )

# > Explicit 2-body gates < #

""" CX (CNOT) gate """
Gate.CX = Gate(
    N=2,
    Ufun = lambda params: Matrix.CX,
    params=collections.OrderedDict(),
    name='CX',
    ascii_symbols=['@', 'X'],
    )
""" CY gate """
Gate.CY = Gate(
    N=2,
    Ufun = lambda params: Matrix.CY,
    params=collections.OrderedDict(),
    name='CY',
    ascii_symbols=['@', 'Y'],
    )
""" CZ gate """
Gate.CZ = Gate(
    N=2,
    Ufun = lambda params: Matrix.CZ,
    params=collections.OrderedDict(),
    name='CZ',
    ascii_symbols=['@', 'Z'],
    )
""" CS gate """
Gate.CS = Gate(
    N=2,
    Ufun = lambda params: Matrix.CS,
    params=collections.OrderedDict(),
    name='CS',
    ascii_symbols=['@', 'S'],
    )
""" SWAP gate """
Gate.SWAP = Gate(
    N=2,
    Ufun = lambda params: Matrix.SWAP,
    params=collections.OrderedDict(),
    name='SWAP',
    ascii_symbols=['X', 'X'],
    )

# > Parametrized 1-body gates < #

@staticmethod
def _GateRx(theta):

    """ Rx (theta) = exp(-i * theta * x) """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -1.j*s], [-1.j*s, c]], dtype=np.complex128)
    
    return Gate(
        N=1,
        Ufun=Ufun,
        params=collections.OrderedDict([('theta', theta)]),
        name='Rx',
        ascii_symbols=['Rx'],
        )
    
@staticmethod
def _GateRy(theta):

    """ Ry (theta) = exp(-i * theta * Y) """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [+s, c]], dtype=np.complex128)

    return Gate(
        N=1,
        Ufun=Ufun,
        params=collections.OrderedDict([('theta', theta)]),
        name='Ry',
        ascii_symbols=['Ry'],
        )
    
@staticmethod
def _GateRz(theta):

    """ Rz (theta) = exp(-i * theta * Z) """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c-1.j*s, 0.0], [0.0, c+1.j*s]], dtype=np.complex128)

    return Gate(
        N=1,
        Ufun=Ufun,
        params=collections.OrderedDict([('theta', theta)]),
        name='Rz',
        ascii_symbols=['Rz'],
        )
    
Gate.Rx = _GateRx
Gate.Ry = _GateRy
Gate.Rz = _GateRz

# > Parametrized 2-body gates < #

@staticmethod
def _GateSO4(A, B, C, D, E, F):
    
    def Ufun(params):
        A = params['A']
        B = params['B']
        C = params['C']
        D = params['D']
        E = params['E']
        F = params['F']
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
        N=2,
        Ufun=Ufun,
        params=collections.OrderedDict([('A', A), ('B', B), ('C', C), ('D', D), ('E', E), ('F', F)]),
        name='SO4',
        ascii_symbols=['SO4A', 'SO4B'],
        )

Gate.SO4 = _GateSO4

@staticmethod
def _GateSO42(thetaIY, thetaYI, thetaXY, thetaYX, thetaZY, thetaYZ):
    
    def Ufun(params):
        A = -(params['thetaIY'] + params['thetaZY'])
        F = -(params['thetaIY'] - params['thetaZY'])
        C = -(params['thetaYX'] + params['thetaXY'])
        D = -(params['thetaYX'] - params['thetaXY'])
        B = -(params['thetaYI'] + params['thetaYZ'])
        E = -(params['thetaYI'] - params['thetaYZ'])
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
        N=2,
        Ufun=Ufun,
        params=collections.OrderedDict([
            ('thetaIY' , thetaIY),
            ('thetaYI' , thetaYI),
            ('thetaXY' , thetaXY),
            ('thetaYX' , thetaYX),
            ('thetaZY' , thetaZY),
            ('thetaYZ' , thetaYZ),
        ]),
        name='SO42',
        ascii_symbols=['SO42A', 'SO42B'],
        )

Gate.SO42 = _GateSO42

@staticmethod
def _CF(theta):

    """ Controlled F gate """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0,  +c,  +s],
            [0.0, 0.0,  +s,  -c],
            ], dtype=np.complex128)
    
    return Gate(
        N=2,
        Ufun=Ufun,
        params=collections.OrderedDict([('theta', theta)]),
        name='CF',
        ascii_symbols=['@', 'F'],
        )

Gate.CF = _CF

# > Special explicit gates < #

@staticmethod
def _GateU1(U):

    """ An explicit 1-body gate that is specified by the user. """

    return Gate(
        N=1,
        Ufun = lambda params : U,
        params=collections.OrderedDict(),
        name='U1',
        ascii_symbols=['U1'],
        )

@staticmethod
def _GateU2(U):

    """ An explicit 2-body gate that is specified by the user. """

    return Gate(
        N=2,
        Ufun = lambda params : U,
        params=collections.OrderedDict(),
        name='U2',
        ascii_symbols=['U2A', 'U2B'],
        )

Gate.U1 = _GateU1
Gate.U2 = _GateU2

# => Gate class <= #

class Circuit(object):

    """ Class Circuit represents a general quantum circuit acting on N
        linearly-arranged cubits. Non-local connectivity is permitted - the
        linear arrangement is strictly for simplicity.

        An example Circuit construction is,

        >>> circuit = Circuit(N=2).H(0).X(1).CX(0,1)
        >>> print(circuit)
        
        A Circuit is always constructed with a fixed number of qubits N, but
        the time window of the circuit is freely expandable from time=0 onward.
        The Circuit starts empty, and is filled one gate at a time by the
        add_gate function or by helper methods such as H, X, CX, etc.
    
        The Circuit attribute Ts (list of int) contains the sorted list of time
        indices T with significant gates, and the Circuit attribute ntime
        (int) contains the total number of time moments, including empty
        moments.

        The core data of a Circuit is the gates attribute, which contains an
        OrderedDict of (time, qubits) : Gate pairs for significant gates. The
        (time, qubits) compound key specifies the time moment of the gate
        (int), and the qubit indices (tuple of int). len(qubits) is always
        gate.N.  
    """

    def __init__(
        self,
        N,
        ):

        """ Initializer.

        Params:
            N (int) - number of qubits in this circuit
        """

        self.N = N
        # All circuits must have at least one qubit
        if self.N <= 0: raise RuntimeError('N <= 0')
    
        # Primary circuit data structure
        self.gates = collections.OrderedDict() # (T, (A, [B], [C], ...)) -> Gate
        # Memoization
        self.Ts = [] # [T] tells ordered, unique time moments
        self.TAs = set() # ({T,A}) tells occupied circuit indices

    # > Simple Circuit characteristics < #

    @property
    def ntime(self):
        """ The total number of time moments in the circuit (including blank moments) """
        return self.Ts[-1] + 1 if len(self.Ts) else 0

    @property
    def ngate(self):
        """ The total number of gates in the circuit. """
        return len(self.gates)

    @property
    def ngate1(self):
        """ The total number of 1-body gates in the circuit. """
        return len([gate for gate in list(self.gates.values()) if gate.N == 1])

    @property
    def ngate2(self):
        """ The total number of 2-body gates in the circuit. """
        return len([gate for gate in list(self.gates.values()) if gate.N == 2])

    # > Gate addition < #

    def add_gate(
        self,
        gate,
        qubits,
        time=None, 
        time_placement='early',
        copy=True,
        ):

        """ Add a gate to self at specified qubits and time, updating self. The
            qubits to add gate to are always explicitly specified. The time to
            add gate to may be explicitly specified in the time argumet (1st
            priority), or a recipe for determining the time placement can be
            specified using the time_placement argument (2nd priority).

        Params:
            qate (Gate) - the gate to add into self. 
            qubits (int or tuple of int) - ordered qubit indices in self to add the
                qubit indices of circuit into. If a single int is provided (for
                one-qubit gate addition), it is converted to a tuple with a
                single int entry.
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
            copy (bool) - copy the gate or not?
        Result:
            self is updated with the added gate. Checks are
                performed to ensure that the addition is valid.
        Returns:
            self - for chaining

        For one body gate, can add as either of:
            circuit.add_gate(gate, A, time)
            circuit.add_gate(gate, (A,), time)
        For two body gate, must add as:
            circuit.add_gate(gate, (A, B), time)
        """

        # Make qubits a tuple regardless of input
        qubits = (qubits,) if isinstance(qubits, int) else qubits

        # Time determination
        if time is None:
            if time_placement == 'early':
                timemax = -1
                for time2, A in self.TAs:
                    if A in qubits:
                        timemax = max(timemax, time2)
                time = timemax + 1
            elif time_placement == 'late':
                time = max(self.ntime - 1, 0)
                if any((time, A) in self.TAs for A in qubits):
                    time += 1 
            elif time_placement == 'next':
                time = self.ntime
            else:
                raise RuntimeError('Unknown time_placement: %s. Allowed values are early, late, next' % time_placement)

        # Check that time >= 0
        if time < 0: raise RuntimeError('Negative time: %d' % time)
        # Check that qubits makes sense for gate.N
        if len(qubits) != gate.N: raise RuntimeError('%d qubits entries provided for %d-body gate' % (len(qubits), gate.N))
        # Check that the requested circuit locations are open
        for A in qubits:
            if (time,A) in self.TAs: 
                raise RuntimeError('time=%d, A=%d circuit location is already occupied' % (time,A))
            if A >= self.N:
                raise RuntimeError('No qubit location %d' % A)
        # Add gate to circuit
        self.gates[(time, qubits)] = gate.copy() if copy else gate
        # Update memoization of TAs and Ts
        for A in qubits:
            self.TAs.add((time,A))
        if time not in self.Ts:
            self.Ts = list(sorted(self.Ts + [time]))

        return self

    def gate(
        self,
        qubits,
        time,
        ):

        """ Return the gate at a given moment time and qubit indices qubits

        Params:
            qubits (int or tuple of int) - the qubit index or indices of the gate
            time (int) - the time index of the gate
        Returns:
            (Gate) - the gate at the specified circuit coordinates

        For one body gate, can use as either of:
            gate = circuit.gate(time, A)
            gate = circuit.gate(time, (A,))
        For two body gate, must use as:
            gate = circuit.gate(time, (A, B))
        """

        # Make qubits a tuple regardless of input
        qubits = (qubits,) if isinstance(qubits, int) else qubits
        return self.gates[(time, qubits)]

    # => Copy/Subsets/Concatenation <= #

    def copy(
        self,
        ):

        """ Return a copy of circuit self so that parameter modifications in
            the copy do not affect self.

        Returns:
            (Circuit) - copy of self with all Gate objects copied deeply enough
                to remove parameter dependencies between self and returned
                Circuit.
        """

        circuit = Circuit(N=self.N)
        for key, gate in self.gates.items():
            T, qubits = key
            circuit.add_gate(time=T, qubits=qubits, gate=gate.copy())
        return circuit

    def subset(
        self,
        times,
        copy=True,
        ):

        """ Return a Circuit with a subset of time moments times.

        Params:
            times (list of int) - ordered time moments to slice into time moments
                [0,1,2,...] in the returned circuit.
            copy (bool) - copy Gate elements to remove parameter dependencies
                between self and returned circuit (True - default) or not
                (False). 
        Returns:
            (Circuit) - the new time-sliced circuit.
        """

        circuit = Circuit(N=self.N)
        for T2, Tref in enumerate(times):
            if Tref >= self.ntime: raise RuntimeError('time >= self.ntime: %d' % Tref)
        for key, gate in self.gates.items():
            T, qubits = key
            if T in times:
                T2 = [T2 for T2, Tref in enumerate(times) if Tref == T][0]
                circuit.add_gate(time=T2, qubits=qubits, gate=gate.copy() if copy else gate)
        return circuit

    @staticmethod
    def concatenate(
        circuits,
        copy=True,
        ):

        """ Concatenate a list of Circuits in time.
        
        Params:
            circuits (list of Circuit) - the ordered list of Circuit objects to
                concatenate in time.
            copy (bool) - copy Gate elements to remove parameter dependencies
                between circuits and returned circuit (True - default) or not
                (False). 
        Returns:
            (Circuit) - the new time-concatenated circuit.
        """

        if any(x.N != circuits[0].N for x in circuits): 
            raise RuntimeError('Circuits must all have same N to be concatenated')
        
        circuit = Circuit(N=circuits[0].N)
        Tstart = 0
        for circuit2 in circuits:   
            for key, gate in circuit2.gates.items():
                T, qubits = key
                circuit.add_gate(time=T+Tstart, qubits=qubits, gate=gate.copy() if copy else gate)
            Tstart += circuit2.ntime
        return circuit

    def deadjoin(
        self,
        qubits,
        copy=True,
        ):

        """ Return a circuit with a subset of qubits.

        Params:
            qubits (list of int) - ordered qubit indices to slice in spatial
                indices into the [0,1,2...] indices in the returned circuit.
            copy (bool) - copy Gate elements to remove parameter dependencies
                between self and returned circuit (True - default) or not
                (False). 
        Returns:
            (Circuit) - the new qubit-sliced circuit.
        """

        for A2, Aref in enumerate(qubits):
            if Aref >= self.N: raise RuntimeError('A >= self.A: %d' % Aref)

        Amap = { v : k for k, v in enumerate(qubits) }

        circuit = Circuit(N=len(qubits))
        for key, gate in self.gates.items():
            T, qubits = key
            if all(x in Amap for x in qubits):
                circuit.add_gate(time=T, qubits=tuple(Amap[x] for x in qubits), gate=gate.copy() if copy else gate)
        return circuit

    @staticmethod
    def adjoin(
        circuits,
        copy=True,
        ):

        """ Adjoin a list of Circuits in spatial qubit indices.
        
        Params:
            circuits (list of Circuit) - the ordered list of Circuit objects to
                adjoin in spatial qubit indices.
            copy (bool) - copy Gate elements to remove parameter dependencies
                between circuits and returned circuit (True - default) or not
                (False). 
        Returns:
            (Circuit) - the new spatially qubit-adjoined circuit.
        """
        circuit = Circuit(N=sum(x.N for x in circuits))
        Astart = 0
        for circuit2 in circuits:   
            for key, gate in circuit2.gates.items():
                T, qubits = key
                circuit.add_gate(time=T, qubits=tuple(x + Astart for x in qubits), gate=gate.copy() if copy else gate)
            Astart += circuit2.N
        return circuit
    
    def reversed(
        self,
        copy=True,
        ):

        """ Return a circuit with gate operations in reversed time order.

        Note that the gates are not transposed/adjointed => this is not
            generally equivalent to time reversal.

        Params:
            copy (bool) - copy Gate elements to remove parameter dependencies
                between self and returned circuit (True - default) or not
                (False). 
        Returns:
            (Circuit) - the new reversed circuit.
        """

        circuit = Circuit(N=self.N)
        for key, gate in self.gates.items():
            T, qubits = key
            circuit.add_gate(time=self.ntime-T-1, qubits=qubits, gate=gate)
        return circuit

    def nonredundant(
        self,
        copy=True,
        ):

        """ Return a circuit with empty time moments removed.

        Params:
            copy (bool) - copy Gate elements to remove parameter dependencies
                between self and returned circuit (True - default) or not
                (False). 
        Returns:
            (Circuit) - the new time-dense circuit.
        """

        circuit = Circuit(N=self.N)
        Tmap = { v : k for k, v in enumerate(sorted(self.Ts)) }
        for key, gate in self.gates.items():
            T, qubits = key
            circuit.add_gate(time=Tmap[T], qubits=qubits, gate=gate)
        return circuit

    def compressed(
        self,
        ):

        """ Return an equivalent time-dense circuit with 1- and 2-body gates
            merged together to minimize the number of gates by using composite
            1- and 2-body gate operations. This operation is designed to reduct
            the runtime of state vector simulation by reducing the number of 1-
            and 2-body gate operations that must be simulated.

        This operation freezes the current parameter values of all gates, and
        constructs composite 1- and 2-body gates from the current values of the gate
        unitary matrices U. Therefore, the returned circuit will have no
        parameters, and compressed will have to be called on the original
        circuit again if the parameters change.

        Returns:
            (Circuit) - the new compressed circuit.
        """

        # Jam consecutive 1-body gates (removes runs of 1-body gates)
        circuit1 = self.copy()
        plan = [[0 for x in range(self.ntime)] for y in range(self.N)]
        for key, gate in circuit1.gates.items():
            T, qubits = key
            if gate.N == 1:
                A, = qubits
                plan[A][T] = 1
            elif gate.N == 2:
                A, B = qubits
                plan[A][T] = 2
                plan[B][T] = -2
            else:
                raise RuntimeError("N > 2")
        circuit2 = Circuit(N=self.N)
        for A, row in enumerate(plan):
            Tstar = None
            U = None
            for T, V in enumerate(row):
                # Start the 1-body gate chain
                if V == 1 and U is None:
                    Tstar = T
                    U = np.copy(circuit1.gates[T,(A,)].U)
                # Continue the 1-body gate chain
                elif V == 1:
                    U = np.dot(circuit1.gates[T,(A,)].U, U)
                # If 2-body gate or end of circuit encountered, place 1-body gate
                if U is not None and (V == 2 or V == -2 or T == self.ntime - 1):
                    circuit2.add_gate(time=Tstar, qubits=(A,), gate=Gate.U1(U=U))
                    Tstar = None
                    U = None
        for key, gate in circuit1.gates.items():
            T, qubits = key
            if gate.N == 2:
                circuit2.add_gate(time=T, qubits=qubits, gate=gate)

        # Jam 1-body gates into 2-body gates if possible (not possible if 1-body gate wire)
        circuit1 = circuit2
        plan = [[0 for x in range(self.ntime)] for y in range(self.N)]
        for key, gate in circuit1.gates.items():
            T, qubits = key
            if gate.N == 1:
                A, = qubits
                plan[A][T] = 1
            elif gate.N == 2:
                A, B = qubits
                plan[A][T] = 2
                plan[B][T] = -2
            else:
                raise RuntimeError("N > 2")
        circuit2 = Circuit(N=self.N)
        jammed_gates = {}                 
        for key, gate in circuit1.gates.items():
            if gate.N != 2: continue
            T, qubits = key
            A, B = qubits
            U = np.copy(gate.U)
            # Left-side 1-body gates
            for T2 in range(T-1,-1,-1):
                if plan[A][T2] == 2 or plan[A][T2] == -2: break
                if plan[A][T2] == 1:
                    gate1 = circuit1.gates[T2, (A,)]
                    U = np.dot(U, np.kron(gate1.U, np.eye(2)))
                    jammed_gates[T2, (A,)] = gate1
                    break
            for T2 in range(T-1,-1,-1):
                if plan[B][T2] == 2 or plan[B][T2] == -2: break
                if plan[B][T2] == 1:
                    gate1 = circuit1.gates[T2, (B,)]
                    U = np.dot(U, np.kron(np.eye(2), gate1.U))
                    jammed_gates[T2, (B,)] = gate1
                    break
            # Right-side 1-body gates (at circuit end)
            if T+1 < self.ntime and max(abs(plan[A][T2]) for T2 in range(T+1, self.ntime)) == 1:
                T2 = [T3 for T3, P in enumerate(plan[A][T+1:self.ntime]) if P == 1][0] + T+1
                gate1 = circuit1.gates[T2, (A,)]
                U = np.dot(np.kron(gate1.U, np.eye(2)), U)
                jammed_gates[T2, (A,)] = gate1
            if T+1 < self.ntime and max(abs(plan[B][T2]) for T2 in range(T+1, self.ntime)) == 1:
                T2 = [T3 for T3, P in enumerate(plan[B][T+1:self.ntime]) if P == 1][0] + T+1
                gate1 = circuit1.gates[T2, (B,)]
                U = np.dot(np.kron(np.eye(2), gate1.U), U)
                jammed_gates[T2, (B,)] = gate1
            circuit2.add_gate(time=T, qubits=qubits, gate=Gate.U2(U=U))
        # Unjammed gates (should all be 1-body on 1-body wires) 
        for key, gate in circuit1.gates.items():
            if gate.N != 1: continue
            T, qubits = key
            if key not in jammed_gates:
                circuit2.add_gate(time=T, qubits=qubits, gate=gate)

        # Jam 2-body gates, if possible
        circuit1 = circuit2
        circuit2 = Circuit(N=self.N)
        jammed_gates = {}
        for T in range(circuit1.ntime):
            circuit3 = circuit1.subset([T])
            for key, gate in circuit3.gates.items():
                if gate.N != 2: continue
                T4, qubits = key
                if (T, qubits) in jammed_gates: continue
                A, B = qubits
                jams = [((T, qubits), gate, False)]
                for T2 in range(T+1, self.ntime):
                    if (T2, (A, B)) in circuit1.gates:
                        jams.append(((T2, (A, B)), circuit1.gates[(T2, (A, B))], False))
                    elif (T2, (B, A)) in circuit1.gates:
                        jams.append(((T2, (B, A)), circuit1.gates[(T2, (B, A))], True))
                    elif (T2, A) in circuit1.TAs:
                        break # Interference
                    elif (T2, B) in circuit1.TAs:
                        break # Interference
                U = np.copy(jams[0][1].U)
                for idx in range(1, len(jams)):
                    key, gate, trans = jams[idx]
                    U2 = np.copy(gate.U)
                    if trans:
                        U2 = np.reshape(np.einsum('ijkl->jilk', np.reshape(U2, (2,)*4)), (4,)*2)
                    U = np.dot(U2,U)
                circuit2.add_gate(time=T, qubits=(A,B), gate=Gate.U2(U=U))
                for key, gate, trans in jams:
                    jammed_gates[key] = gate
        # Unjammed gates (should all be 1-body on 1-body wires)
        for key, gate in circuit1.gates.items():
            if gate.N != 1: continue
            T, qubits = key
            if key not in jammed_gates:
                circuit2.add_gate(time=T, qubits=qubits, gate=gate)

        return circuit2.nonredundant()

    def subcircuit(
        self,
        qubits,
        times, 
        copy=True,
        ):

        """ Return a circuit which is a subset of self in both qubits and time
            (a mixture of deadjoin and subset).

        Params:
            qubits (list of int) - ordered qubit indices to slice in spatial
                indices into the [0,1,2...] indices in the returned circuit.
            times (list of int) - ordered time moments to slice into time moments
                [0,1,2,...] in the returned circuit.
            copy (bool) - copy Gate elements to remove parameter dependencies
                between self and returned circuit (True - default) or not
                (False). 
        Returns:
            (Circuit) - the new qubit- and time-sliced circuit.
        """

        return self.subset(times=times, copy=copy).deadjoin(qubits=qubits, copy=copy)

    def add_circuit(
        self,
        circuit,
        qubits,
        times=None,
        time=None,
        time_placement='early', 
        copy=True,
        ):

        """ Add another circuit to self at specified qubits and times, updating
            self. Essentially a composite version of add_gate. The qubits to
            add circuit to are always explicitly specified. The times to add
            circuit to may be explicitly specified in the times argumet (1st
            priority), the starting time moment may be explicitly specified and
            then the circuit added in a time-contiguous manner from that point
            using the time argument (2nd priority), or a recipe for determining
            the time-contiguous placement can be specified using the
            time_placement argument (3rd priority).

        Params:
            circuit (Circuit) - the circuit to add into self. 
            qubits (tuple of int) - ordered qubit indices in self to add the
                qubit indices of circuit into.
            times (tuple of int) - ordered time moments in self to add the time
                moments of circuit into. If None, the time argument will be
                considered next.
            time (int) - starting time moment in self to add the time moments
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

        if times is None:
            if time is not None:
                times = list(range(time,time+circuit.ntime))
            else:
                if time_placement == 'early':
                    leads = [circuit.ntime for _ in range(len(qubits))]
                    for time2, A in circuit.TAs:
                        leads[A] = min(leads[A], time2)
                    timemax = -1
                    for time2, A in self.TAs:
                        if A in qubits:
                            timemax = max(timemax, time2 - leads[qubits.index(A)])
                    timemax += 1
                    times = list(range(timemax, timemax+circuit.ntime))
                elif time_placement == 'late':
                    timemax = max(self.ntime - 1, 0)
                    if any((timemax, A) in self.TAs for A in qubits):
                        timemax += 1 
                    times = list(range(timemax, timemax+circuit.ntime))
                elif time_placement == 'next':
                    times = list(range(self.ntime, self.ntime+circuit.ntime))
                else:
                    raise RuntimeError('Unknown time_placement: %s. Allowed values are early, late, next' % time_placement)

        if len(qubits) != circuit.N: raise RuntimeError('len(qubits) != circuit.N')
        if len(times) != circuit.ntime: raise RuntimeError('len(times) != circuit.ntime')

        for key, gate in circuit.gates.items():
            time2, qubits2 = key
            time3 = times[time2]
            qubits3 = tuple(qubits[_] for _ in qubits2)
            self.add_gate(time=time3, qubits=qubits3, gate=gate, copy=copy)
        
        return self
             
    # > Gate Addition Sugar < #

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

    def U1(
        self,
        qubitA,
        qubitB,
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
            qubits=(qubitA, qubitB),
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

    # > Parameter Access/Manipulation < #

    @property
    def nparam(self):
        """ The total number of mutable parameters in the circuit. """
        return len(self.param_keys)

    @property
    def param_keys(self):
        """ A list of (time, qubits, param_name) for all mutable parameters in the circuit.

        A global order of (time, qubits, param_name within gate) is used to guarantee
        a stable, lexical ordering of circuit parameters for a given circuit.

        Returns:
            list of (int, tuple of int, str)) - ordered time moments, qubit
                indices, and gate parameter names for all mutable parameters in
                the circuit.
        """
        keys = []
        for key, gate in self.gates.items():
            time, qubits = key
            for name, v in gate.params.items():
                keys.append((time, qubits, name))
        keys.sort(key = lambda x : (x[0], x[1]))
        return keys
        
    @property
    def param_values(self):
        """ A list of param values corresponding to param_keys for all mutable parameters in the circuit. 

        Returns:
            (list of float) - ordered parameter values with order corresponding
                to param_keys for all mutable parameters in the circuit.
        """
        return [self.gates[(time, qubits)].params[name] for time, qubits, name in self.param_keys]

    def set_param_values(
        self,
        param_values,
        param_indices=None,
        ):

        """ Set the param values corresponding to param_keys for all mutable parameters in the circuit.

        Params:
            param_values (list of float) - ordered parameter values with order
                corresponding to param_keys for all mutable parameters in the
                circuit.
            param_indices (list of int or None) - indices of parameter values
                or None. If None, all parameter indices are set.
        Result:
            Parameters of self.gates are updated with new parameter values.
        Returns:
            self - for chaining
        """

        param_keys = self.param_keys

        if param_indices is None:
            param_indices = list(range(len(param_keys)))
    
        for param_index, param_value in zip(param_indices, param_values):
            time, qubits, name = param_keys[param_index]
            self.gates[(time, qubits)].set_param(key=name, param=param_value)

        return self
    
    @property
    def params(self):
        """ An OrderedDict of (time, qubits, param_name) : param_value for all mutable parameters in the circuit. 
            The order follows that of param_keys.

        Returns:
            (OrderedDict of (int, tuple of int, str) : float) - ordered key :
                value pairs for all mutable parameters in the circuit.
        """ 
        return collections.OrderedDict([(k, v) for k, v in zip(self.param_keys, self.param_values)])

    def set_params(
        self,
        params,
        ):

        """ Set an arbitrary number of circuit parameters values by key, value specification.
    
        Params:
            params (OrderedDict of (int, tuple of int, str) : float) - key :
                value pairs for mutable parameters to set.

        Result:
            Parameters of self.gates are updated with new parameter values.
        Returns:
            self - for chaining
        """
    
        for k, v in params.items():
            time, qubits, name = k
            self.gates[(time, qubits)].set_param(key=name, param=v)

        return self

    @property
    def param_str(self):
        """ A human-readable string describing the circuit coordinates,
            parameter names, gate names, and values of all mutable parameters in
            this circuit.
        
        Returns:
            (str) - human-readable string describing parameters in order
                specified by param_keys.
        """ 
        s = ''
        s += '%-5s %-5s %-10s %-10s %-10s: %24s\n' % ('Index', 'Time', 'Qubits', 'Name', 'Gate', 'Value')
        I = 0
        for k, v in self.params.items():
            time, qubits, name = k
            gate = self.gates[(time, qubits)]
            s += '%-5d %-5d %-10s %-10s %-10s: %24.16E\n' % (I, time, qubits, name, gate.name, v)
            I += 1
        return s

    # > ASCII Circuit Diagrams < #

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
        Wd = max(len(str(_)) for _ in range(self.N))
        lstick = '%-*s : |\n' % (2+Wd, 'T')
        for x in range(self.N): 
            lstick += '%*s\n' % (6+Wd, ' ')
            lstick += '|%*d> : -\n' % (Wd, x)

        # Build moment strings
        moments = []
        for T in range(self.ntime):
            moments.append(self.ascii_diagram_moment(
                T=T,
                adjust_for_T=False if time_lines=='neither' else True,
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

    def ascii_diagram_moment(
        self,
        T,
        adjust_for_T=True,
        ):

        """ Return an ASCII string diagram for a given time moment T.

        Users should not generally call this utility routine - see
        ascii_diagram instead.

        Params:
            T (int) - time moment to diagram
            adjust_for_T (bool) - add space adjustments for the length of time
                lines.
        Returns:
            (str) - ASCII diagram for the given time moment.
        """

        circuit = self.subset([T])

        # list (total seconds) of dict of A -> gate symbol
        seconds = [{}]
        # list (total seconds) of dict of A -> interstitial symbol
        seconds2 = [{}]
        for key, gate in circuit.gates.items():
            T2, qubits = key
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
                    if gate.N == 1:
                        seconds[idx][A] = gate.ascii_symbols[0]
                    elif gate.N == 2:
                        Aind = [Aind for Aind, B in enumerate(qubits) if A == B][0]
                        seconds[idx][A] = gate.ascii_symbols[Aind]
                    else:
                        raise RuntimeError('Unknown N>2 gate')
                else:
                    seconds[idx][A] = '|'
                # Gate connector
                if A != min(qubits):
                    seconds2[idx][A] = '|'

        # + [1] for the - null character
        wseconds = [max([len(v) for k, v in second.items()] + [1]) for second in seconds]
        wtot = sum(wseconds)    

        # Adjust widths for T field
        Tsymb = '%d' % T
        if adjust_for_T:
            if wtot < len(Tsymb): wseconds[0] += len(Tsymb) - wtot
            wtot = sum(wseconds)    
        
        Is = ['' for A in range(self.N)]
        Qs = ['' for A in range(self.N)]
        for second, second2, wsecond in zip(seconds, seconds2, wseconds):
            for A in range(self.N):
                Isymb = second2.get(A, ' ')
                IwR = wsecond - len(Isymb)
                Is[A] += Isymb + ' ' * IwR + ' '
                Qsymb = second.get(A, '-')
                QwR = wsecond - len(Qsymb)
                Qs[A] += Qsymb + '-' * QwR + '-'

        strval = Tsymb + ' ' * (wtot + len(wseconds) - len(Tsymb) - 1) + '|\n' 
        for I, Q in zip(Is, Qs):
            strval += I + '\n'
            strval += Q + '\n'

        return strval

    def latex_diagram(
        self,
        row_params='@R=1.0em',
        col_params='@C=1.0em',
        size_params='',
        use_lstick=True,
        ):

        """ Returns a LaTeX Qcircuit diagram specification as an ASCII string. 

        Params:
            row_params (str) - Qcircuit row layout specification
            col_params (str) - Qcircuit col layout specification
            size_params (str) - Qcircuit size layout specification
            use_lstick (bool) - put lstick kets in (True) or not (False)
        Returns:    
            (str) - LaTeX Qcircuit diagram specification as an ASCII string.
        """

        strval = ''

        # Header
        strval += '\\Qcircuit %s %s %s {\n' % (
            row_params,
            col_params,
            size_params,
            )

        # Qubit lines
        lines = ['' for _ in range(self.N)]

        # Lstick  
        if use_lstick:
            for A in range(self.N):
                lines[A] += '\\lstick{|%d\\rangle}\n' % A

        # Moment contents
        for T in range(self.ntime):
            lines2 = self.latex_diagram_moment(
                T=T,    
                )
            for A in range(self.N):
                lines[A] += lines2[A]
        
        # Trailing wires
        for A in range(self.N):
            lines[A] += ' & \\qw \\\\\n'

        # Concatenation
        strval += ''.join(lines)

        # Footer
        strval += '}\n'

        return strval

    def latex_diagram_moment(
        self,
        T,
        ):

        """ Return a LaTeX Qcircuit diagram for a given time moment T.

        Users should not generally call this utility routine - see
        latex_diagram instead.

        Params:
            T (int) - time moment to diagram
        Returns:
            (str) - LaTeX Qcircuit diagram for the given time moment.
        """

        circuit = self.subset([T])

        # list (total seconds) of dict of A -> gate symbol
        seconds = [{}]
        for key, gate in circuit.gates.items():
            T2, qubits = key
            # Find the first second this gate fits within (or add a new one)
            for idx, second in enumerate(seconds):
                fit = not any(A in second for A in range(min(qubits), max(qubits)+1))
                if fit:
                    break
            if not fit:
                seconds.append({})
                idx += 1
            # Place gate lines in circuit
            if gate.N == 1:
                A, = qubits
                seconds[idx][A] = ' & \\gate{%s}\n' % gate.ascii_symbols[0]
            elif gate.N == 2:
                A, B = qubits
                # Special cases
                if gate.name == 'CNOT' or gate.name == 'CX':
                    seconds[idx][A] = ' & \\ctrl{%d}\n' % (B-A) 
                    seconds[idx][B] = ' & \\targ\n'
                elif gate.name == 'CZ':
                    seconds[idx][A] = ' & \\ctrl{%d}\n' % (B-A) 
                    seconds[idx][B] = ' & \\gate{Z}\n'
                elif gate.name == 'SWAP':
                    seconds[idx][A] = ' & \\qswap \\qwx[%d]\n' % (B-A) 
                    seconds[idx][B] = ' & \\qswap\n'
                # General case
                else:
                    seconds[idx][A] = ' & \\gate{%s} \\qwx[%d]\n' % (gate.ascii_symbols[0], (B-A))
                    seconds[idx][B] = ' & \\gate{%s}\n' % gate.ascii_symbols[1]
            else:
                raise RuntimeError('Unknown N>2 body gate: %s' % gate)

        Qs = ['' for A in range(self.N)]
        for second in seconds:
            for A in range(self.N):
                Qs[A] += second.get(A, ' & \\qw \n')

        return Qs

    # > Simulation! < #

    def simulate(
        self,
        wfn=None,
        dtype=np.complex128,
        ):

        """ Propagate wavefunction wfn through this circuit. 

        Params:
            wfn (np.ndarray of shape (2**self.N,) or None)
                - the initial wavefunction. If None, the reference state
                  \prod_{A} |0_A> will be used.
            dtype (real or complex dtype) - the dtype to perform the
                computation at. The input wfn and all gate unitary operators
                will be cast to this type and the returned wfn will be of this
                dtype. Note that using real dtypes (float64 or float32) can
                reduce storage and runtime, but the imaginary parts of the
                input wfn and all gate unitary operators will be discarded
                without checking. In these cases, the user is responsible for
                ensuring that the circuit works on O(2^N) rather than U(2^N)
                and that the output is valid.
        Returns:
            (np.ndarray of shape (2**self.N,) and dtype=dtype) - the
                propagated wavefunction. Note that the input wfn is not
                changed by this operation.
        """

        for time, wfn in self.simulate_steps(wfn, dtype=dtype):
            pass

        return wfn

    def simulate_steps(
        self,
        wfn=None,
        dtype=np.complex128,
        ):

        """ Generator to propagate wavefunction wfn through the circuit one
            moment at a time.

        This is often used as:
        
        for time, wfn1 in simulate_steps(wfn=wfn0):
            print wfn1

        Note that to prevent repeated allocations of (2**N) arrays, this
        operation allocates two (2**N) working copies, and swaps between them
        as gates are applied. References to one of these arrays are returned at
        each moment. Therefore, if you want to save a history of the
        wavefunction, you will need to copy the wavefunction returned at each
        moment by this generator. Note that the input wfn is not changed by
        this operation.

        Params:
            wfn (np.ndarray of shape (2**self.N,) or None)
                - the initial wavefunction. If None, the reference state
                  \prod_{A} |0_A> will be used.
            dtype (real or complex dtype) - the dtype to perform the
                computation at. The input wfn and all gate unitary operators
                will be cast to this type and the returned wfn will be of this
                dtype. Note that using real dtypes (float64 or float32) can
                reduce storage and runtime, but the imaginary parts of the
                input wfn and all gate unitary operators will be discarded
                without checking. In these cases, the user is responsible for
                ensuring that the circuit works on O(2^N) rather than U(2^N)
                and that the output is valid.
        Returns (at each yield):
            (int, np.ndarray of shape (2**self.N,) and dtype=dtype) - the
                time moment and current state of the wavefunction at each step
                along the propagation. Note that the input wfn is not
                changed by this operation.
        """

        # Reference state \prod_A |0_A>
        if wfn is None:
            wfn = np.zeros((2**self.N,), dtype=dtype)
            wfn[0] = 1.0
        else:
            wfn = np.array(wfn, dtype=dtype)

        # Don't modify user data, but don't copy all the time
        wfn1 = np.copy(wfn)
        wfn2 = np.zeros_like(wfn1)

        for time in range(self.ntime):
            circuit = self.subset([time])
            for key, gate in circuit.gates.items():
                time2, qubits = key
                if gate.N == 1:
                    wfn2 = Circuit.apply_gate_1(
                        wfn1=wfn1,
                        wfn2=wfn2,
                        U=np.array(gate.U, dtype=dtype),
                        A=qubits[0],
                        )
                elif gate.N == 2:
                    wfn2 = Circuit.apply_gate_2(
                        wfn1=wfn1,
                        wfn2=wfn2,
                        U=np.array(gate.U, dtype=dtype),
                        A=qubits[0],
                        B=qubits[1],
                        )
                else:
                    raise RuntimeError('Cannot apply gates with N > 2: %s' % gate)
                wfn1, wfn2 = wfn2, wfn1
            yield time, wfn1

    @staticmethod
    def apply_gate_1(
        wfn1,
        wfn2,
        U,
        A,
        ):

        """ Apply a 1-body gate unitary U to wfn1 at qubit A, yielding wfn2.

        The formal operation performed is,

            wfn1_LiR = \sum_{j} U_ij wfn2_LjR

        Here L are the indices of all of the qubits to the left of A (<A), and
        R are the indices of all of the qubits to the right of A (>A).

        This function requires the user to supply both the initial state in
        wfn1 and an array wfn2 to place the result into. This allows this
        function to apply the gate without any new allocations or scratch arrays.

        Params:
            wfn1 (np.ndarray of shape (2**self.N,) and a complex dtype)
                - the initial wavefunction. Unaffected by the operation
            wfn2 (np.ndarray of shape (2**self.N,) and a complex dtype)
                - an array to write the new wavefunction into. Overwritten by
                the operation.
            U (np.ndarray of shape (2,2) and a complex dtype) - the matrix
                representation of the 1-body gate.
            A (int) - the qubit index to apply the gate at.
        Result:
            the data of wfn2 is overwritten with the result of the operation.
        Returns:
            reference to wfn2, for chaining
        """

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if U.shape != (2,2): raise RuntimeError('1-body gate must be (2,2)')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))

        L = 2**(A)     # Left hangover
        R = 2**(N-A-1) # Right hangover
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = (L,2,R)
        wfn2v.shape = (L,2,R)
        np.einsum('LjR,ij->LiR', wfn1v, U, out=wfn2v)

        return wfn2

    @staticmethod
    def apply_gate_2(
        wfn1,
        wfn2,
        U,
        A,
        B,
        ):

        """ Apply a 2-body gate unitary U to wfn1 at qubits A and B, yielding wfn2.

        The formal operation performed is (for the case that A < B),

            wfn1_LiMjR = \sum_{lk} U_ijkl wfn2_LiMjR

        Here L are the indices of all of the qubits to the left of A (<A), M M
        are the indices of all of the qubits to the right of A (>A) and left of
        B (<B), and R are the indices of all of the qubits to the right of B
        (>B). If A > B, permutations of A and B and the gate matrix U are
        performed to ensure that the gate is applied correctly.

        This function requires the user to supply both the initial state in
        wfn1 and an array wfn2 to place the result into. This allows this
        function to apply the gate without any new allocations or scratch arrays.

        Params:
            wfn1 (np.ndarray of shape (2**self.N,) and a complex dtype)
                - the initial wavefunction. Unaffected by the operation
            wfn2 (np.ndarray of shape (2**self.N,) and a complex dtype)
                - an array to write the new wavefunction into. Overwritten by
                the operation.
            U (np.ndarray of shape (4,4) and a complex dtype) - the matrix
                representation of the 1-body gate. This should be packed to
                operate on the product state |A> otimes |B>, as usual.
            A (int) - the first qubit index to apply the gate at.
            B (int) - the second qubit index to apply the gate at.
        Result:
            the data of wfn2 is overwritten with the result of the operation.
        Returns:
            reference to wfn2, for chaining
        """

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if B >= N: raise RuntimeError('B >= N')
        if A == B: raise RuntimeError('A == B')
        if U.shape != (4,4): raise RuntimeError('2-body gate must be (4,4)')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))

        U2 = np.reshape(U, (2,2,2,2))
        if A > B:
            A2, B2 = B, A
            U2 = np.einsum('ijkl->jilk', U2)
        else:
            A2, B2 = A, B

        L = 2**(A2)      # Left hangover
        M = 2**(B2-A2-1) # Middle hangover
        R = 2**(N-B2-1)  # Right hangover
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = (L,2,M,2,R)
        wfn2v.shape = (L,2,M,2,R)
        np.einsum('LkMlR,ijkl->LiMjR', wfn1v, U2, out=wfn2v)

        return wfn2

    @staticmethod
    def compute_1pdm(
        wfn1,
        wfn2,
        A,
        ):

        """ Compute the 1pdm (one-particle density matrix) at qubit A. 

        The 1pdm is formally defined as,

            D_ij = \sum_{L,R} wfn1_LiR^* wfn2_LjR
        
        Here L are the indices of all of the qubits to the left of A (<A), and
        R are the indices of all of the qubits to the right of A (>A).

        If wfn1 is equivalent to wfn2, a Hermitian density matrix will be
        returned. If wfn1 is not equivalent to wfn2, a non-Hermitian transition
        density matrix will be returned (the latter cannot be directly observed
        in a quantum computer, but is a very useful conceptual quantity).

        Params:
            wfn1 (np.ndarray of shape (self.N**2,) and a complex dtype) - the bra wavefunction.
            wfn2 (np.ndarray of shape (self.N**2,) and a complex dtype) - the ket wavefunction.
            A (int) - the index of the qubit to evaluate the 1pdm at
        Returns:
            (np.ndarray of shape (2,2) and complex dtype) - the 1pdm
        """

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))

        L = 2**(A)     # Left hangover
        R = 2**(N-A-1) # Right hangover
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = (L,2,R)
        wfn2v.shape = (L,2,R)
        D = np.einsum('LiR,LjR->ij', wfn1v.conj(), wfn2v)
        return D

    @staticmethod
    def compute_2pdm(
        wfn1,
        wfn2,
        A,
        B,
        ):

        """ Compute the 2pdm (two-particle density matrix) at qubits A and B. 

        The formal operation performed is (for the case that A < B),

            D_ijkl = \sum_{LMR} wfn1_LiMjR^* wfn2_LkMlR

        Here L are the indices of all of the qubits to the left of A (<A), M M
        are the indices of all of the qubits to the right of A (>A) and left of
        B (<B), and R are the indices of all of the qubits to the right of B
        (>B). If A > B, permutations of A and B and the resultant 2pdm are
        performed to ensure that the 2pdm is computed correctly.

        If wfn1 is equivalent to wfn2, a Hermitian density matrix will be
        returned. If wfn1 is not equivalent to wfn2, a non-Hermitian transition
        density matrix will be returned (the latter cannot be directly observed
        in a quantum computer, but is a very useful conceptual quantity).

        Params:
            wfn1 (np.ndarray of shape (self.N**2,) and a complex dtype) - the bra wavefunction.
            wfn2 (np.ndarray of shape (self.N**2,) and a complex dtype) - the ket wavefunction.
            A (int) - the index of the first qubit to evaluate the 2pdm at
            B (int) - the index of the second qubit to evaluate the 2pdm at
        Returns:
            (np.ndarray of shape (4,4) and complex dtype) - the 2pdm in the 
                |A> otimes |B> basis.
        """

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if B >= N: raise RuntimeError('B >= N')
        if A == B: raise RuntimeError('A == B')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))

        if A > B:
            A2, B2 = B, A
        else:
            A2, B2 = A, B

        L = 2**(A2)      # Left hangover
        M = 2**(B2-A2-1) # Middle hangover
        R = 2**(N-B2-1)  # Right hangover
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = (L,2,M,2,R)
        wfn2v.shape = (L,2,M,2,R)
        D = np.einsum('LiMjR,LkMlR->ijkl', wfn1v.conj(), wfn2v)
    
        if A > B:
            D = np.einsum('ijkl->jilk', D)

        return np.reshape(D, (4,4))

    @staticmethod
    def compute_3pdm(
        wfn1,
        wfn2,
        A,
        B,
        C,
        ):

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if B >= N: raise RuntimeError('B >= N')
        if C >= N: raise RuntimeError('C >= N')
        if A == B: raise RuntimeError('A == B')
        if A == C: raise RuntimeError('A == C')
        if B == C: raise RuntimeError('B == C')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))
    
        A2, B2, C2 = sorted((A, B, C))

        L = 2**(A2)      # Left hangover
        M = 2**(B2-A2-1) # Middle1 hangover
        P = 2**(C2-B2-1) # Middle2 hangover
        R = 2**(N-C2-1)  # Right hangover
        
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = (L,2,M,2,P,2,R)
        wfn2v.shape = (L,2,M,2,P,2,R)

        D = np.einsum('LiMjPkR,LlMmPnR->ijklmn', wfn1v.conj(), wfn2v)

        bra_indices = 'ijk'
        ket_indices = 'lmn'
        bra_indices2 = ''.join([bra_indices[(A, B, C).index(_)] for _ in (A2, B2, C2)])
        ket_indices2 = ''.join([ket_indices[(A, B, C).index(_)] for _ in (A2, B2, C2)])

        D = np.einsum('%s%s->%s%s' % (bra_indices, ket_indices, bra_indices2, ket_indices2), D)

        return np.reshape(D, (8, 8))

    @staticmethod
    def compute_4pdm(
        wfn1,
        wfn2,
        A,
        B,
        C,
        D,
        ):

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if B >= N: raise RuntimeError('B >= N')
        if C >= N: raise RuntimeError('C >= N')
        if D >= N: raise RuntimeError('D >= N')
        if A == B: raise RuntimeError('A == B')
        if A == C: raise RuntimeError('A == C')
        if A == D: raise RuntimeError('A == D')
        if B == C: raise RuntimeError('B == C')
        if B == D: raise RuntimeError('B == D')
        if C == D: raise RuntimeError('C == D')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))
    
        A2, B2, C2, D2 = sorted((A, B, C, D))

        L = 2**(A2)      # Left hangover
        M = 2**(B2-A2-1) # Middle1 hangover
        P = 2**(C2-B2-1) # Middle2 hangover
        Q = 2**(D2-C2-1) # Middle3 hangover
        R = 2**(N-D2-1)  # Right hangover
        
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = (L,2,M,2,P,2,Q,2,R)
        wfn2v.shape = (L,2,M,2,P,2,Q,2,R)

        Dm = np.einsum('LiMjPkQlR,LmMnPoQpR->ijklmnop', wfn1v.conj(), wfn2v)

        bra_indices = 'ijkl'
        ket_indices = 'mnop'
        bra_indices2 = ''.join([bra_indices[(A, B, C, D).index(_)] for _ in (A2, B2, C2, D2)])
        ket_indices2 = ''.join([ket_indices[(A, B, C, D).index(_)] for _ in (A2, B2, C2, D2)])

        Dm = np.einsum('%s%s->%s%s' % (bra_indices, ket_indices, bra_indices2, ket_indices2), Dm)

        return np.reshape(Dm, (16, 16))
        
    @staticmethod
    def compute_npdm(
        wfn1,
        wfn2,
        qubits,
        ):

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if any(_ >= N for _ in qubits): raise RuntimeError('qubits >= N')
        if len(set(qubits)) != len(qubits): raise RuntimeError('duplicate entry in qubits')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))
    
        qubits2 = tuple(sorted(qubits))
        hangovers = (2**qubits2[0],) + tuple(2**(qubits2[A+1]-qubits2[A]-1) for A in range(len(qubits2)-1)) + (2**(N-qubits2[-1]-1),)
        shape = []
        for hangover in hangovers[:-1]:
            shape.append(hangover)
            shape.append(2)
        shape.append(hangovers[-1])
        shape = tuple(shape)
        
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = shape
        wfn2v.shape = shape

        hangover_stock = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        bra_stock = 'abcdefghijklm'
        ket_stock = 'nopqrstuvwxyz'
    
        M = len(qubits)
        if M > 13: raise RuntimeError('Technical limit: cannot run N > 13')
    
        bra_str = ''
        ket_str = ''
        for A in range(M):
            bra_str += hangover_stock[A]
            bra_str += bra_stock[A]
            ket_str += hangover_stock[A]
            ket_str += ket_stock[A]
        bra_str += hangover_stock[M]
        ket_str += hangover_stock[M]

        den_str = bra_stock[:M] + ket_stock[:M]

        D = np.einsum('%s,%s->%s' % (bra_str, ket_str, den_str), wfn1v.conj(), wfn2v)

        bra_str2 = ''.join([bra_stock[qubits.index(_)] for _ in qubits2])
        ket_str2 = ''.join([ket_stock[qubits.index(_)] for _ in qubits2])
        den_str2 = bra_str2 + ket_str2

        D = np.einsum('%s->%s' % (den_str, den_str2), D)

        return np.reshape(D, (2**M, 2**M))

    @staticmethod
    def compute_pauli_1(
        wfn,
        A,
        ):

        """ Compute the expectation values of the 1-body Pauli operators at qubit A.

        E.g., the expectation value of the Z operator at qubit A is,

            <Z_A> = <wfn|\hat Z_A|wfn>

        These can be efficiently computed from the 1pdm (they are just an
            alternate representation of the 1pdm).

        Params:
            wfn (np.ndarray of shape (self.N**2,) and a complex dtype) - the wavefunction.
            A (int) - the index of the qubit to evaluate the Pauli measurements at.
        Returns:
            (np.ndarray of shape (4,) and real dtype corresponding to precision
                of wfn dtype) - the Pauli expectation values packed as [I,X,Y,Z].
        """

        D = Circuit.compute_1pdm(
            wfn1=wfn,
            wfn2=wfn,
            A=A,
            )

        I = (D[0,0] + D[1,1]).real
        Z = (D[0,0] - D[1,1]).real
        X = (D[1,0] + D[0,1]).real
        Y = (D[1,0] - D[0,1]).imag
        return np.array([I,X,Y,Z])

    @staticmethod
    def compute_pauli_2(
        wfn,
        A,
        B,
        ):

        """ Compute the expectation values of the 2-qubit Pauli operators at
            qubits A and B.

        E.g., the expectation value of the Z operator at qubit A and the X
        operator at qubit B is,

            <Z_A X_B> = <wfn|\hat Z_A \hat X_B|wfn>

        These can be efficiently computed from the 2pdm (they are just an
            alternate representation of the 2pdm).

        Params:
            wfn (np.ndarray of shape (self.N**2,) and a complex dtype) - the wavefunction.
            A (int) - the index of the first qubit to evaluate the Pauli measurements at.
            B (int) - the index of the second qubit to evaluate the Pauli measurements at.
        Returns:
            (np.ndarray of shape (4,4) and real dtype corresponding to precision
                of wfn dtype) - the Pauli expectation values packed as [I,X,Y,Z].
        """

        D = Circuit.compute_2pdm(
            wfn1=wfn,
            wfn2=wfn,
            A=A,
            B=B,
            )

        Pmats = [Matrix.I, Matrix.X, Matrix.Y, Matrix.Z]
        G = np.zeros((4,4))
        for A, PA in enumerate(Pmats):
            for B, PB in enumerate(Pmats):
                G[A,B] = np.sum(np.kron(PA, PB).conj() * D).real

        return G

    @staticmethod
    def compute_pauli_3(
        wfn,
        A,
        B,
        C,
        ):

        """ Compute the expectation values of the 3-qubit Pauli operators at
            qubits A, B, C

        Params:
            wfn (np.ndarray of shape (self.N**2,) and a complex dtype) - the wavefunction.
            A (int) - the index of the first qubit to evaluate the Pauli measurements at.
            B (int) - the index of the second qubit to evaluate the Pauli measurements at.
            C (int) - the index of the third qubit to evaluate the Pauli measurements at.
        Returns:
            (np.ndarray of shape (4,4,4) and real dtype corresponding to precision
                of wfn dtype) - the Pauli expectation values packed as [I,X,Y,Z].
        """

        D = Circuit.compute_3pdm(
            wfn1=wfn,
            wfn2=wfn,
            A=A,
            B=B,
            C=C,
            )

        Pmats = [Matrix.I, Matrix.X, Matrix.Y, Matrix.Z]
        G = np.zeros((4,4,4))
        for A, PA in enumerate(Pmats):
            for B, PB in enumerate(Pmats):
                for C, PC in enumerate(Pmats):
                    G[A,B,C] = np.sum(np.kron(np.kron(PA, PB), PC).conj() * D).real

        return G

    @staticmethod
    def compute_pauli_4(
        wfn,
        A,
        B,
        C,
        D,
        ):

        """ Compute the expectation values of the 4-qubit Pauli operators at
            qubits A, B, C, D

        Params:
            wfn (np.ndarray of shape (self.N**2,) and a complex dtype) - the wavefunction.
            A (int) - the index of the first qubit to evaluate the Pauli measurements at.
            B (int) - the index of the second qubit to evaluate the Pauli measurements at.
            C (int) - the index of the third qubit to evaluate the Pauli measurements at.
            D (int) - the index of the fourth qubit to evaluate the Pauli measurements at.
        Returns:
            (np.ndarray of shape (4,4,4,4) and real dtype corresponding to precision
                of wfn dtype) - the Pauli expectation values packed as [I,X,Y,Z].
        """

        Dm = Circuit.compute_4pdm(
            wfn1=wfn,
            wfn2=wfn,
            A=A,
            B=B,
            C=C,
            D=D,
            )

        Pmats = [Matrix.I, Matrix.X, Matrix.Y, Matrix.Z]
        G = np.zeros((4,4,4,4))
        for A, PA in enumerate(Pmats):
            for B, PB in enumerate(Pmats):
                for C, PC in enumerate(Pmats):
                    for D, PD in enumerate(Pmats):
                        G[A,B,C,D] = np.sum(np.kron(np.kron(np.kron(PA, PB), PC), PD).conj() * Dm).real

        return G

    @staticmethod
    def compute_pauli_n(
        wfn,
        qubits,
        ):

        D = Circuit.compute_npdm(
            wfn1=wfn,
            wfn2=wfn,
            qubits=qubits,
            )

        Pmats = [Matrix.I, Matrix.X, Matrix.Y, Matrix.Z]
        G = np.zeros((4,)*len(qubits))
        for index in itertools.product(range(4), repeat=len(qubits)):
            P = Pmats[index[0]]
            for B in range(1, len(qubits)):
                P = np.kron(P, Pmats[index[B]])
            G[index] = np.sum(P.conj() * D).real

        return G

    def measure(
        self, 
        nmeasurement=1000,
        wfn=None,
        dtype=np.complex128,
        ):

        """ Randomly sample the measurement outputs of the quantum circuit.
            First calls self.simulate to generate the final statevector, then
            calls compute_measurements_from_statevector to perform the random
            sampling. 

        Params:
            nmeasurement (int) - number of measurements to sample.
            wfn (np.ndarray of shape (2**self.N,) or None)
                - the initial wavefunction. If None, the reference state
                  \prod_{A} |0_A> will be used.
            dtype (real or complex dtype) - the dtype to perform the
                computation at. The input wfn and all gate unitary operators
                will be cast to this type and the returned wfn will be of this
                dtype. Note that using real dtypes (float64 or float32) can
                reduce storage and runtime, but the imaginary parts of the
                input wfn and all gate unitary operators will be discarded
                without checking. In these cases, the user is responsible for
                ensuring that the circuit works on O(2^N) rather than U(2^N)
                and that the output is valid.
        Returns:
            (Measurement) - a Measurement object containing the results of
                randomly sampled projective measurements .
        """
    
        return Circuit.compute_measurements_from_statevector(
            self.simulate(wfn=wfn, dtype=dtype), 
            nmeasurement=nmeasurement,
            )

    @staticmethod
    def compute_measurements_from_statevector(
        statevector,
        nmeasurement,
        ):

        """ Randomly sample the measurement outputs obtained by projectively
            measuring the statevector in the computational basis.

        Params:
            statevector (np.ndarray of shape (2**N,)) - the statevector to
                sample.
            nmeasurement (int) - number of measurements to sample.
        Returns:
            (Measurement) - a Measurement object containing the results of
                randomly sampled projective measurements .
        """

        N = (statevector.shape[0]&-statevector.shape[0]).bit_length()-1
        P = (np.conj(statevector) * statevector).real
        I = list(np.searchsorted(np.cumsum(P), np.random.rand(nmeasurement)))
        return Measurement({ Ket.from_int(k, N) : I.count(k) for k in list(sorted(set(I))) }) 
