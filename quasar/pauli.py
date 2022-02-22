import sortedcontainers
import numpy as np
from .circuit import Matrix
import itertools

class PauliOperator(tuple):
    """ Class PauliOperator represents a one-qubit Pauli
    operator with a given character ('X', 'Y', or 'Z') and
    qubit index.    
    """

    def __new__(
        self,
        qubit,
        char=None,
        ):
        if isinstance(qubit, tuple):
            char = qubit[1]
            qubit = qubit[0]
        if not isinstance(qubit, int): raise RuntimeError('qubit must be int')
        if not isinstance(char, str): raise RuntimeError('char must be str')
        if char not in ['X', 'Y', 'Z']: raise RuntimeError('char must be one of X, Y, or Z')

        return tuple.__new__(PauliOperator, (qubit, char))

    @property
    def qubit(self):
        """int: Return qubit index."""
        return self[0]

    @property
    def char(self):
        """str: Return the character representing the type of Pauli operator ('X','Y', or 'Z')."""
        return self[1]

    def __str__(self):
        return '%s%d' % (self.char, self.qubit)

    @staticmethod
    def from_string(string):
        """ 
        Args:
            string (str): string to convert into a PauliOperator object
        
        Returns:
            PauliOperator: a PauliOperator object from a string composed of
            the character ('X','Y', or 'Z') representing the type of Pauli
            operator and the qubit index. 
        """
        char = string[0]
        qubit = int(string[1:]) 
        return PauliOperator(
            qubit=qubit,
            char=char,
            )

class PauliString(tuple):
    """ Class PauliString represents a string of one or more PauliOperator
    objects as a tuple of PauliOperator objects.
    """

    def __new__(
        self,
        operators,
        ):

        if not isinstance(operators, tuple): raise RuntimeError('operators must be tuple')
        if not all(isinstance(_, PauliOperator) for _ in operators): raise RuntimeError('operators must all be Pauli Operator')
        if len(set(operator.qubit for operator in operators)) != len(operators): raise RuntimeError('operators must all refer to unique qubits')

        return tuple.__new__(PauliString, operators)

    # => Attributes <= #

    @property
    def order(self):
        """int: The number of PauliOperator objects in the PauliString."""
        return len(self)

    @property
    def qubits(self):
        """tuple of ints: The qubit indices occupied by all of the
        PauliOperator objects in the PauliString."""
        return tuple([_.qubit for _ in self])

    @property
    def chars(self):
        """ The characters ('X', 'Y', or 'Z') corresponding to all of
        the PauliOperator objects in the PauliString. 
        """
        return tuple([_.char for _ in self])

    # => String Representations <= #

    def __str__(self):
        if len(self) == 0: return 'I'
        s = ''
        for operator in self[:-1]:
            s += '%s*' % str(operator)
        s += str(self[-1])
        return s

    @staticmethod
    def from_string(string):
        """ Return a PauliString  object from a string expressing a
        linear combination of products of one or more PauliOperator objects. 
 
        Args:
            string (str): string to convert into a PauliOperator object
        
        Returns:
            PauliString: a PauliString  object from a string expressing a
            linear combination of products of one or more PauliOperator objects. 
        """
        if string == 'I': 
            return PauliString(
                operators=tuple(),
                )
        else:
            return PauliString(
                operators=tuple(PauliOperator.from_string(_) for _ in string.split('*')),
                )

    @staticmethod
    def I():
        return PauliString(tuple())

class Pauli(sortedcontainers.SortedDict):
    """ Class Pauli represents Pauli operators as an ``OrderedDict`` of
    ``PauliString : float/complex``pairs. Use ``dict`` access methods to 
    add Pauli strings and access/modify the coefficients of existing strings.
    """

    def __init__(
        self,
        *args,
        **kwargs,
        ):

        super(Pauli, self).__init__(*args, **kwargs)

        for k, v in self.items():
            if not isinstance(k, PauliString): raise RuntimeError('Key must be PauliString: %s' % k) 

    def __contains__(
        self,
        key,
        ):

        if isinstance(key, str): key = PauliString.from_string(key)
        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        return super(Pauli, self).__contains__(key)

    def __getitem__(
        self,
        key,
        ):

        if isinstance(key, str): key = PauliString.from_string(key)
        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        return super(Pauli, self).__getitem__(key)

    def __setitem__(
        self,
        key,
        value,
        ):

        if isinstance(key, str): key = PauliString.from_string(key)
        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        return super(Pauli, self).__setitem__(key, value)

    def get(
        self,
        key,
        default=None,
        ):

        if isinstance(key, str): key = PauliString.from_string(key)
        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        return super(Pauli, self).get(key, default)

    def setdefault(
        self,
        key,
        default=None,
        ):

        if isinstance(key, str): key = PauliString.from_string(key)
        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        return super(Pauli, self).setdefault(key, default)

    def update(self, *args, **kwargs):
        raise RuntimeError('Pauli.update is not a well-defined operation, so we have poisoned this method of dict')
        
    # => String Representations <= #

    def __str__(self):
        lines = []
        for string, value in self.items():
            strval = '%s*%s' % (value, string)
            if strval[0] == '-':
                lines.append(strval)
            else:   
                lines.append('+' + strval)
        return '\n'.join(lines)

    @property
    def summary_str(self):
        """str: A formatted summary string that displays the values of the nqubit, 
        nterm, and max_order attributes of the Pauli object."""
        s = 'Pauli:\n'
        s += '  %-10s = %d\n' % ('nqubit', self.nqubit)
        s += '  %-10s = %d\n' % ('nterm', self.nterm)
        s += '  %-10s = %d\n' % ('max_order', self.max_order)
        return s 
    
    # => Attributes <= #

    @property
    def qubits(self):
        """SortedSet: The unique occupied qubit indices over all strings in the Pauli object."""
        # TODO: Might want to dynamically memoize this
        qubits = sortedcontainers.SortedSet()
        for string in self.keys():
            for qubit in string.qubits:
                qubits.add(qubit)
        return qubits

    @property
    def min_qubit(self):
        """int: The minimum occupied qubit index (or 0 if no occupied qubits) """
        return self.qubits[0] if len(self.qubits) else 0
    
    @property
    def max_qubit(self):
        """int: The maximum occupied qubit index (or -1 if no occupied qubits) """
        return self.qubits[-1] if len(self.qubits) else -1

    @property
    def nqubit(self):
        """int: The total number of qubit indices in the circuit (including empty qubit indices). """
        return self.qubits[-1] - self.qubits[0] + 1 if len(self.qubits) else 0

    @property
    def nqubit_sparse(self):
        """int: The total number of occupied qubit indices in the circuit (excluding empty qubit indices). """
        return len(self.qubits)
    
    @property
    def nterm(self):
        """int: The number of Pauli strings in the Pauli object."""
        return len(self)

    @property
    def max_order(self):
        """int: The maximum number of one-qubit Pauli operators in the Pauli object."""
        return max(_.order for _ in self.keys())

    # => Arithmetic <= #

    def __pos__(self):
        return Pauli(sortedcontainers.SortedDict((k, v) for k, v in self.items()))

    def __neg__(self):
        return Pauli(sortedcontainers.SortedDict((k, -v) for k, v in self.items()))

    def __mul__(self, other):
        
        if isinstance(other, Pauli):

            if self.nterm == 1 and other.nterm == 1:
                value = list(self.values())[0] * list(other.values())[0]
                strings1 = list(self.keys())[0]
                strings2 = list(other.keys())[0]
                qubits1 = strings1.qubits
                qubits2 = strings2.qubits
                operators = []
                for string1 in strings1:
                    if string1.qubit not in qubits2:
                        operators.append(string1)
                    else:
                        # Pauli products on same qubit
                        string2 = strings2[qubits2.index(string1.qubit)]
                        char1 = string1.char
                        char2 = string2.char
                        if char1 == char2:
                            continue # X*X, Y*Y, Z*Z = I
                        elif (char1, char2) == ('X', 'Y'):
                            value *= +1.j
                            operators.append(PauliOperator(qubit=string1.qubit, char='Z'))
                        elif (char1, char2) == ('Y', 'X'):
                            value *= -1.j
                            operators.append(PauliOperator(qubit=string1.qubit, char='Z'))
                        elif (char1, char2) == ('Y', 'Z'):
                            value *= +1.j
                            operators.append(PauliOperator(qubit=string1.qubit, char='X'))
                        elif (char1, char2) == ('Z', 'Y'):
                            value *= -1.j
                            operators.append(PauliOperator(qubit=string1.qubit, char='X'))
                        elif (char1, char2) == ('Z', 'X'):
                            value *= +1.j
                            operators.append(PauliOperator(qubit=string1.qubit, char='Y'))
                        elif (char1, char2) == ('X', 'Z'):
                            value *= -1.j
                            operators.append(PauliOperator(qubit=string1.qubit, char='Y'))
                for string2 in strings2:
                    if string2.qubit not in qubits1:
                        operators.append(string2)
                return Pauli(sortedcontainers.SortedDict([(PauliString(tuple(operators)), value)]))
            else:
                pauli = Pauli(sortedcontainers.SortedDict())
                for k1, v1 in self.items():
                    pauli1 = Pauli(sortedcontainers.SortedDict([(k1, v1)]))
                    for k2, v2 in other.items():
                        pauli2 = Pauli(sortedcontainers.SortedDict([(k2, v2)]))
                        pauli += pauli1 * pauli2 # You see that?!!
                return pauli

        else:
        
            return Pauli(sortedcontainers.SortedDict((k, other*v) for k, v in self.items()))

        return NotImplemented
            
    def __rmul__(self, other):
        
        return Pauli(sortedcontainers.SortedDict((k, other*v) for k, v in self.items()))

    def __truediv__(self, other):
        
        return Pauli(sortedcontainers.SortedDict((k, v/other) for k, v in self.items()))

    def __add__(self, other):

        if isinstance(other, Pauli):

            pauli2 = self.copy()
            for k, v in other.items():
                pauli2[k] = self.get(k, 0.0) + v
            return pauli2

        else:

            pauli2 = self.copy()
            pauli2[PauliString.I()] = self.get(PauliString.I(), 0.0) + other
            return pauli2 

        return NotImplemented

    def __sub__(self, other):

        if isinstance(other, Pauli):

            pauli2 = self.copy()
            for k, v in other.items():
                pauli2[k] = self.get(k, 0.0) - v
            return pauli2

        else:

            pauli2 = self.copy()
            pauli2[PauliString.I()] = self.get(PauliString.I(), 0.0) - other
            return pauli2 

        return NotImplemented

    def __radd__(self, other):
    
        pauli2 = self.copy()
        pauli2[PauliString.I()] = pauli2.get(PauliString.I(), 0.0) + other
        return pauli2 

    def __rsub__(self, other):
    
        pauli2 = -self
        pauli2[PauliString.I()] = pauli2.get(PauliString.I(), 0.0) + other
        return pauli2 

    def __iadd__(self, other):

        if isinstance(other, Pauli):

            for k, v in other.items():
                self[k] = self.get(k, 0.0) + v
            return self

        else:

            self[PauliString.I()] = self.get(PauliString.I(), 0.0) + other
            return self

        return NotImplemented

    def __isub__(self, other):

        if isinstance(other, Pauli):
    
            for k, v in other.items():
                self[k] = self.get(k, 0.0) - v
            return self

        else:
        
            self[PauliString.I()] = self.get(PauliString.I(), 0.0) - other
            return self

        return NotImplemented

    def dot(self, other):
        """ Calculate the dot product of coincident Pauli strings in ``self``
        and another Pauli object.

        Args:
            other (Pauli): the other Pauli object in the dot product with ``self``.
        
        Returns:
            float: the dot product of coincident Pauli strings in ``self`` and **other**.

        """
        if not isinstance(other, Pauli): raise TypeError('other must be Pauli')

        return sum(v*other.get(k, 0.0) for k, v in self.items())

    @property
    def conj(self):
        """Pauli: Return a new version of the Pauli object with the values conjugated."""
        return Pauli(sortedcontainers.SortedDict((k, np.conj(v)) for k, v in self.items()))

    @property
    def norm2(self):
        """float: Return the square root of the dot product of ``self`` with ``self``."""
        return np.sqrt(self.dot(self))
    
    @property
    def norminf(self):
        """float: Return the magnitude of the largest coefficient of ``self``."""
        return np.max(np.abs(list(self.values())))

    @staticmethod
    def zero():
        """ 
        Returns:
            Pauli: a new Pauli object initialized with no strings.

        """
        return Pauli(sortedcontainers.SortedDict())

    @staticmethod   
    def zeros_like(x):
        """ Returns a new Pauli object initialized with the
        strings of another Pauli object, but with coefficient
        values initialized to zero.

        Args:
            x (Pauli): Pauli object whose strings will be used to
                initialize the new Pauli object.
        
        Returns:
            Pauli: a Pauli object initialized with the
            strings of **x**, but with coefficient values
            initialized to zero.

        """
        return Pauli(sortedcontainers.SortedDict((k, 0.0) for k, v in x.items()))

    def sieved(self, cutoff=1.0E-14):
        """ Remove strings which have small or zero coefficients from the ``self``.
        
        Args:
            cutoff (float): remove a Pauli string if the absolute
                value of its coefficient is less than **cutoff** (default - `1.0E-14).
        
        Returns: 
            Pauli: a new Pauli object with strings removed from ``self`` if the absolute 
            value of their coefficient is less than **cutoff**.

        """
        return Pauli(sortedcontainers.SortedDict((k, v) for k, v in self.items() if np.abs(v) > cutoff))

    def real_coefficients(self):
        return Pauli(sortedcontainers.SortedDict((k, v.real) for k, v in self.items()))

    @staticmethod
    def I():
        return Pauli(sortedcontainers.SortedDict([(PauliString.I(), 1.0)]))

    @staticmethod
    def IXYZ():
        """ Returns four PauliStarter objects corresponding to the I, X,
        Y, and Z Pauli operators respectively. 

        When PauliStarter objects are indexed by the ``[]`` operator, they
        generate a corresponding Pauli object on the given qubit index. For
        instance, after generating PauliStarter objects using ``I, X, Y, Z = quasar.Pauli.IXYZ()``,
        ``X[1]`` generates an ``X`` Pauli operator object on qubit index `1`.
        Since indices applied to ``I`` PauliStarter objects have no effect, 
        it is convention to refer to the ``I`` operator using ``I[-`]``. Use
        ``+``, ``-``, ``*``, and ``/``, as well as float and complex
        coefficients to build linear combinations of products of Pauli operators.

        Returns:
            PauliStarter: I, X, Y, Z PauliStarter objects, returned in that order

        """
        return PauliStarter('I'), PauliStarter('X'), PauliStarter('Y'), PauliStarter('Z')

    # > Pauli <-> Computational basis matrix conversion utilities < #

    def to_matrix(
        self,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ):
        """ Convert a Pauli object to the computational-basis Hilbert space. 

        Args:
            min_qubit (int): the minimum occupied qubit index (default - `None`).
            nqubit (int): the total number of qubit indices in the circuit, including
                empty qubit indices (default - `None`).
            dtype (complex dtype): the dtype to perform the computation at (default - `None`).

        Returns:
            ``np.ndarray``, complex dtype: ``self`` expressed in the computational-basis Hilbert space.

        """
        min_qubit = self.min_qubit if min_qubit is None else min_qubit
        nqubit = self.nqubit if nqubit is None else nqubit
    
        matrix = np.zeros((2**nqubit,)*2, dtype=np.complex128)

        for string, value in self.items():
            bra_inds = list(range(2**nqubit))
            factors = np.ones((2**nqubit,), dtype=np.complex128)
            for operator in string:
                qubit, char = operator 
                qubit -= min_qubit
                test = 1 << (nqubit - qubit - 1)
                if char == 'Z':
                    for I in range(2**nqubit):
                        if I & test: factors[I] *= -1.0
                elif char == 'X':
                    for I in range(2**nqubit):
                        bra_inds[I] ^= test
                elif char == 'Y':
                    for I in range(2**nqubit):
                        bra_inds[I] ^= test
                    factors *= 1.j
                    for I in range(2**nqubit):
                        if I & test: factors[I] *= -1.0
                else:
                    raise RuntimeError('Unknown char: %s' % char)
            matrix[bra_inds, range(2**nqubit)] += factors * value

        return np.array(matrix, dtype=dtype)

    @staticmethod
    def from_matrix(
        matrix,
        qubit_indices=None,
        cutoff=1.0E-14,
        ):
        """ Convert a computational-basis Hilbert space operator to an
        equivalent Pauli form.

        Args:
            param_matrix (``np.ndarray``): matrix representing a computational-basis Hilbert space operator.
            qubit_indices (list of ints): qubit indices to be occupied by Pauli operators. If default
                value `None`, all qubit indices are included in **qubit_indices**.
            cutoff (float): remove a Pauli string from the Pauli form if the absolute value
                of its coefficient is less than **cutoff** (default - `1.0E-14`). 

        Returns: 
            Pauli: **matrix** expressed in an equivalent Pauli form.

        """
        if not isinstance(matrix, np.ndarray): 
            raise RuntimeError('operator must be ndarray')

        nqubit = int(np.round(np.log2(matrix.shape[0])))
        if matrix.shape != (2**nqubit,)*2:
            raise RuntimeError('matrix shape must be (2**nqubit, 2**nqubit)')

        if qubit_indices is None:
            qubit_indices = list(range(nqubit))

        pauli = Pauli.zero()
        I, X, Y, Z = Pauli.IXYZ()
        pauli_operators = (
            I,
            X,
            Y,
            Z,
            )
        pauli_vectors = (
            Matrix.I,
            Matrix.X,
            Matrix.Y,
            Matrix.Z,
            )
        for key in itertools.product((0,1,2,3), repeat=nqubit):
            pauli_vector = 1.0
            for index, value in enumerate(key):
                pauli_vector = np.kron(pauli_vector, pauli_vectors[value])
            pauli_coef = np.sum(pauli_vector.conj() * matrix) / (2.0**nqubit)
            # Can save some time 
            if np.abs(pauli_coef) < cutoff: 
                continue
            pauli_operator = 1.0 * I[-1]
            for index, value in enumerate(key):
                qubit_index = qubit_indices[index]
                pauli_operator *= pauli_operators[value][qubit_index]
            pauli += pauli_coef * pauli_operator

        pauli = pauli.sieved(cutoff=cutoff)
                
        return pauli 

    def matrix_vector_product(
        self,
        statevector,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ):
        """ Evaluate how a Pauli object acts on a statevector in
        the computational-basis Hilbert space.

        Args:
            statevector (``np.ndarray`` of shape 2**N): the statevector on which to 
                evaluate how **self** acts in the computational-basis Hilbert space.
            min_qubit (int): the minimum occupied qubit index (default - `None`).
            nqubit (int): the total number of qubit indices in the circuit, including
                empty qubiit indices (default - `None`).
            dtype (real or complex dtype): the dtype to perform the computation at (default - `None`).

        Returns:
            ``np.ndarray``, complex dtype: the result of **self** acting on **statevector** in the
            computational-basis Hilbert space.

        """
        min_qubit = self.min_qubit if min_qubit is None else min_qubit
        nqubit = self.nqubit if nqubit is None else nqubit
    
        if statevector.shape != (2**nqubit,): raise RuntimeError('statevector must be shape (2**nqubit,)')
        sigmavector = np.zeros((2**nqubit,), dtype=np.complex128)

        for string, value in self.items():
            bra_inds = list(range(2**nqubit))
            factors = np.ones((2**nqubit,), dtype=np.complex128)
            for operator in string:
                qubit, char = operator 
                qubit -= min_qubit
                test = 1 << (nqubit - qubit - 1)
                if char == 'Z':
                    for I in range(2**nqubit):
                        if I & test: factors[I] *= -1.0
                elif char == 'X':
                    for I in range(2**nqubit):
                        bra_inds[I] ^= test
                elif char == 'Y':
                    for I in range(2**nqubit):
                        bra_inds[I] ^= test
                    factors *= 1.j
                    for I in range(2**nqubit):
                        if I & test: factors[I] *= -1.0
                else:
                    raise RuntimeError('Unknown char: %s' % char)
            sigmavector[bra_inds] += factors * value * statevector

        return np.array(sigmavector, dtype=dtype)
    
class PauliExpectation(Pauli):
    """ Class PauliExpectation represents the expectation value of a Pauli-sparse
    Hermitian operator.
    """

    def __str__(self):
        lines = []
        for string, value in self.items():
            lines.append('<%s> = %s' % (string, value))
        return '\n'.join(lines)

    @staticmethod
    def zero():
        return PauliExpectation(sortedcontainers.SortedDict())

    @staticmethod   
    def zeros_like(x):
        return PauliExpectation(sortedcontainers.SortedDict((k, 0.0) for k, v in x.items()))

class PauliStarter(object):

    def __init__(
        self,
        char,
        ):

        if char not in ['I', 'X', 'Y', 'Z']: raise RuntimeError('char must be one of I, X, Y, or Z')
        self.char = char

    def __getitem__(self, qubit):
        if self.char == 'I':
            return Pauli.I()
        else:
            return Pauli(sortedcontainers.SortedDict([(PauliString((PauliOperator(qubit=qubit, char=self.char),)), 1.0)]))
