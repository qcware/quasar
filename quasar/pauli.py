import sortedcontainers
import numpy as np
from .circuit import Matrix
import itertools

class PauliOperator(tuple):

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
        return self[0]

    @property
    def char(self):
        return self[1]

    def __str__(self):
        return '%s%d' % (self.char, self.qubit)

    @staticmethod
    def from_string(string):
        char = string[0]
        qubit = int(string[1:]) 
        return PauliOperator(
            qubit=qubit,
            char=char,
            )

class PauliString(tuple):

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
        return len(self)

    @property
    def qubits(self):
        return tuple([_.qubit for _ in self])

    @property
    def chars(self):
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
        s = 'Pauli:\n'
        s += '  %-10s = %d\n' % ('nqubit', self.nqubit)
        s += '  %-10s = %d\n' % ('nterm', self.nterm)
        s += '  %-10s = %d\n' % ('max_order', self.max_order)
        return s 
    
    # => Attributes <= #

    @property
    def qubits(self):
        # TODO: Might want to dynamically memoize this
        qubits = sortedcontainers.SortedSet()
        for string in self.keys():
            for qubit in string.qubits:
                qubits.add(qubit)
        return qubits

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
    def nterm(self):
        return len(self)

    @property
    def max_order(self):
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

        if not isinstance(other, Pauli): raise TypeError('other must be Pauli')

        return sum(v*other.get(k, 0.0) for k, v in self.items())

    @property
    def conj(self):
        return Pauli(sortedcontainers.SortedDict((k, np.conj(v)) for k, v in self.items()))

    @property
    def norm2(self):
        return np.sqrt(self.dot(self))
    
    @property
    def norminf(self):
        return np.max(np.abs(list(self.values())))

    @staticmethod
    def zero():
        return Pauli(sortedcontainers.SortedDict())

    @staticmethod   
    def zeros_like(x):
        return Pauli(sortedcontainers.SortedDict((k, 0.0) for k, v in x.items()))

    def sieved(self, cutoff=1.0E-14):
        return Pauli(sortedcontainers.SortedDict((k, v) for k, v in self.items() if np.abs(v) > cutoff))

    def real_coefficients(self):
        return Pauli(sortedcontainers.SortedDict((k, v.real) for k, v in self.items()))

    @staticmethod
    def I():
        return Pauli(sortedcontainers.SortedDict([(PauliString.I(), 1.0)]))

    @staticmethod
    def IXYZ():
        return PauliStarter('I'), PauliStarter('X'), PauliStarter('Y'), PauliStarter('Z')

    # > Pauli <-> Computational basis matrix conversion utilities < #

    def to_matrix(
        self,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        ):

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
