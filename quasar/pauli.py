import collections
import numpy as np

class PauliOperator(tuple):

    def __new__(
        self,
        qubit,
        char,
        ):

        if not isinstance(qubit, int): raise RuntimeError('qubit must be int')
        if qubit < 0: raise RuntimeError('qubit must be positive')
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

PauliString.I = PauliString(tuple())

class Pauli(collections.OrderedDict):

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
        s += '  %-10s = %d\n' % ('N', self.N)
        s += '  %-10s = %d\n' % ('nterm', self.nterm)
        s += '  %-10s = %d\n' % ('max_order', self.max_order)
        return s 
    
    # => Attributes <= #

    @property
    def N(self):
        return max(max(_.qubits) for _ in self.keys() if _.order > 0) + 1
        
    @property
    def nterm(self):
        return len(self)

    @property
    def max_order(self):
        return max(_.order for _ in self.keys())

    # => Arithmetic <= #

    def __pos__(self):
        return Pauli(collections.OrderedDict((k, v) for k, v in self.items()))

    def __neg__(self):
        return Pauli(collections.OrderedDict((k, -v) for k, v in self.items()))

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
                return Pauli(collections.OrderedDict([(PauliString(tuple(operators)), value)]))
            else:
                pauli = Pauli(collections.OrderedDict())
                for k1, v1 in self.items():
                    pauli1 = Pauli(collections.OrderedDict([(k1, v1)]))
                    for k2, v2 in other.items():
                        pauli2 = Pauli(collections.OrderedDict([(k2, v2)]))
                        pauli += pauli1 * pauli2 # You see that?!!
                return pauli

        else:
        
            return Pauli(collections.OrderedDict((k, other*v) for k, v in self.items()))

        return NotImplemented
            
    def __rmul__(self, other):
        
        return Pauli(collections.OrderedDict((k, other*v) for k, v in self.items()))

    def __truediv__(self, other):
        
        return Pauli(collections.OrderedDict((k, v/other) for k, v in self.items()))

    def __add__(self, other):

        if isinstance(other, Pauli):

            pauli2 = self.copy()
            for k, v in other.items():
                pauli2[k] = self.get(k, 0.0) + v
            return pauli2

        else:

            pauli2 = self.copy()
            pauli2[PauliString.I] = self.get(PauliString.I, 0.0) + other
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
            pauli2[PauliString.I] = self.get(PauliString.I, 0.0) - other
            return pauli2 

        return NotImplemented

    def __radd__(self, other):
    
        pauli2 = self.copy()
        pauli2[PauliString.I] = pauli2.get(PauliString.I, 0.0) + other
        return pauli2 

    def __rsub__(self, other):
    
        pauli2 = -self
        pauli2[PauliString.I] = pauli2.get(PauliString.I, 0.0) + other
        return pauli2 

    def __iadd__(self, other):

        if isinstance(other, Pauli):

            for k, v in other.items():
                self[k] = self.get(k, 0.0) + v
            return self

        else:

            self[PauliString.I] = self.get(PauliString.I, 0.0) + other
            return self

        return NotImplemented

    def __isub__(self, other):

        if isinstance(other, Pauli):
    
            for k, v in other.items():
                self[k] = self.get(k, 0.0) - v
            return self

        else:
        
            self[PauliString.I] = self.get(PauliString.I, 0.0) - other
            return self

        return NotImplemented

    def dot(self, other):

        if not isinstance(other, Pauli): raise TypeError('other must be Pauli')

        return sum(v*other.get(k, 0.0) for k, v in self.items())

    @property
    def conj(self):
        return Pauli(collections.OrderedDict((k, np.conj(v)) for k, v in self.items()))

    @property
    def norm2(self):
        return np.sqrt(self.dot(self))
    
    @property
    def norminf(self):
        return np.max(np.abs(list(self.values())))

    @staticmethod
    def zero():
        return Pauli(collections.OrderedDict())

    @staticmethod   
    def zeros_like(x):
        return Pauli(collections.OrderedDict((k, 0.0) for k, v in x.items()))

    def sieved(self, cutoff=1.0E-14):
        return Pauli(collections.OrderedDict((k, v) for k, v in self.items() if np.abs(v) > cutoff))

    @staticmethod
    def I():
        return Pauli(collections.OrderedDict([(PauliString.I, 1.0)]))

    @staticmethod
    def IXYZ():
        return Pauli(collections.OrderedDict([(PauliString.I, 1.0)])), PauliStarter('X'), PauliStarter('Y'), PauliStarter('Z')

    # > Extra utility for run_pauli_expectation < #

    def extract_orders(
        self,
        orders,
        ):

        """ Return a subset of Pauli with only terms with specific orders retained.
        
        Params:
            orders (int, or tuple of int) - tuple of orders to retain
        Returns:
            (Pauli) - a version of this Pauli, but with only strings with order
                present in orders retained.
        """
        if isinstance(orders, int): orders=(orders,)
        
        return Pauli(collections.OrderedDict([(k, v) for k, v in self.items() if k.order in orders]))

    @property
    def qubits(self):

        return tuple([_.qubits for _ in self.keys()])

    @property
    def chars(self):

        return tuple([_.chars for _ in self.keys()])

    @property
    def unique_chars(self):
        
        return tuple(sorted(set(''.join(''.join(_) for _ in self.chars))))

    def compute_hilbert_matrix(
        self,
        dtype=np.complex128,
        N=None,
        ):
    
        N = self.N if N is None else N
        O = np.zeros((2**N,)*2, dtype=np.complex128)

        for string, value in self.items():
            bra_inds = list(range(2**N))
            factors = np.ones((2**N,), dtype=np.complex128)
            for operator in string:
                qubit, char = operator 
                test = 1 << (N - qubit - 1)
                if char == 'Z':
                    for I in range(2**N):
                        if I & test: factors[I] *= -1.0
                elif char == 'X':
                    for I in range(2**N):
                        bra_inds[I] ^= test
                elif char == 'Y':
                    for I in range(2**N):
                        bra_inds[I] ^= test
                    factors *= 1.j
                    for I in range(2**N):
                        if I & test: factors[I] *= -1.0
                else:
                    raise RuntimeError('Unknown char: %s' % char)
            O[bra_inds, range(2**N)] += factors * value

        return np.array(O, dtype=dtype)

    def compute_hilbert_matrix_vector_product(
        self,
        statevector,
        ):

        N = self.N
        if statevector.shape != (2**N,): raise RuntimeError('statevector must be shape (2**N,)')
        sigmavector = np.zeros((2**N,), dtype=np.complex128)

        for string, value in self.items():
            bra_inds = list(range(2**N))
            factors = np.ones((2**N,), dtype=np.complex128)
            for operator in string:
                qubit, char = operator 
                test = 1 << (N - qubit - 1)
                if char == 'Z':
                    for I in range(2**N):
                        if I & test: factors[I] *= -1.0
                elif char == 'X':
                    for I in range(2**N):
                        bra_inds[I] ^= test
                elif char == 'Y':
                    for I in range(2**N):
                        bra_inds[I] ^= test
                    factors *= 1.j
                    for I in range(2**N):
                        if I & test: factors[I] *= -1.0
                else:
                    raise RuntimeError('Unknown char: %s' % char)
            sigmavector[bra_inds] += factors * value * statevector

        return np.array(sigmavector, dtype=statevector.dtype)
    
class PauliExpectation(Pauli):

    def __str__(self):
        lines = []
        for string, value in self.items():
            lines.append('<%s> = %s' % (string, value))
        return '\n'.join(lines)

    @staticmethod   
    def zeros_like(x):
        return PauliExpectation(collections.OrderedDict((k, 0.0) for k, v in x.items()))

class PauliStarter(object):

    def __init__(
        self,
        char,
        ):

        if char not in ['X', 'Y', 'Z']: raise RuntimeError('char must be one of X, Y, or Z')
        self.char = char

    def __getitem__(self, qubit):
        return Pauli(collections.OrderedDict([(PauliString((PauliOperator(qubit=qubit, char=self.char),)), 1.0)]))

class PauliJordanWigner(object):

    class Composition(object):

        def __init__(
            self,
            creation=True,
            ):

            self.creation = creation

        def __getitem__(
            self,
            index,
            ):

            if not isinstance(index, int): raise RuntimeError('index must be int')

            Zstr = []
            for index2 in range(index):
                Zstr.append(PauliOperator(qubit=index2, char='Z'))
            Xstr = Zstr + [PauliOperator(qubit=index, char='X')]
            Ystr = Zstr + [PauliOperator(qubit=index, char='Y')]
            Xkey = PauliString(tuple(Xstr))
            Ykey = PauliString(tuple(Ystr))
            
            return Pauli(collections.OrderedDict([
                (Xkey, 0.5),
                (Ykey, -0.5j if self.creation else +0.5j),
                ]))

    @staticmethod
    def composition_operators():
        return PauliJordanWigner.Composition(creation=True), PauliJordanWigner.Composition(creation=False)

    class one_body(object):

        def __getitem__(
            self,   
            indices,
            ):

            """ Returns 0.5 * (p^+ q + q^+ p) """

            if not isinstance(indices, tuple): raise RuntimeError('indices must be tuple')
            if len(indices) != 2: raise RuntimeError('indices must be len 2')
            if not all(isinstance(_, int) for _ in indices): raise RuntimeError('indices must be tuple of int')

            if indices[0] == indices[1]:
                I, X, Y, Z = Pauli.IXYZ()
                return 0.5 * I - 0.5 * Z[indices[0]]
            else:
                index1 = min(indices)
                index2 = max(indices)
                Zstr = []
                for index3 in range(index1+1, index2):
                    Zstr.append(PauliOperator(qubit=index3, char='Z'))
                Xstr = [PauliOperator(qubit=index1, char='X')] + Zstr + [PauliOperator(qubit=index2, char='X')]
                Ystr = [PauliOperator(qubit=index1, char='Y')] + Zstr + [PauliOperator(qubit=index2, char='Y')]
                Xkey = PauliString(tuple(Xstr))
                Ykey = PauliString(tuple(Ystr))
                
                return Pauli(collections.OrderedDict([
                    (Xkey, 0.25),
                    (Ykey, 0.25),
                    ]))
    
    
    class two_body(object):
        '''
        Two body terms: p^+ q r^+ s
        '''
        
        def __getitem__(
            self,   
            indices,
            ):
            pass
        
        
        def two_states(
            self,   
            indices,
            ):
            
            # check
            if len(indices) != 4: raise RuntimeError('indices must be len 4')
            if all((indices.count(x)==2) for x in set(indices)): raise RuntimeError('indices must two pairs of identical integers')
            
            I, X, Y, Z = Pauli.IXYZ()
            i,j,k,l = np.argsort(indices)
            p, q = sorted(set(indices))
            
            # pauli operators
            init_str = 'abba'
            new_str = init_str[i] +init_str[j] +init_str[k] +init_str[l]
            if new_str=='abba' or 'baab':
                hamiltonian_pauli = +1/8 * (I + Z[p] + Z[q] + Z[p]*Z[q])
            elif new_str=='abab' or 'baba':
                hamiltonian_pauli = -1/8 * (I + Z[p] + Z[q] + Z[p]*Z[q])
            elif new_str=='aabb' or 'bbaa':
                hamiltonian_pauli = 0
            
            return hamiltonian_pauli
        
        
        
        def four_states(
            self,   
            indices,
            ):
        
            """
            Reference: Equation (34) in https://arxiv.org/pdf/0705.1928.pdf . The reference uses physicist's notation, while below we use chemist's notation, so the expression is a bit different. 
            
            Returns: p^+ q  r^+ s            
            """
  
            # check
            if len(indices) != 4: raise RuntimeError('indices must be len 4')
            if len(set(indices)) != 4: raise RuntimeError('all indices must be unique')
            
            p,q,r,s = indices
            I, X, Y, Z = Pauli.IXYZ()
            
            # (1) prefactor
            from sympy import Eijk
            i,j,k,l = np.argsort(indices)
            prefactor = 1/8 * Eijk(i, j, k, l)
            
            # (2) class of ordering
            init_str = 'abab'
            new_str = init_str[i] +init_str[j] +init_str[k] +init_str[l]
            if new_str=='aabb' or 'bbaa':
                postfactors = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0]
            elif new_str=='abab' or 'baba':
                postfactors = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]
            elif new_str=='abba':
                postfactors = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        
            # (3) pauli operators
            Zstr1 = Z[indices[i]+1]
            for idx in range(indices[i]+1, indices[j]-1):
                Zstr1 *= Z[idx]
            Zstr2 = Z[indices[k]+1]
            for idx in range(indices[k]+1, indices[l]-1):
                Zstr2 *= Z[idx]
            op0 = X[indices[i]] * Zstr1 * X[indices[j]] * X[indices[k]] * Zstr2 * X[indices[l]]
            op1 = X[indices[i]] * Zstr1 * X[indices[j]] * Y[indices[k]] * Zstr2 * Y[indices[l]]
            op2 = X[indices[i]] * Zstr1 * Y[indices[j]] * X[indices[k]] * Zstr2 * Y[indices[l]]
            op3 = X[indices[i]] * Zstr1 * Y[indices[j]] * Y[indices[k]] * Zstr2 * X[indices[l]]
            op4 = Y[indices[i]] * Zstr1 * X[indices[j]] * X[indices[k]] * Zstr2 * Y[indices[l]]
            op5 = Y[indices[i]] * Zstr1 * X[indices[j]] * Y[indices[k]] * Zstr2 * X[indices[l]]
            op6 = Y[indices[i]] * Zstr1 * Y[indices[j]] * X[indices[k]] * Zstr2 * X[indices[l]]
            op7 = Y[indices[i]] * Zstr1 * Y[indices[j]] * Y[indices[k]] * Zstr2 * Y[indices[l]]
            ops = [op0, op1, op2, op3, op4, op5, op6, op7]
            
            # combine
            hamiltonian_pauli = Pauli(collections.OrderedDict())
            for idx in range(8):
                hamiltonian_pauli += prefactor * postfactors[idx] * ops[idx]
            
            return hamiltonian_pauli
            
            
            
            
            
    
    
    @staticmethod
    def substitution1_operator():
        return PauliJordanWigner.Substitution1()
    
