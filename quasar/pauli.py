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
            if not isinstance(v, (float, complex)): raise RuntimeError('Value must be float or complex: %s' % v) 

    def __contains__(
        self,
        key,
        ):

        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        return super(Pauli, self).__contains__(key)

    def __getitem__(
        self,
        key,
        ):

        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        return super(Pauli, self).__getitem__(key)

    def __setitem__(
        self,
        key,
        value,
        ):

        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        if not isinstance(value, (float, complex)): raise RuntimeError('Value must be float or complex: %s' % value)
        return super(Pauli, self).__setitem__(key, value)

    def get(
        self,
        key,
        default=None,
        ):

        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        if default is not None and not isinstance(default, (float, complex)): raise RuntimeError('default must be float or complex: %s' % default)
        return super(Pauli, self).get(key, default)

    def setdefault(
        self,
        key,
        default=None,
        ):

        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        if default is not None and not isinstance(default, (float, complex)): raise RuntimeError('default must be float or complex: %s' % default)
        return super(Pauli, self).setdefault(key, default)

    def update(self, *args, **kwargs):
        raise RuntimeError('Pauli.update is not a well-defined operation, so we have poisoned this method of dict')
        
    # => String Representations <= #

    def __str__(self):
        lines = []
        for string, value in self.items():
            if isinstance(value, float): sign = -1 if value < 0.0 else +1
            elif isinstance(value, complex): sign = -1 if value.real < 0.0 else +1
            else: raise RuntimeError('value must be float or complex: %s' % value)
            lines.append('%s%s*%s' % ('-' if sign == -1 else '+', sign*value, string))
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
        
        if isinstance(other, (float, complex)):
        
            return Pauli(collections.OrderedDict((k, other*v) for k, v in self.items()))

        elif isinstance(other, Pauli):

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

        return NotImplemented
            
    def __rmul__(self, other):
        
        if isinstance(other, (float, complex)):
        
            return Pauli(collections.OrderedDict((k, other*v) for k, v in self.items()))

        return NotImplemented

    def __truediv__(self, other):
        
        if isinstance(other, (float, complex)):
        
            return Pauli(collections.OrderedDict((k, v/other) for k, v in self.items()))

        return NotImplemented

    def __add__(self, other):

        if isinstance(other, Pauli):

            pauli2 = self.copy()
            for k, v in other.items():
                pauli2[k] = self.get(k, 0.0) + v
            return pauli2

        elif isinstance(other, (float, complex)):

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

        elif isinstance(other, (float, complex)):

            pauli2 = self.copy()
            pauli2[PauliString.I] = self.get(PauliString.I, 0.0) - other
            return pauli2 

        return NotImplemented

    def __iadd__(self, other):

        if isinstance(other, Pauli):

            for k, v in other.items():
                self[k] = self.get(k, 0.0) + v
            return self

        elif isinstance(other, (float, complex)):

            self[PauliString.I] = self.get(PauliString.I, 0.0) + other
            return self

        return NotImplemented

    def __isub__(self, other):

        if isinstance(other, Pauli):
    
            for k, v in other.items():
                self[k] = self.get(k, 0.0) - v
            return self

        elif isinstance(other, (float, complex)):

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
    def IXYZ():
        return Pauli(collections.OrderedDict([(PauliString.I, 1.0)])), PauliStarter('X'), PauliStarter('Y'), PauliStarter('Z')

class PauliStarter(object):

    def __init__(
        self,
        char,
        ):

        if char not in ['X', 'Y', 'Z']: raise RuntimeError('char must be one of X, Y, or Z')
        self.char = char

    def __getitem__(self, qubit):
        return Pauli(collections.OrderedDict([(PauliString((PauliOperator(qubit=qubit, char=self.char),)), 1.0)]))