import numpy as np
import itertools

class PauliOperator(tuple):

    def __new__(
        self,
        index,
        char,
        ):

        if not isinstance(index, int): raise RuntimeError('index must be int')
        if not isinstance(char, str): raise RuntimeError('char must be str')
        if char not in ['X', 'Y', 'Z']: raise RuntimeError('char must be one of X, Y, or Z')

        return tuple.__new__(PauliOperator, (index, char))

    @property
    def index(self):
        return self[0]

    @property
    def char(self):
        return self[1]

    def __str__(self):
        return '%s%d' % (self.char, self.index)

    @staticmethod
    def from_string(string):
        char = string[0]
        index = int(string[1:]) 
        return PauliOperator(
            index=index,
            char=char,
            )

class PauliString(tuple):

    def __new__(
        self,
        operators,
        ):

        if not isinstance(operators, tuple): raise RuntimeError('operators must be tuple')
        if not all(isinstance(_, PauliOperator) for _ in operators): raise RuntimeError('operators must all be Pauli Operator')
        if len(set(operator.index for operator in operators)) != len(operators): raise RuntimeError('operators must all refer to unique qubit indices')

        return tuple.__new__(PauliString, operators)

    # => Attributes <= #

    @property
    def operators(self):
        return self

    @property
    def order(self):
        return len(self.operators)

    @property
    def indices(self):
        return tuple([_.index for _ in self.operators])

    @property
    def chars(self):
        return tuple([_.char for _ in self.operators])

    # => String Representations <= #

    def __str__(self):
        if len(self.operators) == 0: return '1'
        s = ''
        for operator in self.operators[:-1]:
            s += '%s*' % str(operator)
        s += str(self.operators[-1])
        return s

    @staticmethod
    def from_string(string):
        if string == '1': 
            return PauliString(
                operators=tuple(),
                )
        else:
            return PauliString(
                operators=tuple(PauliOperator.from_string(_) for _ in string.split('*')),
                )

    def __repr__(self):
        return 'PauliString(%s)' % repr(self.operators)
        

class Pauli(object):

    def __init__(
        self,
        strings,
        values=None,
        ):

        self.strings = strings
        if values is None:
            self.values = np.zeros((len(strings),))
        else:
            self.values = values 
        
        if not isinstance(self.strings, tuple): raise RuntimeError('strings must be tuple')
        if not isinstance(self.values, np.ndarray): raise RuntimeError('values must be np.ndarray')

        if len(self.strings) != self.nterm: raise RuntimeError('len(strings) must be nterm')
        if self.values.shape != (self.nterm,): raise RuntimeError('values.shape must be (nterm,)')

        if len(set(self.strings)) != len(self.strings): raise RuntimeError('strings are not unique')

        self._index_map = { str(k) : k2 for k2, k in enumerate(self.strings) }

    # => Attributes <= #

    @property
    def N(self):
        return max(max(_.indices) for _ in self.strings if _.order > 0) + 1
        
    @property
    def nterm(self):
        return len(self.strings)

    @property
    def max_order(self):
        return max(_.order for _ in self.strings)

    def indices(self, order):
        return tuple(sorted(set(_.indices for _ in self.strings if _.order==order))) 
    
    # => String Representations <= #

    def __str__(self):
        s = 'Pauli:\n'
        s += '  %-10s = %d\n' % ('N', self.N)
        s += '  %-10s = %d\n' % ('nterm', self.nterm)
        s += '  %-10s = %d\n' % ('max_order', self.max_order)
        return s 

    def __repr__(self):
        return 'Pauli(%r, %r)' % (self.strings, self.values)

    @property
    def content_str(self):
        return '\n'.join(['%s*%s' % (value, string) for string, value in zip(self.strings, self.values)])

    # => Dict-Type Views <= #

    def __getitem__(self, key):
        return self.values[self._index_map[str(key)]]

    def __setitem__(self, key, value):
        self.values[self._index_map[str(key)]] = value

    def __contains__(self, key):
        return self._index_map.__contains__(str(key))

    @staticmethod
    def equivalent_strings(x, y):
        return x.strings == y.strings

    # => Arithmetic <= #

    def __pos__(self):
        return Pauli(
            strings=self.strings,
            values=+self.values,
            )

    def __neg__(self):
        return Pauli(
            strings=self.strings,
            values=-self.values,
            )

    def __mul__(self, other):
        
        if isinstance(other, float):
        
            return Pauli(
                strings=self.strings,
                values=other*self.values,
                )

        elif isinstance(other, Pauli):

            if self.nterm != 1 or other.nterm != 1: raise RuntimeError('Can only multiply single string Paulis')

            return Pauli(
                strings=tuple([PauliString(self.strings[0] + other.strings[0])]),
                values=self.values*other.values,
                )

        return NotImplemented
            
    def __rmul__(self, other):
        
        if isinstance(other, float): 
        
            return Pauli(
                strings=self.strings,
                values=other*self.values,
                )

        return NotImplemented

    def __truediv__(self, other):
        
        if isinstance(other, float): 
        
            return Pauli(
                strings=self.strings,
                values=self.values/other,
                )

        return NotImplemented

    def __add__(self, other):

        if isinstance(other, Pauli):

            strings = [_ for _ in self.strings]
            values = [_ for _ in self.values]
            
            for string, value in zip(other.strings, other.values):
                index = self._index_map.get(str(string), None)
                if index is None:
                    strings.append(string)
                    values.append(value)        
                else:
                    values[index] += value
            
            return Pauli(
                strings=tuple(strings),
                values=np.array(values),
                )

        return NotImplemented

    def __sub__(self, other):

        if isinstance(other, Pauli):

            strings = [_ for _ in self.strings]
            values = [_ for _ in self.values]
            
            for string, value in zip(other.strings, other.values):
                index = self._index_map.get(str(string), None)
                if index is None:
                    strings.append(string)
                    values.append(-value)        
                else:
                    values[index] -= value
            
            return Pauli(
                strings=tuple(strings),
                values=np.array(values),
                )

        return NotImplemented

    def dot(self, other):

        if not isinstance(other, Pauli): raise TypeError('other must be Pauli')

        prod = 0.0
        for string, value in zip(other.strings, other.values):
            index = self._index_map.get(str(string), None)
            if index is not None: prod += self.values[index] * value
    
        return prod

    @property
    def norm2(self):
        return np.sqrt(self.dot(self))
    
    @property
    def norminf(self):
        return np.max(np.abs(self.values))

    @staticmethod   
    def zeros_like(x):

        if not isinstance(x, Pauli): raise TypeError('x must be Pauli')
    
        return Pauli(
            strings=x.strings,
            values=np.zeros_like(x.values),
            )

class PauliStarter(object):

    def __init__(
        self,
        char,
        ):

        if char not in ['X', 'Y', 'Z']: raise RuntimeError('char must be one of X, Y, or Z')
        self.char = char

    def __getitem__(self, index):
        return Pauli(
            strings=tuple([PauliString((PauliOperator(index=index, char=self.char),))]),
            values=np.array((1.0,)),
            )

PauliStarter.X = PauliStarter('X')
PauliStarter.Y = PauliStarter('Y')
PauliStarter.Z = PauliStarter('Z')

Pauli.X = PauliStarter.X
Pauli.Y = PauliStarter.Y
Pauli.Z = PauliStarter.Z
