import collections
import numpy as np

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

class Pauli(collections.OrderedDict):

    def __init__(
        self,
        *args,
        **kwargs,
        ):

        super(Pauli, self).__init__(*args, **kwargs)

        for k, v in self.items():
            if not isinstance(k, PauliString): raise RuntimeError('Key must be PauliString: %s' % k) 
            if not isinstance(v, float): raise RuntimeError('Value must be float: %s' % v) 

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
        if not isinstance(value, float): raise RuntimeError('Value must be float: %s' % value)
        return super(Pauli, self).__setitem__(key, value)

    def get(
        self,
        key,
        default=None,
        ):

        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        if default is not None and not isinstance(default, float): raise RuntimeError('default must be float: %s' % default)
        return super(Pauli, self).get(key, default)

    def setdefault(
        self,
        key,
        default=None,
        ):

        if not isinstance(key, PauliString): raise RuntimeError('Key must be PauliString: %s' % key)
        if default is not None and not isinstance(default, float): raise RuntimeError('default must be float: %s' % default)
        return super(Pauli, self).setdefault(key, default)

    def update(self, *args, **kwargs):
        raise RuntimeError('Pauli.update is not a well-defined operation, so we have poisoned this method of dict')
        
    # => String Representations <= #

    def __str__(self):
        return '\n'.join(['%s*%s' % (value, string) for string, value in self.items()])

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
        return max(max(_.indices) for _ in self.keys() if _.order > 0) + 1
        
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
        
        if isinstance(other, float):
        
            return Pauli(collections.OrderedDict((k, other*v) for k, v in self.items()))

        elif isinstance(other, Pauli):

            if self.nterm != 1 or other.nterm != 1: raise RuntimeError('Can only multiply single string Paulis')

            return Pauli(collections.OrderedDict([(PauliString(list(self.keys())[0] + list(other.keys())[0]), list(self.values())[0]*list(other.values())[0])]))

        return NotImplemented
            
    def __rmul__(self, other):
        
        if isinstance(other, float):
        
            return Pauli(collections.OrderedDict((k, other*v) for k, v in self.items()))

        return NotImplemented

    def __truediv__(self, other):
        
        if isinstance(other, float):
        
            return Pauli(collections.OrderedDict((k, v/other) for k, v in self.items()))

        return NotImplemented

    def __add__(self, other):

        if isinstance(other, Pauli):

            pauli2 = self.copy()
            for k, v in other.items():
                pauli2[k] = self.get(k, 0.0) + v
            return pauli2

        return NotImplemented

    def __sub__(self, other):

        if isinstance(other, Pauli):

            pauli2 = self.copy()
            for k, v in other.items():
                pauli2[k] = self.get(k, 0.0) - v
            return pauli2

        return NotImplemented

    def __iadd__(self, other):

        if isinstance(other, Pauli):
            for k, v in other.items():
                self[k] = self.get(k, 0.0) + v
            return self

        return NotImplemented

    def __isub__(self, other):

        if isinstance(other, Pauli):
            for k, v in other.items():
                self[k] = self.get(k, 0.0) - v
            return self

        return NotImplemented

    def dot(self, other):

        if not isinstance(other, Pauli): raise TypeError('other must be Pauli')

        return sum(v*other.get(k,0.0) for k, v in self.items())

    @property
    def norm2(self):
        return np.sqrt(self.dot(self))
    
    @property
    def norminf(self):
        return np.max(np.abs(self.values()))

    @staticmethod   
    def zeros_like(x):
        return Pauli(collections.OrderedDict((k, 0.0) for k, v in self.items()))

class PauliStarter(object):

    def __init__(
        self,
        char,
        ):

        if char not in ['X', 'Y', 'Z']: raise RuntimeError('char must be one of X, Y, or Z')
        self.char = char

    def __getitem__(self, index):
        return Pauli(collections.OrderedDict([(PauliString((PauliOperator(index=index, char=self.char),)), 1.0)]))

PauliStarter.X = PauliStarter('X')
PauliStarter.Y = PauliStarter('Y')
PauliStarter.Z = PauliStarter('Z')

Pauli.X = PauliStarter.X
Pauli.Y = PauliStarter.Y
Pauli.Z = PauliStarter.Z
