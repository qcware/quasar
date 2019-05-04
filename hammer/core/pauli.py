import numpy as np
import itertools

class PauliOperator(object):

    def __init__(
        self,
        index,
        char,
        ):

        self.index = index
        self.char = char

        if not isinstance(self.index, int): raise RuntimeError('index must be int')
        if not isinstance(self.char, str): raise RuntimeError('char must be str')
        if self.char not in ['X', 'Y', 'Z']: raise RuntimeError('char must be one of X, Y, or Z')

    # => String Representations <= #

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
    
    # => Relationals <= #

    def __repr__(self):
        return 'PauliOperator(%r, %r)' % (self.index, self.char)

    def __gt__(self, other):
        return (self.index, self.char) > (other.index, other.char)

    def __ge__(self, other):
        return (self.index, self.char) >= (other.index, other.char)

    def __lt__(self, other):
        return (self.index, self.char) < (other.index, other.char)

    def __le__(self, other):
        return (self.index, self.char) <= (other.index, other.char)

    def __eq__(self, other):
        return (self.index, self.char) == (other.index, other.char)
    
    def __neq__(self, other):
        return (self.index, self.char) != (other.index, other.char)
    
    # => Hashing <= #

    def __hash__(self):
        return (self.index, self.char).__hash__()
    
class PauliString(object):

    def __init__(
        self,
        operators,
        ):

        self.operators = operators

        if not isinstance(self.operators, tuple): raise RuntimeError('operators must be tuple')
        if not all(isinstance(_, PauliOperator) for _ in self.operators): raise RuntimeError('operators must all be Pauli Operator')
        if len(set(self.indices)) != len(self.operators): raise RuntimeError('operators must all refer to unique qubit indices')

    # => Attributes <= #

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
            s += '%s*' % operator
        s += '%s' % self.operators[-1]
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

    # => Relationals <= #

    def __gt__(self, other):
        if len(self.operators) > len(other.operators): return True
        if len(self.operators) < len(other.operators): return False
        return self.operators > other.operators

    def __ge__(self, other):
        if len(self.operators) > len(other.operators): return True
        if len(self.operators) < len(other.operators): return False
        return self.operators >= other.operators

    def __lt__(self, other):
        if len(self.operators) < len(other.operators): return True
        if len(self.operators) > len(other.operators): return False
        return self.operators < other.operators

    def __le__(self, other):
        if len(self.operators) < len(other.operators): return True
        if len(self.operators) > len(other.operators): return False
        return self.operators <= other.operators

    def __eq__(self, other):
        if len(self.operators) != len(other.operators): return False
        return self.operators == other.operators

    def __neq__(self, other):
        if len(self.operators) != len(other.operators): return True
        return self.operators != other.operators

    # => Hashing <= #

    def __hash__(self):
        return self.operators.__hash__()

class Pauli(object):

    def __init__(
        self,
        strings,
        values,
        ):

        self.strings = strings
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
            values=other.values,
            )

    def __neg__(self):
        return Pauli(
            strings=self.strings,
            values=-other.values,
            )

    def __mul__(self, other):
        
        if not isinstance(other, float): raise TypeError('other must be float')
        
        return Pauli(
            strings=self.strings,
            values=other*self.values,
            )

    def __rmul__(self, other):
        
        if not isinstance(other, float): raise TypeError('other must be float')
        
        return Pauli(
            strings=self.strings,
            values=other*self.values,
            )

    def __add__(self, other):

        if not isinstance(other, Pauli): raise TypeError('other must be Pauli')

        if self.strings != other.strings:
            raise RuntimeError('self and other do not have equivalent strings')
        
        return Pauli(
            strings=self.strings,
            values=self.values + other.values,
            )

    def __sub__(self, other):

        if not isinstance(other, Pauli): raise TypeError('other must be Pauli')

        if self.strings != other.strings:
            raise RuntimeError('self and other do not have equivalent strings')
        
        return Pauli(
            strings=self.strings,
            values=self.values - other.values,
            )

    def dot(self, other):

        if not isinstance(other, Pauli): raise TypeError('other must be Pauli')
    
        if self.strings != other.strings:
            raise RuntimeError('self and other do not have equivalent strings')

        return np.sum(self.values * other.values)

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

    # => Utility Constructors <= #
    
    @staticmethod
    def build_from_qubo(
        Q,
        C=None,
        cutoff=0.0,
        ):

        # Make sure the problem is symmetric
        QS = 0.5 * (Q + Q.T)

        # QUBO -> Ising mapping
        C2 = C if C else 0.0
        C2 += 0.25 * np.sum(QS)
        C2 += 0.25 * np.sum(np.diag(QS))
        Z2 = -0.50 * np.sum(QS, 1)
        Q2 =  0.25 * (QS - np.diag(np.diag(QS))) 

        strings = []
        values = []
        if np.abs(C2) > cutoff:
            strings.append(PauliString(tuple()))
            values.append(C2)
        for i in range(Z2.shape[0]):
            if np.abs(Z2[i]) > cutoff:
                strings.append(PauliString((
                    PauliOperator(index=i, char='Z'),
                    )))
                values.append(Z2[i])
        for i in range(Q2.shape[0]):
            for j in range(Q2.shape[1]):
                if np.abs(Q2[i,j]) > cutoff:
                    strings.append(PauliString((
                        PauliOperator(index=i, char='Z'), 
                        PauliOperator(index=j, char='Z'),
                        )))
                    values.append(Q2[i,j])
        strings = tuple(strings)
        values = np.array(values)

        return Pauli(
            strings=strings, 
            values=values,
            )

if __name__ == '__main__':

    Q = np.array([
        [-5,  2,  4,  0],
        [ 2, -3,  1,  0],
        [ 4,  1, -8,  5],
        [ 0,  0,  5, -6],
        ], dtype=np.float64)

    pauli = Pauli.build_from_qubo(Q=Q)
    pauli2 = Pauli.build_from_qubo(Q=Q+6)
    print(pauli.strings)    
    print(pauli.values)    
    print(pauli)
    print('%r' % pauli)
    print(pauli.max_order)
    print(Pauli.equivalent_strings(pauli, pauli))

    print(pauli + pauli)
    print(pauli - pauli)
    print(3.0 * pauli)
    print(pauli * 3.0)
    print(pauli * 3.0 - 2.0 * pauli)
    # pauli -= pauli
    print(pauli)

    print(PauliString.from_string('Z1*Z3*X2'))
    print(PauliString.from_string('X1'))
    print(PauliString.from_string('1'))

    print(pauli['Z1*Z2'])
    print(pauli[PauliString.from_string('Z1*Z2')])

    pauli['Z1*Z2'] += 2.0
    print(pauli['Z1*Z2'])
    # print(pauli['Z1*X1'])

    print(pauli == pauli)
    print(pauli != pauli)
    print(pauli == pauli2)

    print(pauli.norm2)
    print(pauli.norminf)

    print('1' in pauli)

    print(pauli.indices(0))
    print(pauli.indices(1))
    print(pauli.indices(2))
