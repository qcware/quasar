import collections
import numpy as np

class Ket(int):
    
    def __getitem__(
        self,
        key,
        ):

        return (self & (1 << key)) >> key

    def 

class Ket(str):

    def __new__(
        self,
        string,
        ):

        if not isinstance(string, str): raise RuntimeError('string must be str')
        if not all(_ == '0' or _ == '1' for _ in string): raise RuntimeError('Invalid Ket: %s' % string)

        return str.__new__(Ket, string)

    def __getitem__(
        self,
        key,
        ):

        return int(super(Ket, self).__getitem__(key))

    @property
    def nqubit(self):
        return len(self)

    @staticmethod
    def from_int(value, nqubit):
        if value >= (1 << nqubit): raise RuntimeError('value >= 2**nqubit: value=%s, nqubit=%s' % (value, nqubit))
        if value<0: raise RuntimeError('value must be a positive int')
        if not isinstance(value, int): raise RuntimeError('value must be int')
        start = bin(value)[2:]
        padded = ('0' * (nqubit - len(start))) + start
        return Ket(padded)

    @staticmethod
    def from_int_reverse(value, nqubit):
        if value >= (1 << nqubit): raise RuntimeError('value >= 2**nqubit: value=%s, nqubit=%s' % (value, nqubit))
        if value<0: raise RuntimeError('value must be a positive int')
        if not isinstance(value, int): raise RuntimeError('value must be int')
        start = bin(value)[2:]
        padded = ('0' * (nqubit - len(start))) + start
        return Ket(padded[::-1])

class Histogram(object):

    def items(self):
        return self.histogram.items()

    def keys(self):
        return self.histogram.keys()

    def values(self):
        return self.histogram.values()

    def __contains__(
        self,
        key,
        ):

        key = Ket(key)
        return self.histogram.__contains__(key)

    def __getitem__(
        self,
        key,
        ):

        key = Ket(key)
        return self.histogram.__getitem__(key)

    def get(
        self,
        key,
        default=None,
        ):

        key = Ket(key)
        return self.histogram.get(key, default)

    @property
    def nqubit(self):
        return next(iter(self.histogram.keys())).nqubit if len(self.histogram) else None

class ProbabilityHistogram(Histogram):

    def __init__(
        self,
        histogram={},
        nmeasurement=None,
        ):

        self.histogram = histogram.copy()
        self.nmeasurement = nmeasurement

        for k, v in self.histogram.items():
            if not isinstance(k, Ket): raise RuntimeError('Key must be Ket: %s' % k) 
            if not isinstance(v, float): raise RuntimeError('Value must be float: %s' % v) 

        if len(self.histogram) == 0: return
        if not all(_.nqubit == self.nqubit for _ in self.histogram.keys()): raise RuntimeError('All keys must have same nqubit')

    def __str__(self):
        s = ''
        s += 'nmeasurement : %r\n' % (self.nmeasurement) 
        for k in sorted(self.histogram.keys()):
            s += '|%s> : %8.6f\n' % (k, self.histogram[k])
        return s

    def to_count_histogram(self):
        if self.nmeasurement is None:
            raise RuntimeError('Cannot convert to count histogram: nmeasurement is None (infinite sampling)')
        return CountHistogram(
            histogram={ k : int(round(v * self.nmeasurement)) for k, v in self.items() },
            nmeasurement=self.nmeasurement,
            )

class CountHistogram(Histogram):

    def __init__(
        self,
        histogram={},
        nmeasurement=None,
        ):

        self.histogram = histogram.copy()
        self.nmeasurement = nmeasurement

        if self.nmeasurement is None: 
            raise RuntimeError('nmeasurement cannot be None in CountHistogram')

        for k, v in self.histogram.items():
            if not isinstance(k, Ket): raise RuntimeError('Key must be Ket: %s' % k) 
            if not isinstance(v, int): raise RuntimeError('Value must be int: %s' % v) 

        if len(self.histogram) == 0: return
        if not all(_.nqubit == self.nqubit for _ in self.histogram.keys()): raise RuntimeError('All keys must have same nqubit')
        if sum(self.histogram.values()) != self.nmeasurement: raise RuntimeError('Values do not sum to nmeasurement')

    def __str__(self):
        maxval = max(_ for _ in self.histogram.values())
        maxlen = max(1, int(np.floor(np.log10(maxval))) + 1)        
        s = ''
        s += 'nmeasurement : %r\n' % (self.nmeasurement) 
        for k in sorted(self.histogram.keys()):
            s += '|%s> : %*d\n' % (k, maxlen, self.histogram[k])
        return s

    def to_probability_histogram(self):
        if self.nmeasurement is None:
            raise RuntimeError('nmeasurement cannot be None in CountHistogram')
        return ProbabilityHistogram(
            histogram={ k : v / self.nmeasurement for k, v in self.items() },
            nmeasurement=self.nmeasurement,
            )
