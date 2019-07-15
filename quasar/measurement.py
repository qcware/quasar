import collections
import numpy as np

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
    def N(self):
        return len(self)

    @staticmethod
    def from_int(value, N):
        if value >= (1 << N): raise RuntimeError('value >= 2**N: value=%s, N=%s' % (value, N))
        if value<0: raise RuntimeError('value must be a positive int')
        if not isinstance(value, int): raise RuntimeError('value must be int')
        start = bin(value)[2:]
        padded = ('0' * (N - len(start))) + start
        return Ket(padded)

class MeasurementResult(dict):    

    def __init__(
        self, 
        *args,
        **kwargs,
        ):

        super(MeasurementResult, self).__init__(*args, **kwargs)

        for k, v in self.items():
            if not isinstance(k, Ket): raise RuntimeError('Key must be Ket: %s' % k) 
            if not isinstance(v, int): raise RuntimeError('Value must be int: %s' % v) 
            if v < 0: raise RuntimeError('Value must be positive: %s' % v)

        if len(self) == 0: return
        if not all(_.N == self.N for _ in self.keys()): raise RuntimeError('All keys must have same N')

    def __contains__(
        self,
        key,
        ):

        
        key = Ket(key)
        return super(MeasurementResult, self).__contains__(key)

    def __getitem__(
        self,
        key,
        ):

        key = Ket(key)
        return super(MeasurementResult, self).__getitem__(key)

    def __setitem__(
        self,
        key,
        value,
        ):

        key = Ket(key)
        if not isinstance(value, int): raise RuntimeError('Value must be int: %s' % value)
        if value < 0: raise RuntimeError('Value must be positive: %s' % value)
        if len(self) and key.N != self.N: raise RuntimeError('All keys must have same N')
        return super(MeasurementResult, self).__setitem__(key, value)

    def get(
        self,
        key,
        default=None,
        ):

        key = Ket(key)
        if default is not None and not isinstance(default, int): raise RuntimeError('default must be int: %s' % default)
        return super(MeasurementResult, self).get(key, default)

    def setdefault(
        self,
        key,
        default=None,
        ):

        key = Ket(key)
        if default is not None and not isinstance(default, int): raise RuntimeError('default must be int: %s' % default)
        if default < 0: raise RuntimeError('default must be positive: %s' % default)
        if len(self) and key.N != self.N: raise RuntimeError('All keys must have same N')
        return super(MeasurementResult, self).setdefault(key, default)

    def update(self, *args, **kwargs):
        raise RuntimeError('MeasurementResult.update is not a well-defined operation, so we have poisoned this method of dict')

    @property
    def N(self):
        return next(iter(self.keys())).N if len(self) else None
        
    @property
    def nmeasurement(self):
        return sum(v for v in self.values()) if len(self) else 0

    def __str__(self):
        maxval = max(_ for _ in self.values())
        maxlen = max(1, int(np.floor(np.log10(maxval))) + 1)        
        s = ''
        for k in sorted(self.keys()):
            s += '|%s> : %*d\n' % (k, maxlen, self[k])
        return s

    def subset(
        self,
        qubits,
        ):

        if not all(isinstance(qubit, int) for qubit in qubits): raise RuntimeError('qubits must be int')
        if any(qubit >= self.N or qubit < 0 for qubit in qubits): raise RuntimeError('qubits must be in [0, N)')
        if len(set(qubits)) != len(qubits): raise RuntimeError('qubits must be unique')

        measurement = MeasurementResult()
        for k, v in self.items():
            k2 = ''.join(str(k[qubit]) for qubit in qubits)
            measurement[k2] = self[k] + measurement.get(k2, 0)
        return measurement

class OptimizationResult(collections.OrderedDict):    

    """ Ket : (E, N) """

    def __init__(
        self, 
        *args,
        **kwargs,
        ):

        super(OptimizationResult, self).__init__(*args, **kwargs)

        for k, v in self.items():
            if not isinstance(k, Ket): raise RuntimeError('Key must be Ket: %s' % k) 
            if not isinstance(v, tuple): raise RuntimeError('Value must be tuple: %s' % v) 

        if len(self) == 0: return
        if not all(_.N == self.N for _ in self.keys()): raise RuntimeError('All keys must have same N')

    def __contains__(
        self,
        key,
        ):

        
        key = Ket(key)
        return super(OptimizationResult, self).__contains__(key)

    def __getitem__(
        self,
        key,
        ):

        key = Ket(key)
        return super(OptimizationResult, self).__getitem__(key)

    def __setitem__(
        self,
        key,
        value,
        ):

        key = Ket(key)
        if len(self) and key.N != self.N: raise RuntimeError('All keys must have same N')
        return super(OptimizationResult, self).__setitem__(key, value)

    def get(
        self,
        key,
        default=None,
        ):

        key = Ket(key)
        return super(OptimizationResult, self).get(key, default)

    def setdefault(
        self,
        key,
        default=None,
        ):

        key = Ket(key)
        if len(self) and key.N != self.N: raise RuntimeError('All keys must have same N')
        return super(OptimizationResult, self).setdefault(key, default)

    def update(self, *args, **kwargs):
        raise RuntimeError('OptimizationResult.update is not a well-defined operation, so we have poisoned this method of dict')

    @property
    def N(self):
        return next(iter(self.keys())).N if len(self) else None

    @property
    def energy_sorted(self):
        result = OptimizationResult()
        for E, N, ket in sorted([(v[0], -v[1], k) for k, v in self.items()]):
            result[ket] = (E, -N)
        return result
        
    def __str__(self):
        sort = self.energy_sorted
        maxval = max(_[1] for _ in sort.values())
        maxlen = max(1, int(np.floor(np.log10(maxval))) + 1)        
        s = ''
        for k, v in sort.items():
            s += '|%s> : %10.6f %*d\n' % (k, v[0], maxlen, v[1])
        return s

    @staticmethod
    def merge(results):
        result = OptimizationResult() 
        for result2 in results:
            for ket, v in result2.items():
                E1, N1 = v
                E2, N2 = result.get(ket, (np.inf, 0))
                result[ket] = (min(E1, E2), N1 + N2)
        return result
