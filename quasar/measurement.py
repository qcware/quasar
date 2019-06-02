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
        start = bin(value)[2:]
        padded = ('0' * (N - len(start))) + start
        return Ket(padded)

class Measurement(dict):    

    def __init__(
        self, 
        *args,
        **kwargs,
        ):

        super(Measurement, self).__init__(*args, **kwargs)

        for k, v in self.items():
            if not isinstance(k, Ket): raise RuntimeError('Key must be Ket: %s' % k) 
            if not isinstance(v, int): raise RuntimeError('Value must be int: %s' % v) 
            if v < 0: raise RuntimeError('Value must be positive: %s' % v)

        selfN = self.N
        if not all(_.N == self.N for _ in self.keys()): raise RuntimeError('All keys must have same N')

    def __contains__(
        self,
        key,
        ):

        if not isinstance(key, Ket): raise RuntimeError('Key must be Ket: %s' % key)
        return super(Measurement, self).__contains__(key)

    def __getitem__(
        self,
        key,
        ):

        if not isinstance(key, Ket): raise RuntimeError('Key must be Ket: %s' % key)
        return super(Measurement, self).__getitem__(key)

    def __setitem__(
        self,
        key,
        value,
        ):

        if not isinstance(key, Ket): raise RuntimeError('Key must be Ket: %s' % key)
        if not isinstance(value, int): raise RuntimeError('Value must be int: %s' % value)
        if value < 0: raise RuntimeError('Value must be positive: %s' % value)
        if key.N != self.N: raise RuntimeError('All keys must have same N')
        return super(Measurement, self).__setitem__(key, value)

    def get(
        self,
        key,
        default=None,
        ):

        if not isinstance(key, Ket): raise RuntimeError('Key must be Ket: %s' % key)
        if default is not None and not isinstance(default, int): raise RuntimeError('default must be int: %s' % default)
        return super(Measurement, self).get(key, default)

    def setdefault(
        self,
        key,
        default=None,
        ):

        if not isinstance(key, Ket): raise RuntimeError('Key must be Ket: %s' % key)
        if default is not None and not isinstance(default, int): raise RuntimeError('default must be int: %s' % default)
        if default < 0: raise RuntimeError('default must be positive: %s' % default)
        if key.N != self.N: raise RuntimeError('All keys must have same N')
        return super(Measurement, self).setdefault(key, default)

    def update(self, *args, **kwargs):
        raise RuntimeError('Measurement.update is not a well-defined operation, so we have poisoned this method of dict')

    @property
    def N(self):
        return next(iter(self.keys())).N
        
    @property
    def nmeasurement(self):
        return sum(v for v in self.values())

    def __str__(self):
        maxval = max(_ for _ in self.values())
        maxlen = max(1, int(np.floor(np.log10(maxval))) + 1)        
        s = ''
        for k in sorted(self.keys()):
            s += '|%s> : %*d\n' % (k, maxlen, self[k])
        return s

