import numpy as np

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

        if isinstance(key, str): key = int(key, base=2)
        return self.histogram.__contains__(key)

    def __getitem__(
        self,
        key,
        ):

        if isinstance(key, str): key = int(key, base=2)
        return self.histogram.__getitem__(key)

    def get(
        self,
        key,
        default=None,
        ):

        if isinstance(key, str): key = int(key, base=2)
        return self.histogram.get(key, default)

class ProbabilityHistogram(Histogram):
    """ Class ProbabilityHistogram represents the output of the
    measurement process for a Circuit. A ProbabilityHistogram object
    contains the number of qubits measured (``nqubit`` attribute), the total
    number of measurements made (``nmeasurement`` attribute), and a histogram
    of binary strings representing each collective measurement result
    (``Ket`` strings) together with the probability that each one was observed. 

    To access the probability corresponding to a specific ``Ket`` string, index into
    the ProbabilityHistogram instance with the integer equivalent of the binary
    ``Ket`` string or the binary string itself.
    """

    def __init__(
        self,
        nqubit,
        histogram={},
        nmeasurement=None,
        ):

        self.nqubit = nqubit
        self.histogram = histogram.copy()
        self.nmeasurement = nmeasurement

        if not isinstance(self.nqubit, int): 
            raise RuntimeError('nqubit must be int')
        if self.nmeasurement is not None and not isinstance(self.nmeasurement, (int)):
            raise RuntimeError('nmeasurement must be int or None')
        if not all(isinstance(key, int) for key in self.histogram.keys()):
            raise RuntimeError('Keys must be int')
        if not all(isinstance(value, float) for value in self.histogram.values()):
            raise RuntimeError('Value must be float')

    def __str__(self): 
    
        s = ''
        s += 'nqubit       : %r\n' % (self.nqubit)
        s += 'nmeasurement : %r\n' % (self.nmeasurement)
        for k in sorted(self.histogram.keys()):
            ket = bin(k)[2:]
            ket = '0' * (self.nqubit - len(ket)) + ket
            s += '|%s> : %8.6f\n' % (ket, self.histogram[k])
        return s

    def to_count_histogram(self):
        """ Convert from a ProbabilityHistogram to a CountHistogram.""" 

        if self.nmeasurement is None:
            raise RuntimeError('Cannot convert to count histogram: nmeasurement is None (infinite sampling)')
        return CountHistogram(
            nqubit=self.nqubit,
            histogram={ k : int(round(v * self.nmeasurement)) for k, v in self.items() },
            nmeasurement=self.nmeasurement,
            )

    def subset(
        self,
        qubits,
        ):

        if len(set(qubits)) != len(qubits): raise RuntimeError('qubits are not unique')
        if any(_ < 0 or _ >= self.nqubit for _ in qubits): raise RuntimeError('qubits do not lie in [0, nqubit)')

        nqubit2 = len(qubits)
        histogram2 = {}
        for key, value in self.histogram.items():
            key2 = 0
            for qubit2, qubit in enumerate(qubits):
                key2 += ((key & (1 << (self.nqubit - 1 - qubit))) >> (self.nqubit - 1 - qubit)) << (nqubit2 - 1 - qubit2)
            histogram2[key2] = value + histogram2.get(key2, 0.0)

        return ProbabilityHistogram(
            nqubit=nqubit2,
            histogram=histogram2,
            nmeasurement=self.nmeasurement,
            )

class CountHistogram(Histogram):
    """ Class CountHistogram represents the output of thhe measurement
    process for a Circuit. A CountHistogram object contains the number
    of qubits measured (``nqubit`` attribute), the total number of
    measurements made (``nmeasurement`` attribute), and a histogram
    of binary strings representing the collective measurement result
    (``Ket`` strings) together with the integral number of times that
    each one was observed.

    To access the probability corresponding to a specific ``Ket``
    string, index into the CountHistogram instance with the
    integer equivalent of the binary ``Ket`` string or the binary
    string itself.
    """

    def __init__(
        self,
        nqubit,
        histogram={},
        nmeasurement=None,
        ):

        self.nqubit = nqubit
        self.histogram = histogram.copy()
        self.nmeasurement = nmeasurement

        if not isinstance(self.nqubit, int): 
            raise RuntimeError('nqubit must be int')
        if not isinstance(self.nmeasurement, (int)):
            raise RuntimeError('nmeasurement must be int')
        if not all(isinstance(key, int) for key in self.histogram.keys()):
            raise RuntimeError('Keys must be int')
        if not all(isinstance(value, int) for value in self.histogram.values()):
            raise RuntimeError('Value must be int')

    def __str__(self): 
    
        maxval = max(_ for _ in self.histogram.values())
        maxlen = max(1, int(np.floor(np.log10(maxval))) + 1)        
        s = ''
        s += 'nqubit       : %r\n' % (self.nqubit)
        s += 'nmeasurement : %r\n' % (self.nmeasurement)
        for k in sorted(self.histogram.keys()):
            ket = bin(k)[2:]
            ket = '0' * (self.nqubit - len(ket)) + ket
            s += '|%s> : %*d\n' % (ket, maxlen, self.histogram[k])
        return s

    def to_probability_histogram(self):
        """ Convert from a CountHistogram to a ProbabilityHistogram."""

        if self.nmeasurement is None:
            raise RuntimeError('nmeasurement cannot be None in CountHistogram')
        return ProbabilityHistogram(
            nqubit=self.nqubit,
            histogram={ k : v / self.nmeasurement for k, v in self.items() },
            nmeasurement=self.nmeasurement,
            )

    def subset(
        self,
        qubits,
        ):

        if len(set(qubits)) != len(qubits): raise RuntimeError('qubits are not unique')
        if any(_ < 0 or _ >= self.nqubit for _ in qubits): raise RuntimeError('qubits do not lie in [0, nqubit)')

        nqubit2 = len(qubits)
        histogram2 = {}
        for key, value in self.histogram.items():
            key2 = 0
            for qubit2, qubit in enumerate(qubits):
                key2 += ((key & (1 << (self.nqubit - 1 - qubit))) >> (self.nqubit - 1 - qubit)) << (nqubit2 - 1 - qubit2)
            histogram2[key2] = value + histogram2.get(key2, 0)

        return CountHistogram(
            nqubit=nqubit2,
            histogram=histogram2,
            nmeasurement=self.nmeasurement,
            )
