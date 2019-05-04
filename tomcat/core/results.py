class Ket(object):

    def __init__(
        self,
        bits,
        ):

        self.bits = bits
        
    def __getitem__(self, index):
        return self.bits[index]

    def __str__(self):
        return ''.join(str(_) for _ in self.bits)

    def __lt__(self, other):
        return self.bits < other.bits

    def __le__(self, other):
        return self.bits <= other.bits

    def __gt__(self, other):
        return self.bits > other.bits

    def __ge__(self, other):
        return self.bits >= other.bits
            
    def __eq__(self, other):
        return self.bits == other.bits
            
    def __neq__(self, other):
        return self.bits != other.bits

    @staticmethod
    def from_int(val, N):
        return Ket(bits=tuple((val & (1 << A)) >> A for A in reversed(range(N))))

    @property
    def int(self):
        return sum((1 << len(self.bits)-1-A) * B for A, B in enumerate(self.bits))

    @staticmethod
    def bit_reversal_permutation(N):
        seq = [0]
        for k in range(N):
            seq = [2*_ for _ in seq] + [2*_+1 for _ in seq]
        return seq

    @staticmethod
    def build_kets(N):
        return [Ket.from_int(I, N) for I in Ket.bit_reversal_permutation(N=N)]

class Counts(object):

    def __init__(
        self,
        counts,
        ):

        self.counts = counts

if __name__ == '__main__':
    
    print(Ket.from_int(56, N=9))
    print(Ket.from_int(56, N=9).int)

    
     
        
