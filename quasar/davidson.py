import numpy as np

class Davidson(object):

    def __init__(
        self,
        nstate,
        nsubspace,
        convergence_threshold=1.0E-7,
        condition_threshold=0.0,
        dtype=np.complex128,
        ):

        self.nstate = nstate
        self.nsubspace = nsubspace 
        self.convergence_threshold = convergence_threshold
        self.condition_threshold = condition_threshold
        self.dtype = dtype

        if self.nsubspace < self.nstate:
            raise RuntimeError('nsubspace < nstate')

        self.A = np.zeros((0,0), dtype=self.dtype) 
        self.S = np.zeros((0,0), dtype=self.dtype) 

        self.bs = []
        self.Abs = []
        self.Sbs = []

    def __str__(self):
        s = 'Davidson:\n'
        s += '  %-21s = %11d\n' % ('nstate', self.nstate)
        s += '  %-21s = %11d\n' % ('nsubspace', self.nsubspace)
        s += '  %-21s = %11.3E\n' % ('convergence_threshold', self.convergence_threshold)
        s += '  %-21s = %11.3E\n' % ('condition_threshold', self.condition_threshold)
        s += '  %-21s = %11s\n' % ('dtype', self.dtype)
        return s
    
    @property
    def max_rnorm(self):
        return np.max(self.rnorms)

    @property
    def is_converged(self):
        return self.max_rnorm <= self.convergence_threshold

    def add_vectors(
        self,
        bs,
        Abs,
        Sbs,
        ):

        if len(bs) != len(Abs) or len(bs) != len(Sbs):
            raise RuntimeError('bs, Abs, Sbs must all be same size')

        nold = len(self.bs)
        nnew = len(bs)
        ntot = nold + nnew

        if ntot > self.nsubspace:
            raise RuntimeError('number of new vectors exceeds nsubspace')

        self.bs  += bs
        self.Abs += Abs
        self.Sbs += Sbs

        A = np.zeros((ntot, ntot), dtype=self.dtype)
        S = np.zeros((ntot, ntot), dtype=self.dtype)
        A[:nold, :nold] = self.A
        S[:nold, :nold] = self.S

        for index in range(nold, ntot):
            S[index, index] = np.sum(self.bs[index].conj() * self.Sbs[index])
            A[index, index] = np.sum(self.bs[index].conj() * self.Abs[index])

        for index1 in range(ntot):
            for index2 in range(ntot):
                if index1 >= index2: continue
                if index1 < nold and index2 < nold: continue
                if S[index2, index2] > S[index1, index1]:
                    Sval = np.sum(self.bs[index1].conj() * self.Sbs[index2])
                    Aval = np.sum(self.bs[index1].conj() * self.Abs[index2])
                else:
                    Sval = np.sum(self.bs[index2].conj() * self.Sbs[index1]).conj()
                    Aval = np.sum(self.bs[index2].conj() * self.Abs[index1]).conj()
                S[index1, index2] = Sval
                S[index2, index1] = Sval.conj() 
                A[index1, index2] = Aval
                A[index2, index1] = Aval.conj() 
        
        self.A = A
        self.S = S

        s, U = np.linalg.eigh(self.S)
        mask = s > self.condition_threshold * np.max(np.abs(s))
        X = U[:, mask] * np.outer(np.ones((ntot,), dtype=self.dtype), s[mask]**(-0.5))
        A2 = np.dot(np.dot(X.conj().T, self.A), X)
        a, V = np.linalg.eigh(A2)
        C = np.dot(X, V)

        self.evals = []
        self.evecs = []
        for index in range(min(self.nstate, ntot)):
            self.evals.append(a[index])
            v = np.zeros_like(self.bs[0])
            for k in range(ntot):
                v += C[k, index] * self.bs[k]
            self.evecs.append(v)

        self.rs = []
        self.rnorms = []
        for index in range(min(self.nstate, ntot)):
            v = -self.evals[index] * self.evecs[index]
            for k in range(ntot):
                v += C[k, index] * self.Abs[k]
            self.rs.append(v)
            self.rnorms.append(np.sqrt(np.abs(np.sum(v.conj() * v))))

        self.gs = []
        self.hs = []
        for index in range(min(self.nstate, ntot)):
            if self.rnorms[index] < self.convergence_threshold: continue
            self.gs.append(self.rs[index])
            self.hs.append(self.evals[index])

        return self.gs, self.hs
        
    def add_preconditioned(
        self,
        ds,
        ):
    
        nold = len(self.bs)
        nnew = len(ds)
        ntot = nold + nnew
        if (ntot > self.nsubspace):
            self.A = np.zeros((0, 0), dtype=self.dtype)
            self.S = np.zeros((0, 0), dtype=self.dtype)
            self.bs = []
            self.Abs = []
            self.Sbs = []
            self.cs = self.evecs
        else:
            self.cs = ds

        return self.cs

def run_davidson(
    Ab_method,
    Adiag,
    nstate=1,
    nguess_per_state=1,
    nsubspace_per_state=5,
    maxiter=100,
    print_level=1,
    convergence_threshold=1.0E-7,
    condition_threshold=0.0,
    dtype=np.complex128,
    ):

    nguess = nstate * nguess_per_state
    nsubspace = nstate * nsubspace_per_state

    dav = Davidson(
        nstate=nstate,
        nsubspace=nsubspace,
        convergence_threshold=convergence_threshold,
        condition_threshold=condition_threshold,
        dtype=dtype,
        )
    if print_level:
        print(dav)

    I = np.argsort(Adiag)
    
    bs = []
    for index in range(nguess):
        b = np.zeros(Adiag.shape, dtype=dtype)
        b[I[index]] = 1.0
        bs.append(b)

    import time
    
    if print_level:
        print('%-4s: %11s %11s %11s %11s' % ('Iter', 'Max Rnorm','N sigma', 'N subspace', 'Time [s]'))
    converged = False
    start = time.time()
    for iteration in range(maxiter):
        
        Abs = [Ab_method(b) for b in bs]

        rs, es = dav.add_vectors(bs=bs, Abs=Abs, Sbs=bs)

        stop = time.time()
    
        if print_level:
            print('%-4d: %11.3E %11d %11d %11.3E' % (iteration, dav.max_rnorm, len(bs), len(dav.bs), stop - start)) 

        start = stop

        if dav.is_converged:
            converged = True
            break

        ds = [-r / (Adiag - e) for r, e in zip(rs, es)]

        bs = dav.add_preconditioned(ds)

    if print_level:
        print('')
        if converged:
            print('Davidson Converged')
        else:
            print('Davidson Failed')
        print('')

    return dav.evals, dav.evecs
