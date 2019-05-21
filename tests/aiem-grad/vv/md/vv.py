class VV(object):

    def __init__(
        self,
        dt,
        M,
        X,
        P,
        F,
        V,
        ):

        self.dt = dt
        self.M = np.copy(M)
        self.X = np.copy(X)
        self.P = np.copy(P)
        self.F = np.copy(F)
        self.V = V

        self.I = 0
    
    @property
    def step(
        self,
        F,
        V,
        ):

        self.X = self.Xnew
        self.P = self.P + 0.5 * (self.F + F) * self.dt
        self.F = np.copy(F)
        self.V = V
        
    @property
    def Xnew(self):
        return self.X + (self.P / self.M) * self.dt + 0.5 * (self.F / self.M) * self.dt**2

    @property
    def t(self):
        return self.I * self.dt

    @property
    def K(self):
        return 0.5 * np.sum(self.P**2 / self.M)

    @property
    def E(self):
        return self.K + self.V

    @property
    def T(self):
        return 2.0 * self.K / self.DOF

    @property
    def DOF(self):
        return self.M.size
