import numpy as np
    
class ParameterGroup(object):

    @property
    def nparam(self):
        raise NotImplementedError

    @property
    def nraw(self):
        raise NotImplementedError

    def compute_raw(self, params):
        raise NotImplementedError

    def compute_chain_rule1(self, params, Graw):
        raise NotImplementedError

    @property
    def active_raw(self):
        raise NotImplementedError

class FixedParameterGroup(ParameterGroup):

    def __init__(
        self,
        params_raw,
        ):

        self._params_raw = params_raw

    @property
    def nparam(self):
        return 0

    @property
    def nraw(self):
        return len(self._params_raw)

    def compute_raw(self, params):
        return self._params_raw.copy()

    def compute_chain_rule1(self, params, Graw):
        return np.zeros((0,))
        
    @property
    def active_raw(self):
        return [False]*self.nraw

class IdentityParameterGroup(ParameterGroup):

    def __init__(
        self,
        nparam,
        ):

        self._nparam = nparam

    @property
    def nparam(self):
        return self._nparam

    @property
    def nraw(self):
        return self._nparam

    def compute_raw(self, params):
        return params.copy()

    def compute_chain_rule1(self, params, Graw):
        return Graw.copy()

    @property
    def active_raw(self):
        return [True]*self.nraw

class LinearParameterGroup(ParameterGroup):

    def __init__(
        self,
        transform,
        ):

        self.transform = transform

        if not isinstance(self.transform, np.ndarray): raise RuntimeError('transform must be np.ndarray')
        if self.transform.ndim != 2: raise RuntimeError('transform must be shape (nraw, nparam)')

    @property
    def nparam(self):
        return self.transform.shape[1]

    @property
    def nraw(self):
        return self.transform.shape[0]

    def compute_raw(self, params):
        return np.dot(self.transform, params)

    def compute_chain_rule1(self, params, Graw):
        return np.dot(self.transform.T, Graw)

    @property
    def active_raw(self):
        return [True]*self.nraw
    
class CompositeParameterGroup(ParameterGroup):

    def __init__(
        self,
        groups,
        ):

        self.groups = groups

    @property
    def nparam(self):
        return sum(_.nparam for _ in self.groups)

    @property
    def nraw(self):
        return sum(_.nraw for _ in self.groups)

    def compute_raw(self, params):
        raws = []
        offset = 0
        for group in self.groups:
            raws.append(group.compute_raw(params[offset:offset+group.nparam]))
            offset += group.nparam
        return np.hstack(raws)
        
    def compute_chain_rule1(self, params, Graw):
        Gs = []
        offset = 0
        offset_raw = 0
        for group in self.groups:
            Gs.append(group.compute_chain_rule1(params[offset:offset+group.nparam], Graw[offset_raw:offset_raw+group.nraw]))
            offset += group.nparam
            offset_raw += group.nraw
        return np.hstack(Gs)
        
    @property
    def active_raw(self):
        active = []
        for group in self.groups:
            active += group.active_raw
        return active
        
