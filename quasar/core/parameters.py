import numpy as np
    
class ParameterGroup(object):

    @property
    def nparam(self):
        raise NotImplemented

    @property
    def nraw(self):
        raise NotImplemented

    def compute_raw(self, params):
        raise NotImplemented

    def compute_chain_rule1(self, params, Graw):
        raise NotImplemented

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
        
        
