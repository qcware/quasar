import numpy as np
    
class ParameterGroup(object):

    @property
    def nparameter(self):
        raise NotImplementedError

    @property
    def nraw(self):
        raise NotImplementedError

    def compute_raw(self, parameters):
        raise NotImplementedError

    def compute_chain_rule1(self, parameters, Graw):
        raise NotImplementedError

    @property
    def active_raw(self):
        raise NotImplementedError

class FixedParameterGroup(ParameterGroup):

    def __init__(
        self,
        parameters_raw,
        ):

        self._parameters_raw = parameters_raw

    @property
    def nparameter(self):
        return 0

    @property
    def nraw(self):
        return len(self._parameters_raw)

    def compute_raw(self, parameters):
        return self._parameters_raw.copy()

    def compute_chain_rule1(self, parameters, Graw):
        return np.zeros((0,))
        
    @property
    def active_raw(self):
        return [False]*self.nraw

class IdentityParameterGroup(ParameterGroup):

    def __init__(
        self,
        nparameter,
        ):

        self._nparameter = nparameter

    @property
    def nparameter(self):
        return self._nparameter

    @property
    def nraw(self):
        return self._nparameter

    def compute_raw(self, parameters):
        return parameters.copy()

    def compute_chain_rule1(self, parameters, Graw):
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
        if self.transform.ndim != 2: raise RuntimeError('transform must be shape (nraw, nparameter)')

    @property
    def nparameter(self):
        return self.transform.shape[1]

    @property
    def nraw(self):
        return self.transform.shape[0]

    def compute_raw(self, parameters):
        return np.dot(self.transform, parameters)

    def compute_chain_rule1(self, parameters, Graw):
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
    def nparameter(self):
        return sum(_.nparameter for _ in self.groups)

    @property
    def nraw(self):
        return sum(_.nraw for _ in self.groups)

    def compute_raw(self, parameters):
        raws = []
        offset = 0
        for group in self.groups:
            raws.append(group.compute_raw(parameters[offset:offset+group.nparameter]))
            offset += group.nparameter
        return np.hstack(raws)
        
    def compute_chain_rule1(self, parameters, Graw):
        Gs = []
        offset = 0
        offset_raw = 0
        for group in self.groups:
            Gs.append(group.compute_chain_rule1(parameters[offset:offset+group.nparameter], Graw[offset_raw:offset_raw+group.nraw]))
            offset += group.nparameter
            offset_raw += group.nraw
        return np.hstack(Gs)
        
    @property
    def active_raw(self):
        active = []
        for group in self.groups:
            active += group.active_raw
        return active
        
