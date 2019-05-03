import numpy as np
import re
from ..util import memoized_property
from ..core import PauliOperator, PauliString, Pauli

# ==> AIEM Connectivity (Abstracts A <-> B Connectivity Pattern) <== #

class AIEMConnectivity(object):
    
    def __init__(
        self,
        connectivity,
        ):

        self.connectivity = connectivity

        if not isinstance(self.connectivity, np.ndarray): raise RuntimeError('connectivity must be np.ndarray')
        if self.connectivity.shape != (self.N,)*2: raise RuntimeError('connectivity.shape != (N,N)')
        if self.connectivity.dtype != np.bool: raise RuntimeError('connectivity.dtype must be np.bool')

    # => Topology <= #

    @memoized_property
    def N(self):
        return self.connectivity.shape[0]

    @memoized_property
    def is_linear(self):
        M = np.zeros((self.N,)*2, dtype=np.bool) 
        for A in range(self.N-1):
            M[A,A+1] = M[A+1,A] = True
        return np.max(np.abs(M ^ self.connectivity)) == 0

    @memoized_property
    def is_cyclic(self):
        M = np.zeros((self.N,)*2, dtype=np.bool) 
        for A in range(self.N-1):
            M[A,A+1] = M[A+1,A] = True
        M[0,-1] = M[-1,0] = True
        return np.max(np.abs(M ^ self.connectivity)) == 0

    @memoized_property
    def is_all(self):
        M = np.zeros((self.N,)*2, dtype=np.bool) 
        for A in range(self.N):
            for B in range(A):
                M[A,B] = M[B,A] = True
        return np.max(np.abs(M ^ self.connectivity)) == 0

    @memoized_property
    def is_special(self):
        return not (self.is_all or self.is_linear or self.is_cyclic)

    @memoized_property
    def connectivity_str(self):
        if self.is_linear: return 'linear'
        elif self.is_cyclic: return 'cyclic'
        elif self.is_all: return 'all'
        elif self.is_special: return 'special'
        else: raise RuntimeError('Impossible')

    @memoized_property
    def ABs(self):
        ret = [] 
        for A in range(self.N):
            for B in range(self.N):
                if self.connectivity[A,B]:
                    ret.append((A,B))
        return ret

# ==> AIEMMonomer (Monomer Properties) <== #

class AIEMMonomer(AIEMConnectivity):

    def __init__(
        self,
        connectivity,
        EH,
        ET,
        EP,
        QH,
        QT,
        QP,
        MH,
        MT,
        MP,
        R0,
        ):

        AIEMConnectivity.__init__(self, connectivity=connectivity)

        self.EH = EH
        self.ET = ET
        self.EP = EP
        self.QH = QH
        self.QT = QT
        self.QP = QP
        self.MH = MH
        self.MT = MT
        self.MP = MP
        self.R0 = R0

        if not isinstance(self.EH, np.ndarray): raise RuntimeError('EH must be np.ndarray')    
        if not isinstance(self.ET, np.ndarray): raise RuntimeError('ET must be np.ndarray')    
        if not isinstance(self.EP, np.ndarray): raise RuntimeError('EP must be np.ndarray')    
        if not isinstance(self.QH, np.ndarray): raise RuntimeError('QH must be np.ndarray')    
        if not isinstance(self.QT, np.ndarray): raise RuntimeError('QT must be np.ndarray')    
        if not isinstance(self.QP, np.ndarray): raise RuntimeError('QP must be np.ndarray')    
        if not isinstance(self.MH, np.ndarray): raise RuntimeError('MH must be np.ndarray')    
        if not isinstance(self.MT, np.ndarray): raise RuntimeError('MT must be np.ndarray')    
        if not isinstance(self.MP, np.ndarray): raise RuntimeError('MP must be np.ndarray')    
        if not isinstance(self.R0, np.ndarray): raise RuntimeError('R0 must be np.ndarray')    
    
        if self.EH.shape != (self.N,): raise RuntimeError('EH.shape != (N,)')
        if self.ET.shape != (self.N,): raise RuntimeError('ET.shape != (N,)')
        if self.EP.shape != (self.N,): raise RuntimeError('EP.shape != (N,)')
        if self.QH.shape != (self.N,): raise RuntimeError('QH.shape != (N,)')
        if self.QT.shape != (self.N,): raise RuntimeError('QT.shape != (N,)')
        if self.QP.shape != (self.N,): raise RuntimeError('QP.shape != (N,)')
        if self.MH.shape != (self.N,3): raise RuntimeError('MH.shape != (N,3)')
        if self.MT.shape != (self.N,3): raise RuntimeError('MT.shape != (N,3)')
        if self.MP.shape != (self.N,3): raise RuntimeError('MP.shape != (N,3)')
        if self.R0.shape != (self.N,3): raise RuntimeError('R0.shape != (N,3)')
    
        if self.EH.dtype != np.float64: raise RuntimeError('EH.dtype must be np.float64')
        if self.ET.dtype != np.float64: raise RuntimeError('ET.dtype must be np.float64')
        if self.EP.dtype != np.float64: raise RuntimeError('EP.dtype must be np.float64')
        if self.QH.dtype != np.float64: raise RuntimeError('QH.dtype must be np.float64')
        if self.QT.dtype != np.float64: raise RuntimeError('QT.dtype must be np.float64')
        if self.QP.dtype != np.float64: raise RuntimeError('QP.dtype must be np.float64')
        if self.MH.dtype != np.float64: raise RuntimeError('MH.dtype must be np.float64')
        if self.MT.dtype != np.float64: raise RuntimeError('MT.dtype must be np.float64')
        if self.MP.dtype != np.float64: raise RuntimeError('MP.dtype must be np.float64')
        if self.R0.dtype != np.float64: raise RuntimeError('R0.dtype must be np.float64')

    @staticmethod
    def zeros_like(other):
        return AIEMMonomer(
            connectivity=np.copy(other.connectivity),
            EH=np.zeros_like(other.EH), 
            ET=np.zeros_like(other.ET), 
            EP=np.zeros_like(other.EP), 
            QH=np.zeros_like(other.QH), 
            QT=np.zeros_like(other.QT), 
            QP=np.zeros_like(other.QP), 
            MH=np.zeros_like(other.MH), 
            MT=np.zeros_like(other.MT), 
            MP=np.zeros_like(other.MP), 
            R0=np.zeros_like(other.R0), 
            )

    @staticmethod
    def copy(other):
        return AIEMMonomer(
            connectivity=np.copy(other.connectivity),
            EH=np.copy(other.EH), 
            ET=np.copy(other.ET), 
            EP=np.copy(other.EP), 
            QH=np.copy(other.QH), 
            QT=np.copy(other.QT), 
            QP=np.copy(other.QP), 
            MH=np.copy(other.MH), 
            MT=np.copy(other.MT), 
            MP=np.copy(other.MP), 
            R0=np.copy(other.R0), 
            )

    @staticmethod
    def from_ed_npzfile(
        npzfile,
        N=None,
        connectivity='all',
        zero_gauge=False,
        ):

        """ Create an AIEMMonomer object from Ed's npzfile format for AIEM
    
        Params:
            npzfile (str) - the filepath to the .npz data file.
            N (int or None) - the maximum number of monomers to use. If None,
                the full set of monomers in the data packet are used. Setting a
                finite N allows one to truncate a larger exciton model.
            connectivity (str) - A string to indicate the connection topology
                to restrict to. Allowed values are:
                    'all' - no restrictions
                    'linear' - (A,A+1) pairs (no periodicity)
                    'cyclic' - (A,A+1) pairs (periodicity)
            zero_gauge (bool) - set all monomer energies EH to zero? (helps
                improve precision in finite difference gradients)
        Return:
            (AIEMMonomer) - the resultant AIEMMonomer
        """

        dat = np.load(npzfile)

        if N is None:
            N = dat[Ekey].shape[0]

        # Connectivity Matrix
        connectivity2 = np.zeros((N,N), dtype=np.bool)
        for A in range(N):
            # Restrict connectivities
            if connectivity == 'all':
                Bs = list(range(A))
            elif connectivity == 'linear':
                Bs = [A+1] if A+1 < N else []
            elif connectivity == 'cyclic':
                Bs = [A+1] if A+1 < N else [0]
            else:
                raise RuntimeError('Invalid connectivity: %s' % connectivity)
            for B in Bs:
                connectivity2[A,B] = True
                connectivity2[B,A] = True

        prop = AIEMMonomer(
            connectivity=connectivity2,
            EH=dat['EH'][:N],
            ET=dat['ET'][:N],
            EP=dat['EP'][:N],
            QH=np.zeros((N,)),
            QT=np.zeros((N,)),
            QP=np.zeros((N,)),
            MH=dat['MH'][:N,:],
            MT=dat['MT'][:N,:],
            MP=dat['MP'][:N,:],
            R0=dat['R0'][:N,:],
            )

        if zero_gauge:
            prop.EP -= prop.EH
            prop.EH[:] = 0.0

        return prop

    @staticmethod
    def from_tc_exciton_files(
        filenames,
        charges=None,
        N=None,
        connectivity='all',
        zero_gauge=False,
        ):

        if N is None:
            N = len(filenames)

        if N > len(filenames):
            raise RuntimeError('N > len(filenames)')
    
        if charges is None:
            charges = np.zeros((N,))

        # Connectivity Matrix
        connectivity2 = np.zeros((N,N), dtype=np.bool)
        for A in range(N):
            # Restrict connectivities
            if connectivity == 'all':
                Bs = list(range(A))
            elif connectivity == 'linear':
                Bs = [A+1] if A+1 < N else []
            elif connectivity == 'cyclic':
                Bs = [A+1] if A+1 < N else [0]
            else:
                raise RuntimeError('Invalid connectivity: %s' % connectivity)
            for B in Bs:
                connectivity2[A,B] = True
                connectivity2[B,A] = True

        # Monomer properties
        EH = np.zeros((N,))
        ET = np.zeros((N,))
        EP = np.zeros((N,))
        QH = np.zeros((N,))
        QT = np.zeros((N,))
        QP = np.zeros((N,))
        MH = np.zeros((N,3))
        MT = np.zeros((N,3))
        MP = np.zeros((N,3))
        R0 = np.zeros((N,3))

        # Fixed H/P charges
        QH = np.array(charges[:N])
        QP = np.array(charges[:N])

        re_EH = re.compile(r'^Energy\s+0:\s+(\S+)\s*$')
        re_EP = re.compile(r'^Energy\s+1:\s+(\S+)\s*$')
        re_MH = re.compile(r'^Dipole\s+0:\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
        re_MT = re.compile(r'^Transition\s+Dipole\s+0\s+->\s+1:\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
        re_MP = re.compile(r'^Dipole\s+1:\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
        re_R0 = re.compile(r'^COM:\s+(\S+)\s+(\S+)\s+(\S+)\s*$')

        for A, filename in enumerate(filenames):
            if A >= N: break # Might not need all files
            lines = open(filename).readlines()
            for line in lines:
                # EH
                mobj = re.match(re_EH, line) 
                if mobj:
                    EH[A] = float(mobj.group(1))
                # EP
                mobj = re.match(re_EP, line) 
                if mobj:
                    EP[A] = float(mobj.group(1))
                # MH
                mobj = re.match(re_MH, line) 
                if mobj:
                    MH[A,0] = float(mobj.group(1))
                    MH[A,1] = float(mobj.group(2))
                    MH[A,2] = float(mobj.group(3))
                # MT
                mobj = re.match(re_MT, line) 
                if mobj:
                    MT[A,0] = float(mobj.group(1))
                    MT[A,1] = float(mobj.group(2))
                    MT[A,2] = float(mobj.group(3))
                # MP
                mobj = re.match(re_MP, line) 
                if mobj:
                    MP[A,0] = float(mobj.group(1))
                    MP[A,1] = float(mobj.group(2))
                    MP[A,2] = float(mobj.group(3))
                # COM
                mobj = re.match(re_R0, line) 
                if mobj:
                    R0[A,0] = float(mobj.group(1))
                    R0[A,1] = float(mobj.group(2))
                    R0[A,2] = float(mobj.group(3))

        monomer = AIEMMonomer(
            connectivity=connectivity2,
            EH=EH,
            ET=ET,
            EP=EP,
            QH=QH,
            QT=QT,
            QP=QP,
            MH=MH,
            MT=MT,
            MP=MP,
            R0=R0,
            )

        if zero_gauge:
            monomer.EP -= monomer.EH
            monomer.EH[:] = 0.0

        return monomer
    
# ==> AIEMMonomerGrad (Monomer Property Gradients) <== #

class AIEMMonomerGrad(AIEMConnectivity):

    def __init__(
        self,
        connectivity,
        EH,
        ET,
        EP,
        QH,
        QT,
        QP,
        MH,
        MT,
        MP,
        R0,
        ):

        AIEMConnectivity.__init__(self, connectivity=connectivity)

        self.EH = EH
        self.ET = ET
        self.EP = EP
        self.QH = QH
        self.QT = QT
        self.QP = QP
        self.MH = MH
        self.MT = MT
        self.MP = MP
        self.R0 = R0

        if not isinstance(self.EH, list): raise RuntimeError('EH must be list')    
        if not isinstance(self.ET, list): raise RuntimeError('ET must be list')    
        if not isinstance(self.EP, list): raise RuntimeError('EP must be list')    
        if not isinstance(self.QH, list): raise RuntimeError('QH must be list')    
        if not isinstance(self.QT, list): raise RuntimeError('QT must be list')    
        if not isinstance(self.QP, list): raise RuntimeError('QP must be list')    
        if not isinstance(self.MH, list): raise RuntimeError('MH must be list')    
        if not isinstance(self.MT, list): raise RuntimeError('MT must be list')    
        if not isinstance(self.MP, list): raise RuntimeError('MP must be list')    
        if not isinstance(self.R0, list): raise RuntimeError('R0 must be list')    
    
        if len(self.EH) != self.N: raise RuntimeError('len(EH) != N')
        if len(self.ET) != self.N: raise RuntimeError('len(ET) != N')
        if len(self.EP) != self.N: raise RuntimeError('len(EP) != N')
        if len(self.QH) != self.N: raise RuntimeError('len(QH) != N')
        if len(self.QT) != self.N: raise RuntimeError('len(QT) != N')
        if len(self.QP) != self.N: raise RuntimeError('len(QP) != N')
        if len(self.MH) != self.N: raise RuntimeError('len(MH) != N')
        if len(self.MT) != self.N: raise RuntimeError('len(MT) != N')
        if len(self.MP) != self.N: raise RuntimeError('len(MP) != N')
        if len(self.R0) != self.N: raise RuntimeError('len(R0) != N')

        for I in range(self.N):
    
            if not isinstance(self.EH[I], np.ndarray): raise RuntimeError('EH[I] must be np.ndarray')    
            if not isinstance(self.ET[I], np.ndarray): raise RuntimeError('ET[I] must be np.ndarray')    
            if not isinstance(self.EP[I], np.ndarray): raise RuntimeError('EP[I] must be np.ndarray')    
            if not isinstance(self.QH[I], np.ndarray): raise RuntimeError('QH[I] must be np.ndarray')    
            if not isinstance(self.QT[I], np.ndarray): raise RuntimeError('QT[I] must be np.ndarray')    
            if not isinstance(self.QP[I], np.ndarray): raise RuntimeError('QP[I] must be np.ndarray')    
            if not isinstance(self.MH[I], np.ndarray): raise RuntimeError('MH[I] must be np.ndarray')    
            if not isinstance(self.MT[I], np.ndarray): raise RuntimeError('MT[I] must be np.ndarray')    
            if not isinstance(self.MP[I], np.ndarray): raise RuntimeError('MP[I] must be np.ndarray')    
            if not isinstance(self.R0[I], np.ndarray): raise RuntimeError('R0[I] must be np.ndarray')    

            if self.EH[I].shape != (self.natom[I],3): raise RuntimeError('EH.shape != (natom[I],3)')
            if self.ET[I].shape != (self.natom[I],3): raise RuntimeError('ET.shape != (natom[I],3)')
            if self.EP[I].shape != (self.natom[I],3): raise RuntimeError('EP.shape != (natom[I],3)')
            if self.QH[I].shape != (self.natom[I],3): raise RuntimeError('QH.shape != (natom[I],3)')
            if self.QT[I].shape != (self.natom[I],3): raise RuntimeError('QT.shape != (natom[I],3)')
            if self.QP[I].shape != (self.natom[I],3): raise RuntimeError('QP.shape != (natom[I],3)')
            if self.MH[I].shape != (3,self.natom[I],3): raise RuntimeError('MH.shape != (3,natom[I],3)')
            if self.MT[I].shape != (3,self.natom[I],3): raise RuntimeError('MT.shape != (3,natom[I],3)')
            if self.MP[I].shape != (3,self.natom[I],3): raise RuntimeError('MP.shape != (3,natom[I],3)')
            if self.R0[I].shape != (3,self.natom[I],3): raise RuntimeError('R0.shape != (3,natom[I],3)')
    
            if self.EH[I].dtype != np.float64: raise RuntimeError('EH[I].dtype must be np.float64')
            if self.ET[I].dtype != np.float64: raise RuntimeError('ET[I].dtype must be np.float64')
            if self.EP[I].dtype != np.float64: raise RuntimeError('EP[I].dtype must be np.float64')
            if self.QH[I].dtype != np.float64: raise RuntimeError('QH[I].dtype must be np.float64')
            if self.QT[I].dtype != np.float64: raise RuntimeError('QT[I].dtype must be np.float64')
            if self.QP[I].dtype != np.float64: raise RuntimeError('QP[I].dtype must be np.float64')
            if self.MH[I].dtype != np.float64: raise RuntimeError('MH[I].dtype must be np.float64')
            if self.MT[I].dtype != np.float64: raise RuntimeError('MT[I].dtype must be np.float64')
            if self.MP[I].dtype != np.float64: raise RuntimeError('MP[I].dtype must be np.float64')
            if self.R0[I].dtype != np.float64: raise RuntimeError('R0[I].dtype must be np.float64')

    @staticmethod
    def zeros_like(other):
        return AIEMMonomerGrad(
            connectivity=np.copy(other.connectivity),
            EH=[np.zeros_like(_) for _ in other.EH],
            ET=[np.zeros_like(_) for _ in other.ET],
            EP=[np.zeros_like(_) for _ in other.EP],
            QH=[np.zeros_like(_) for _ in other.QH],
            QT=[np.zeros_like(_) for _ in other.QT],
            QP=[np.zeros_like(_) for _ in other.QP],
            MH=[np.zeros_like(_) for _ in other.MH],
            MT=[np.zeros_like(_) for _ in other.MT],
            MP=[np.zeros_like(_) for _ in other.MP],
            R0=[np.zeros_like(_) for _ in other.R0],
            )

    @staticmethod
    def copy(other):
        return AIEMMonomerGrad(
            connectivity=np.copy(other.connectivity),
            EH=[np.copy(_) for _ in other.EH],
            ET=[np.copy(_) for _ in other.ET],
            EP=[np.copy(_) for _ in other.EP],
            QH=[np.copy(_) for _ in other.QH],
            QT=[np.copy(_) for _ in other.QT],
            QP=[np.copy(_) for _ in other.QP],
            MH=[np.copy(_) for _ in other.MH],
            MT=[np.copy(_) for _ in other.MT],
            MP=[np.copy(_) for _ in other.MP],
            R0=[np.copy(_) for _ in other.R0],
            )

    @memoized_property
    def natom(self):
        return [_.shape[0] for _ in self.EH]

    @memoized_property
    def natom_total(self):
        return sum(self.natom)

    @staticmethod
    def from_ed_npzfile(
        npzfile,
        N=None,
        connectivity='all',
        ):

        """ Create an AIEMMonomerGrad object from Ed's npzfile format for AIEM
    
        Params:
            npzfile (str) - the filepath to the .npz data file.
            N (int or None) - the maximum number of monomers to use. If None,
                the full set of monomers in the data packet are used. Setting a
                finite N allows one to truncate a larger exciton model.
            connectivity (str) - A string to indicate the connection topology
                to restrict to. Allowed values are:
                    'all' - no restrictions
                    'linear' - (A,A+1) pairs (no periodicity)
                    'cyclic' - (A,A+1) pairs (periodicity)
        Return:
            (AIEMMonomerGrad) - the resultant AIEMMonomerGrad
        """

        dat = np.load(npzfile)

        if N is None:
            N = dat[Ekey].shape[0]

        # Connectivity Matrix
        connectivity2 = np.zeros((N,N), dtype=np.bool)
        for A in range(N):
            # Restrict connectivities
            if connectivity == 'all':
                Bs = list(range(A))
            elif connectivity == 'linear':
                Bs = [A+1] if A+1 < N else []
            elif connectivity == 'cyclic':
                Bs = [A+1] if A+1 < N else [0]
            else:
                raise RuntimeError('Invalid connectivity: %s' % connectivity)
            for B in Bs:
                connectivity2[A,B] = True
                connectivity2[B,A] = True

        return AIEMMonomerGrad(
            connectivity=connectivity2,
            EH=list(dat['dEH'][:N]),
            ET=list(dat['dET'][:N]),
            EP=list(dat['dEP'][:N]),
            MH=list(dat['dMH'][:N]),
            MT=list(dat['dMT'][:N]),
            MP=list(dat['dMP'][:N]),
            R0=list(dat['dR0'][:N]),
            )

    @staticmethod
    def from_tc_exciton_files(
        filenames,
        charges=None,
        N=None,
        connectivity='all',
        zero_gauge=False,
        ):

        if N is None:
            N = len(filenames)

        if N > len(filenames):
            raise RuntimeError('N > len(filenames)')
    
        if charges is None:
            charges = np.zeros((N,))

        # Connectivity Matrix
        connectivity2 = np.zeros((N,N), dtype=np.bool)
        for A in range(N):
            # Restrict connectivities
            if connectivity == 'all':
                Bs = list(range(A))
            elif connectivity == 'linear':
                Bs = [A+1] if A+1 < N else []
            elif connectivity == 'cyclic':
                Bs = [A+1] if A+1 < N else [0]
            else:
                raise RuntimeError('Invalid connectivity: %s' % connectivity)
            for B in Bs:
                connectivity2[A,B] = True
                connectivity2[B,A] = True

        # Monomer property gradients
        EH = []
        EP = []
        MH = []
        MT = []
        MP = []
        R0 = [] 

        re_EH = re.compile(r'^Gradient\s+0:\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
        re_EP = re.compile(r'^Gradient\s+1:\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
        re_MH = re.compile(r'^Dipole\s+([XYZ])\s+Derivative\s+0:\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
        re_MT = re.compile(r'^Transition Dipole\s+([XYZ])\s+Derivative\s+0\s+->\s+1:\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
        re_MP = re.compile(r'^Dipole\s+([XYZ])\s+Derivative\s+1:\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
        re_R0 = re.compile(r'^COM\s+Derivative:\s+(\S+)\s+(\S+)\s+(\S+)\s*$')

        for A, filename in enumerate(filenames):
            if A >= N: break # Might not need all files
            lines = open(filename).readlines()
            EH2 = []
            EP2 = []
            R02 = { 'X' : [], 'Y' : [], 'Z' : [] }
            MH2 = { 'X' : [], 'Y' : [], 'Z' : [] }
            MT2 = { 'X' : [], 'Y' : [], 'Z' : [] }
            MP2 = { 'X' : [], 'Y' : [], 'Z' : [] }
            for line in lines:
                # EH 
                mobj = re.match(re_EH, line)
                if mobj:
                    EH2.append([
                        float(mobj.group(1)),
                        float(mobj.group(2)),
                        float(mobj.group(3)),
                        ])
                # EP
                mobj = re.match(re_EP, line)
                if mobj:
                    EP2.append([
                        float(mobj.group(1)),
                        float(mobj.group(2)),
                        float(mobj.group(3)),
                        ])
                # MH
                mobj = re.match(re_MH, line)
                if mobj:
                    MH2[mobj.group(1)].append([
                        float(mobj.group(2)),
                        float(mobj.group(3)),
                        float(mobj.group(4)),
                        ])
                # MT
                mobj = re.match(re_MT, line)
                if mobj:
                    MT2[mobj.group(1)].append([
                        float(mobj.group(2)),
                        float(mobj.group(3)),
                        float(mobj.group(4)),
                        ])
                # MP
                mobj = re.match(re_MP, line)
                if mobj:
                    MP2[mobj.group(1)].append([
                        float(mobj.group(2)),
                        float(mobj.group(3)),
                        float(mobj.group(4)),
                        ])
                # R0 (Special)
                mobj = re.match(re_R0, line)
                if mobj:
                    R02['X'].append([
                        float(mobj.group(1)),
                        0.0,
                        0.0,
                        ])
                    R02['Y'].append([
                        0.0,
                        float(mobj.group(2)),
                        0.0,
                        ])
                    R02['Z'].append([
                        0.0,
                        0.0,
                        float(mobj.group(3)),
                        ])
            EH.append(np.array(EH2))
            EP.append(np.array(EP2))
            MH.append(np.array([MH2['X'], MH2['Y'], MH2['Z']]))
            MT.append(np.array([MT2['X'], MT2['Y'], MT2['Z']]))
            MP.append(np.array([MP2['X'], MP2['Y'], MP2['Z']]))
            R0.append(np.array([R02['X'], R02['Y'], R02['Z']]))

        # Definitionally zero
        natom = [_.shape[0] for _ in EH]
        ET = [np.zeros((_,3)) for _ in natom]
        QH = [np.zeros((_,3)) for _ in natom]
        QT = [np.zeros((_,3)) for _ in natom]
        QP = [np.zeros((_,3)) for _ in natom]

        monomer = AIEMMonomerGrad(
            connectivity=connectivity2,
            EH=EH,
            ET=ET,
            EP=EP,
            QH=QH,
            QT=QT,
            QP=QP,
            MH=MH,
            MT=MT,
            MP=MP,
            R0=R0,
            )

        return monomer

# => AIEM Operator in Monomer Basis <= #

class AIEMOperator(AIEMConnectivity):
    
    def __init__(
        self,
        connectivity,
        EH,
        ET,
        EP,
        VHH,
        VHT,
        VHP,
        VTH,
        VTT,
        VTP,
        VPH,
        VPT,
        VPP,
        ):

        AIEMConnectivity.__init__(self, connectivity=connectivity)

        self.EH = EH
        self.ET = ET
        self.EP = EP
        self.VHH = VHH
        self.VHT = VHT
        self.VHP = VHP
        self.VTH = VTH
        self.VTT = VTT
        self.VTP = VTP
        self.VPH = VPH
        self.VPT = VPT
        self.VPP = VPP

        if not isinstance(self.EH, np.ndarray): raise RuntimeError('EH must be np.ndarray')
        if not isinstance(self.ET, np.ndarray): raise RuntimeError('ET must be np.ndarray')
        if not isinstance(self.EP, np.ndarray): raise RuntimeError('EP must be np.ndarray')
        if not isinstance(self.VHH, np.ndarray): raise RuntimeError('VHH must be np.ndarray')
        if not isinstance(self.VHT, np.ndarray): raise RuntimeError('VHT must be np.ndarray')
        if not isinstance(self.VHP, np.ndarray): raise RuntimeError('VHP must be np.ndarray')
        if not isinstance(self.VTH, np.ndarray): raise RuntimeError('VTH must be np.ndarray')
        if not isinstance(self.VTT, np.ndarray): raise RuntimeError('VTT must be np.ndarray')
        if not isinstance(self.VTP, np.ndarray): raise RuntimeError('VTP must be np.ndarray')
        if not isinstance(self.VPH, np.ndarray): raise RuntimeError('VPH must be np.ndarray')
        if not isinstance(self.VPT, np.ndarray): raise RuntimeError('VPT must be np.ndarray')
        if not isinstance(self.VPP, np.ndarray): raise RuntimeError('VPP must be np.ndarray')

        if self.EH.shape != (self.N,): raise RuntimeError('EH.shape != (N,)')
        if self.ET.shape != (self.N,): raise RuntimeError('ET.shape != (N,)')
        if self.EP.shape != (self.N,): raise RuntimeError('EP.shape != (N,)')
        if self.VHH.shape != (self.N,)*2: raise RuntimeError('VHH.shape != (N,N)')
        if self.VHT.shape != (self.N,)*2: raise RuntimeError('VHT.shape != (N,N)')
        if self.VHP.shape != (self.N,)*2: raise RuntimeError('VHP.shape != (N,N)')
        if self.VTH.shape != (self.N,)*2: raise RuntimeError('VTH.shape != (N,N)')
        if self.VTT.shape != (self.N,)*2: raise RuntimeError('VTT.shape != (N,N)')
        if self.VTP.shape != (self.N,)*2: raise RuntimeError('VTP.shape != (N,N)')
        if self.VPH.shape != (self.N,)*2: raise RuntimeError('VPH.shape != (N,N)')
        if self.VPT.shape != (self.N,)*2: raise RuntimeError('VPT.shape != (N,N)')
        if self.VPP.shape != (self.N,)*2: raise RuntimeError('VPP.shape != (N,N)')

        if self.EH.dtype != np.float64: raise RuntimeError('EH.dtype must be np.float64')
        if self.ET.dtype != np.float64: raise RuntimeError('ET.dtype must be np.float64')
        if self.EP.dtype != np.float64: raise RuntimeError('EP.dtype must be np.float64')
        if self.VHH.dtype != np.float64: raise RuntimeError('VHH.dtype must be np.float64')
        if self.VHT.dtype != np.float64: raise RuntimeError('VHT.dtype must be np.float64')
        if self.VHP.dtype != np.float64: raise RuntimeError('VHP.dtype must be np.float64')
        if self.VTH.dtype != np.float64: raise RuntimeError('VTH.dtype must be np.float64')
        if self.VTT.dtype != np.float64: raise RuntimeError('VTT.dtype must be np.float64')
        if self.VTP.dtype != np.float64: raise RuntimeError('VTP.dtype must be np.float64')
        if self.VPH.dtype != np.float64: raise RuntimeError('VPH.dtype must be np.float64')
        if self.VPT.dtype != np.float64: raise RuntimeError('VPT.dtype must be np.float64')
        if self.VPP.dtype != np.float64: raise RuntimeError('VPP.dtype must be np.float64')
    
        if np.max(np.abs((1 - self.connectivity) * self.VHH)) != 0.0: raise RuntimeError('Nonzero masked element in VHH')
        if np.max(np.abs((1 - self.connectivity) * self.VHT)) != 0.0: raise RuntimeError('Nonzero masked element in VHT')
        if np.max(np.abs((1 - self.connectivity) * self.VHP)) != 0.0: raise RuntimeError('Nonzero masked element in VHP')
        if np.max(np.abs((1 - self.connectivity) * self.VTH)) != 0.0: raise RuntimeError('Nonzero masked element in VTH')
        if np.max(np.abs((1 - self.connectivity) * self.VTT)) != 0.0: raise RuntimeError('Nonzero masked element in VTT')
        if np.max(np.abs((1 - self.connectivity) * self.VTP)) != 0.0: raise RuntimeError('Nonzero masked element in VTP')
        if np.max(np.abs((1 - self.connectivity) * self.VPH)) != 0.0: raise RuntimeError('Nonzero masked element in VPH')
        if np.max(np.abs((1 - self.connectivity) * self.VPT)) != 0.0: raise RuntimeError('Nonzero masked element in VPT')
        if np.max(np.abs((1 - self.connectivity) * self.VPP)) != 0.0: raise RuntimeError('Nonzero masked element in VPP')

        if np.max(np.abs(self.VHH - self.VHH.T)) != 0.0: raise RuntimeError('VHH != VHH.T')
        if np.max(np.abs(self.VTT - self.VTT.T)) != 0.0: raise RuntimeError('VTT != VTT.T')
        if np.max(np.abs(self.VPP - self.VPP.T)) != 0.0: raise RuntimeError('VPP != VPP.T')
        if np.max(np.abs(self.VHT - self.VTH.T)) != 0.0: raise RuntimeError('VHT != VTH.T')
        if np.max(np.abs(self.VHP - self.VPH.T)) != 0.0: raise RuntimeError('VHP != VPH.T')
        if np.max(np.abs(self.VTP - self.VPT.T)) != 0.0: raise RuntimeError('VTP != VPT.T')

    @staticmethod
    def zeros_like(other):
        return AIEMOperator(
            connectivity=np.copy(other.connectivity),
            EH=np.zeros_like(other.EH),
            ET=np.zeros_like(other.ET),
            EP=np.zeros_like(other.EP),
            VHH=np.zeros_like(other.VHH),
            VHT=np.zeros_like(other.VHT),
            VHP=np.zeros_like(other.VHP),
            VTH=np.zeros_like(other.VTH),
            VTT=np.zeros_like(other.VTT),
            VTP=np.zeros_like(other.VTP),
            VPH=np.zeros_like(other.VPH),
            VPT=np.zeros_like(other.VPT),
            VPP=np.zeros_like(other.VPP),
            )

    @staticmethod
    def copy(other):
        return AIEMOperator(
            connectivity=np.copy(other.connectivity),
            EH=np.copy(other.EH),
            ET=np.copy(other.ET),
            EP=np.copy(other.EP),
            VHH=np.copy(other.VHH),
            VHT=np.copy(other.VHT),
            VHP=np.copy(other.VHP),
            VTH=np.copy(other.VTH),
            VTT=np.copy(other.VTT),
            VTP=np.copy(other.VTP),
            VPH=np.copy(other.VPH),
            VPT=np.copy(other.VPT),
            VPP=np.copy(other.VPP),
            )
    
    @staticmethod
    def axpby(
        a,
        x,
        b,
        y,
        ):

        if np.max(np.abs(x.connectivity ^ y.connectivity)) != 0: raise RuntimeError('x.connectivity != y.connectivity')

        return AIEMOperator(
            connectivity=x.connectivity,
            EH=a*x.EH+b*y.EH,
            ET=a*x.ET+b*y.ET,
            EP=a*x.EP+b*y.EP,
            VHH=a*x.VHH+b*y.VHH,
            VHT=a*x.VHT+b*y.VHT,
            VHP=a*x.VHP+b*y.VHP,
            VTH=a*x.VTH+b*y.VTH,
            VTT=a*x.VTT+b*y.VTT,
            VTP=a*x.VTP+b*y.VTP,
            VPH=a*x.VPH+b*y.VPH,
            VPT=a*x.VPT+b*y.VPT,
            VPP=a*x.VPP+b*y.VPP,
            )

# => AIEM Operator in Pauli Basis <= #

class AIEMPauli(AIEMConnectivity):

    def __init__(
        self,
        connectivity,
        E,
        X,
        Z,
        XX,
        XZ,
        ZX,
        ZZ,
        ):

        AIEMConnectivity.__init__(self, connectivity=connectivity)

        self.E = E
        self.X = X
        self.Z = Z
        self.XX = XX
        self.XZ = XZ
        self.ZX = ZX
        self.ZZ = ZZ

        if not isinstance(self.E, float): raise RuntimeError('E must be float')
        if not isinstance(self.X, np.ndarray): raise RuntimeError('X must be np.ndarray')
        if not isinstance(self.Z, np.ndarray): raise RuntimeError('Z must be np.ndarray')
        if not isinstance(self.XX, np.ndarray): raise RuntimeError('XX must be np.ndarray')
        if not isinstance(self.XZ, np.ndarray): raise RuntimeError('XZ must be np.ndarray')
        if not isinstance(self.ZX, np.ndarray): raise RuntimeError('ZX must be np.ndarray')
        if not isinstance(self.ZZ, np.ndarray): raise RuntimeError('ZZ must be np.ndarray')
        
        if self.X.shape != (self.N,): raise RuntimeError('X.shape != (N,)')
        if self.Z.shape != (self.N,): raise RuntimeError('Z.shape != (N,)')
        if self.XX.shape != (self.N,)*2: raise RuntimeError('XX.shape != (N,N)')
        if self.XZ.shape != (self.N,)*2: raise RuntimeError('XZ.shape != (N,N)')
        if self.ZX.shape != (self.N,)*2: raise RuntimeError('ZX.shape != (N,N)')
        if self.ZZ.shape != (self.N,)*2: raise RuntimeError('ZZ.shape != (N,N)')

        if self.X.dtype != np.float64: raise RuntimeError('X.dtype must be np.float64')
        if self.Z.dtype != np.float64: raise RuntimeError('Z.dtype must be np.float64')
        if self.XX.dtype != np.float64: raise RuntimeError('XX.dtype must be np.float64')
        if self.XZ.dtype != np.float64: raise RuntimeError('XZ.dtype must be np.float64')
        if self.ZX.dtype != np.float64: raise RuntimeError('ZX.dtype must be np.float64')
        if self.ZZ.dtype != np.float64: raise RuntimeError('ZZ.dtype must be np.float64')
        
        if np.max(np.abs((1 - self.connectivity) * self.XX)) != 0.0: raise RuntimeError('Nonzero masked element in XX')
        if np.max(np.abs((1 - self.connectivity) * self.XZ)) != 0.0: raise RuntimeError('Nonzero masked element in XZ')
        if np.max(np.abs((1 - self.connectivity) * self.ZX)) != 0.0: raise RuntimeError('Nonzero masked element in ZX')
        if np.max(np.abs((1 - self.connectivity) * self.ZZ)) != 0.0: raise RuntimeError('Nonzero masked element in ZZ')

        if np.max(np.abs(self.XX - self.XX.T)) != 0.0: raise RuntimeError('XX != XX.T')
        if np.max(np.abs(self.ZZ - self.ZZ.T)) != 0.0: raise RuntimeError('ZZ != ZZ.T')
        if np.max(np.abs(self.XZ - self.ZX.T)) != 0.0: raise RuntimeError('XZ != ZX.T')

    @staticmethod
    def zeros_like(other):
        return AIEMPauli(
            connectivity=np.copy(other.connectivity),
            E=0.0,
            X=np.zeros_like(other.X),
            Z=np.zeros_like(other.Z),
            XX=np.zeros_like(other.XX),
            XZ=np.zeros_like(other.XZ),
            ZX=np.zeros_like(other.ZX),
            ZZ=np.zeros_like(other.ZZ),
            )
    
    @staticmethod
    def copy(other):
        return AIEMPauli(
            connectivity=np.copy(other.connectivity),
            E=other.E,
            X=np.copy(other.X),
            Z=np.copy(other.Z),
            XX=np.copy(other.XX),
            XZ=np.copy(other.XZ),
            ZX=np.copy(other.ZX),
            ZZ=np.copy(other.ZZ),
            )
    
    @staticmethod
    def axpby(
        a,
        x,
        b,
        y,
        ):

        if np.max(np.abs(x.connectivity ^ y.connectivity)) != 0: raise RuntimeError('x.connectivity != y.connectivity')

        return AIEMPauli(
            connectivity=x.connectivity,
            E=a*x.E+b*y.E,
            X=a*x.X+b*y.X,
            Z=a*x.Z+b*y.Z,
            XX=a*x.XX+b*y.XX,
            XZ=a*x.XZ+b*y.XZ,
            ZX=a*x.ZX+b*y.ZX,
            ZZ=a*x.ZZ+b*y.ZZ,
            )


class AIEMUtil(object):

    @staticmethod
    def monomer_to_hamiltonian(
        monomer,
        ):

        connectivity = np.copy(monomer.connectivity)
        EH = np.copy(monomer.EH)
        ET = np.copy(monomer.ET)
        EP = np.copy(monomer.EP)
        VHH = np.zeros((monomer.N,)*2)
        VHT = np.zeros((monomer.N,)*2)
        VHP = np.zeros((monomer.N,)*2)
        VTH = np.zeros((monomer.N,)*2)
        VTT = np.zeros((monomer.N,)*2)
        VTP = np.zeros((monomer.N,)*2)
        VPH = np.zeros((monomer.N,)*2)
        VPT = np.zeros((monomer.N,)*2)
        VPP = np.zeros((monomer.N,)*2)

        tasks = [
            (monomer.QH, monomer.QH, monomer.MH, monomer.MH, VHH),
            (monomer.QH, monomer.QT, monomer.MH, monomer.MT, VHT),
            (monomer.QH, monomer.QP, monomer.MH, monomer.MP, VHP),
            (monomer.QT, monomer.QH, monomer.MT, monomer.MH, VTH),
            (monomer.QT, monomer.QT, monomer.MT, monomer.MT, VTT),
            (monomer.QT, monomer.QP, monomer.MT, monomer.MP, VTP),
            (monomer.QP, monomer.QH, monomer.MP, monomer.MH, VPH),
            (monomer.QP, monomer.QT, monomer.MP, monomer.MT, VPT),
            (monomer.QP, monomer.QP, monomer.MP, monomer.MP, VPP),
            ]

        for A, B in monomer.ABs:
            R0A = monomer.R0[A,:]
            R0B = monomer.R0[B,:]
            RAB = R0B - R0A
            DAB = np.sqrt(np.sum(RAB**2))
            for QAs, QBs, MAs, MBs, Vs in tasks:
                QA = QAs[A]
                QB = QBs[B]
                MA = MAs[A,:]
                MB = MBs[B,:]
                V = 0.0
                # Charge-Charge
                V += 1.0 * QA * QB / DAB
                # Charge-Dipole
                V += 1.0 * QA * np.sum(MA * RAB) / DAB**3
                # Dipole-Dipole
                V += 1.0 * np.sum(MA * MB) / DAB**3
                V -= 3.0 * np.sum(MA * RAB) * np.sum(MB * RAB) / DAB**5
                Vs[A,B] = V

        # Symmetrize
        VHH = 0.5 * (VHH + VHH.T)
        VTT = 0.5 * (VTT + VTT.T)
        VPP = 0.5 * (VPP + VPP.T)
        VHT = 0.5 * (VHT + VTH.T)
        VHP = 0.5 * (VHP + VPH.T)
        VTP = 0.5 * (VTP + VPT.T)
        VTH = VHT.T
        VPH = VHP.T
        VPT = VTP.T
        
        return AIEMOperator(
            connectivity=connectivity,
            EH=EH,
            ET=ET,
            EP=EP,
            VHH=VHH,
            VHT=VHT,
            VHP=VHP,
            VTH=VTH,
            VTT=VTT,
            VTP=VTP,
            VPH=VPH,
            VPT=VPT,
            VPP=VPP,
            )

    @staticmethod
    def operator_to_pauli(
        operator,
        ):

        connectivity = np.copy(operator.connectivity)
        E = 0.0
        X = np.zeros_like(operator.EH) 
        Z = np.zeros_like(operator.EH) 
        XX = np.zeros_like(operator.VHH)
        XZ = np.zeros_like(operator.VHH)
        ZX = np.zeros_like(operator.VHH)
        ZZ = np.zeros_like(operator.VHH)

        E += 1.0 * np.sum(operator.EH) / 2.0
        E += 1.0 * np.sum(operator.EP) / 2.0
        E += 0.5 * np.sum(operator.VHH) / 4.0
        E += 0.5 * np.sum(operator.VHP) / 4.0
        E += 0.5 * np.sum(operator.VPH) / 4.0
        E += 0.5 * np.sum(operator.VPP) / 4.0

        X += 1.0 * operator.ET
        X += 0.5 * np.sum(operator.VTH, 1) / 2.0
        X += 0.5 * np.sum(operator.VTP, 1) / 2.0
        X += 0.5 * np.sum(operator.VHT, 0) / 2.0
        X += 0.5 * np.sum(operator.VPT, 0) / 2.0

        Z += 1.0 * operator.EH / 2.0
        Z -= 1.0 * operator.EP / 2.0
        Z += 0.5 * np.sum(operator.VHH, 1) / 4.0
        Z += 0.5 * np.sum(operator.VHP, 1) / 4.0
        Z -= 0.5 * np.sum(operator.VPH, 1) / 4.0
        Z -= 0.5 * np.sum(operator.VPP, 1) / 4.0
        Z += 0.5 * np.sum(operator.VHH, 0) / 4.0
        Z += 0.5 * np.sum(operator.VPH, 0) / 4.0
        Z -= 0.5 * np.sum(operator.VHP, 0) / 4.0
        Z -= 0.5 * np.sum(operator.VPP, 0) / 4.0

        XX += 1.0 * operator.VTT

        XZ += operator.VTH / 2.0
        XZ -= operator.VTP / 2.0

        ZX = XZ.T

        ZZ += operator.VHH / 4.0
        ZZ -= operator.VHP / 4.0
        ZZ -= operator.VPH / 4.0
        ZZ += operator.VPP / 4.0
        ZZ = 0.5 * (ZZ + ZZ.T) # Prevent ulp loss

        return AIEMPauli(
            connectivity=connectivity,
            E=E,
            X=X,
            Z=Z,
            XX=XX,
            XZ=XZ,
            ZX=ZX,
            ZZ=ZZ,
            )
        
    @staticmethod
    def aiem_pauli_to_pauli(
        aiem,
        ):

        strings = [] 
        values = []

        strings.append(PauliString.from_string('1'))
        values.append(aiem.E)

        for A in range(aiem.N):
            strings.append(PauliString.from_string('X%d' % A))
            values.append(aiem.X[A])
            strings.append(PauliString.from_string('Z%d' % A))
            values.append(aiem.Z[A])

        for A, B in aiem.ABs:
            strings.append(PauliString.from_string('X%d*X%d' % (A,B)))
            values.append(aiem.XX[A,B])
            strings.append(PauliString.from_string('X%d*Z%d' % (A,B)))
            values.append(aiem.XZ[A,B])
            strings.append(PauliString.from_string('Z%d*X%d' % (A,B)))
            values.append(aiem.ZX[A,B])
            strings.append(PauliString.from_string('Z%d*Z%d' % (A,B)))
            values.append(aiem.ZZ[A,B])

        strings = tuple(strings)
        values = np.array(values)

        pauli = Pauli(strings, values) 
    
        return pauli

    @staticmethod
    def pauli_to_aiem_pauli(
        pauli,
        ):

        if pauli.max_order > 2: raise RuntimeError('AIEMPauli is at most order 2')

        canonical = pauli.canonical

        N = canonical.N
        E = canonical['1']
        X = np.array([canonical['X%d' % _] for _ in range(N)])
        Z = np.array([canonical['Z%d' % _] for _ in range(N)])

        XX = np.zeros((N,N))
        XZ = np.zeros((N,N))
        ZX = np.zeros((N,N))
        ZZ = np.zeros((N,N))

        indices2 = canonical.indices(order=2)
        print(indices2)
        
        return AIEMPauli(
            connectivity=connectivity,
            E=E,
            X=X,
            Z=Z,
            XX=XX,
            XZ=XZ,
            ZX=ZX,
            ZZ=ZZ,
            )
        
        
        
            
if __name__ == '__main__':

    monomer = AIEMMonomer.from_tc_exciton_files(
        filenames=['../data/bchl-a-8-stack/tc/%d/exciton.dat' % _ for _ in range(1,8+1)],
        )    
    monomer_grad = AIEMMonomerGrad.from_tc_exciton_files(
        filenames=['../data/bchl-a-8-stack/tc/%d/exciton.dat' % _ for _ in range(1,8+1)],
        )    

    aiem_hamiltonian = AIEMUtil.monomer_to_hamiltonian(monomer)
    print(aiem_hamiltonian.VHH)

    aiem_pauli = AIEMUtil.operator_to_pauli(aiem_hamiltonian)
    print(aiem_pauli.ZZ)
