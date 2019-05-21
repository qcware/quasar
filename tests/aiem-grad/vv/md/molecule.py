import re
import numpy as np
import masses

ang_per_bohr = 0.52917724924 # TC
bohr_per_ang = 1.0 / ang_per_bohr
    
class Molecule(object):
    
    def __init__(
        self,
        symbols,
        xyz,
        ):

        self.symbols = symbols
        self.xyz = xyz

    @property
    def natom(self):
        return len(self.symbols)

    @staticmethod 
    def from_xyz_file(
        filename,
        scale=bohr_per_ang,
        ):    

        lines = open(filename).readlines()[2:]

        symbols = []
        xyz = np.zeros((len(lines),3))
    
        for A, line in enumerate(lines):
            mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
            symbols.append(mobj.group(1))
            xyz[A,0] = scale * float(mobj.group(2))
            xyz[A,1] = scale * float(mobj.group(3))
            xyz[A,2] = scale * float(mobj.group(4))

        return Molecule(symbols=symbols, xyz=xyz)

    def xyz_string(
        self,
        scale=ang_per_bohr,
        comment='',
        formatstr='%-2s %14.8f %14.8f %14.8f\n',
        ):

        s = ''
        s += '%d\n' % (self.natom)
        s += '%s\n' % (comment)     
        for A in range(self.natom):
            s += formatstr % (
                self.symbols[A], 
                scale * self.xyz[A,0],
                scale * self.xyz[A,1],
                scale * self.xyz[A,2],
                )
        return s

    def save_xyz_file(
        self,
        filename,
        mode='w',
        **kwargs):
        
        open(filename, mode).write(self.xyz_string(**kwargs))
            

    @staticmethod
    def build_masses(
        molecule,
        ):

        M = [masses.mass_table[symbol.upper()] for symbol in molecule.symbols]
        return np.array([M]*3).T

if __name__ == '__main__':

    molecule = Molecule.from_xyz_file('../../../../data/aiem/bchl-a-2-stack-fd/geom/1.xyz')
    print(molecule.natom)
    print(molecule.symbols)
    print(molecule.xyz)
    print(molecule.xyz_string())
    
    print(Molecule.build_masses(molecule))
