from .molecule import Molecule

class AIEM_LOT(object):

    def __init__(
        self,
        molecules,
        aiem,
        connectivity,
        tcjobdir='tc',
        tccommand='terachem input.dat >& output.dat',
        gradient_kwargs={},
        ):

        self.molecules = molecules
        self.aiem = aiem
        self.connectivity = connectivity
        self.tcjobdir = tcjobdir
        self.tccommand = tccommand
        self.gradient_kwargs = gradient_kwargs

    def update_xyz(self, xyz):

        # New Molecules
        molecules = []
        offset = 0
        for molecule in self.molecules:
            molecules.append(Molecule(
                symbols=molecule.symbols,
                xyz=xyz[offset:offset+molecule.natom,:],    
                ))
            offset += molecule.natom

        # geom.xyz for new TC Jobs
        for A, molecule in enumerate(molecules):
            molecule.save_xyz_file(
                filenames='%s/%d/geom.xyz' % (self.tcjobdir, A),
                format_string='%-2s %24.16E %24.16E %24.16E\n',
                )

        # New TC jobs
        for A in range(len(molecules)):
            ret = os.getcwd()
            os.chdir('%s/%d' % (self.tcjobdir, A))
            # os.system(self.tccommand)
            os.chdir(ret)

        # New AIEMMonomer
        aiem_monomer = quasar.AIEMMonomer.from_tc_exciton_files(
            filenames=['%s/%d/exciton.dat' for A in range(aiem.N)],
            N=aiem.N,
            connectivity=self.connectivity,
            )
                
        # New AIEMMonomerGrad
        aiem_monomer_grad = quasar.AIEMMonomerGrad.from_tc_exciton_files(
            filenames=['%s/%d/exciton.dat' for A in range(aiem.N)],
            N=aiem.N,
            connectivity=self.connectivity,
            )

        # New AIEM
        aiem = quasar.AIEM(self.aiem.options.copy().set_values({
            'aiem_monomer' : aiem_monomer,
            'aiem_monomer_grad' : aiem_monomer_grad,
            }))
        aiem.compute_energy()

        return AIEM_LOT(
            molecules=molecules,
            aiem=aiem,
            connectivity=self.connectivity,
            tcjobdir=self.tcjobdir,
            tccommand=self.tccommand,
            gradient_kwargs=self.gradient_kwargs,
            )

    def molecule(self):

        symbols = []
        xyzs = []
        for molecule in self.molecules:
            symbols += molecule.symbols
            xyzs += molecule.xyz

        return Molecule(
            symbols=symbols,
            xyz=np.vstack(xyzs)

    def compute_energy(self, I=0):

        return self.aiem.vqe_total_E[I=I]

    def compute_gradient(self, I=0):
        
        return self.aiem.compute_vqe_gradient(I=I, **self.gradient_kwargs)
