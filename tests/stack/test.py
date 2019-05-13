import numpy as np
import quasar

if __name__ == '__main__':

    datapath = '../../data/aiem/bchl-a-8-stack/tc'
    filenames = ['%s/%d/exciton.dat' % (datapath, _) for _ in range(1, 8+1)]

    aiem_monomer = quasar.AIEMMonomer.from_tc_exciton_files(
        filenames=filenames,
        N=8,
        connectivity='all',
        )

    aiem_monomer.save_ed_npzfile('exciton.npz')
