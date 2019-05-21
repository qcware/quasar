import numpy as np

def latex_float(f):
    if f == 0.0: return '$0$'
    float_str = "{0:.1e}".format(f)
    base, exponent = float_str.split("e")
    return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))

def compute_deviation(
    grad1,
    grad2,
    ):

    dg = grad1 - grad2
    dev = np.array([np.max(np.abs(dg[3*A:3*A+3])) for A in range(5)])
    return dev

if __name__ == '__main__':

    datFF = np.load('data-vqeFalse-cisFalse.npz')
    datFT = np.load('data-vqeFalse-cisTrue.npz')
    datTF = np.load('data-vqeTrue-cisFalse.npz')
    datTT = np.load('data-vqeTrue-cisTrue.npz')
    
    devs = {
        'vqe_FF'  : compute_deviation(datFF['G_vqe'], datFF['G_vqe_fd']),
        'vqe_FT'  : compute_deviation(datFT['G_vqe'], datFT['G_vqe_fd']),
        'vqe_TF'  : compute_deviation(datTF['G_vqe'], datTF['G_vqe_fd']),
        'vqe_TT'  : compute_deviation(datTT['G_vqe'], datTT['G_vqe_fd']),
        'fci'     : compute_deviation(datTT['G_fci'], datTT['G_fci_fd']),
        'cis'     : compute_deviation(datTT['G_cis'], datTT['G_cis_fd']),
        'fci-vqe' : compute_deviation(datTT['G_fci'], datTT['G_vqe']),
        'fci-cis' : compute_deviation(datTT['G_fci'], datTT['G_cis']),
        'vqe-cis' : compute_deviation(datTT['G_vqe'], datTT['G_cis']),
        }

    symbols = ['Mg', 'O', 'N', 'C', 'H']
    monomers = [0, 0, 0, 1, 1]
    atoms = [20, 41, 16, 22, 31]

    terms = [r'%s$_{%s}^{%d}$' % (symbol, 'A' if monomer == 0 else 'B', atom) for symbol, monomer, atom in zip(symbols, monomers, atoms)]

    methods = {
        'vqe_FF'  : 'VQE(N,N)',
        'vqe_FT'  : 'VQE(N,Y)',
        'vqe_TF'  : 'VQE(Y,N)',
        'vqe_TT'  : 'VQE(Y,Y)',
        'fci'     : 'FCI',
        'cis'     : 'CIS',
        'fci-vqe' : 'FCI-VQE',
        'fci-cis' : 'FCI-CIS',
        'vqe-cis' : 'VQE-CIS',
    }

    s = '' 
    s += '%20s ' % ('Method')
    for term in terms:
        s += '& %40s ' % (term)
    s += '\\\\\n' 
    for key, value in devs.items():
        s += '%20s ' % (methods[key])
        for delta in value:
            s += '& %40s ' % (latex_float(delta))
        s += '\\\\\n' 
    print(s)
