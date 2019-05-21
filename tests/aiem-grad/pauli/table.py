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

    dev = np.zeros((7,))
    for g1, g2 in zip(grad1, grad2):
        dev[0] = max(dev[0], np.max(np.abs(g1.E - g2.E)))
        dev[1] = max(dev[1], np.max(np.abs(g1.X - g2.X)))
        dev[2] = max(dev[2], np.max(np.abs(g1.Z - g2.Z)))
        dev[3] = max(dev[3], np.max(np.abs(g1.XX - g2.XX)))
        dev[4] = max(dev[4], np.max(np.abs(g1.XZ - g2.XZ)))
        dev[5] = max(dev[5], np.max(np.abs(g1.ZX - g2.ZX)))
        dev[6] = max(dev[6], np.max(np.abs(g1.ZZ - g2.ZZ)))
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

    terms = [
        r'$\gamma_{\mathcal{E}}^{\Theta}$', 
        r'$\gamma_{\mathcal{X}_{A}}^{\Theta}$', 
        r'$\gamma_{\mathcal{Z}_{A}}^{\Theta}$', 
        r"$\Gamma_{\mathcal{XX}_{AA'}}^{\Theta}$", 
        r"$\Gamma_{\mathcal{XZ}_{AA'}}^{\Theta}$", 
        r"$\Gamma_{\mathcal{ZX}_{AA'}}^{\Theta}$", 
        r"$\Gamma_{\mathcal{ZZ}_{AA'}}^{\Theta}$", 
        ]

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
