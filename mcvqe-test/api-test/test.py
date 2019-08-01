import mcvqe

if __name__ == '__main__':

    results = mcvqe.run_mcvqe(
        backend_name='quasar',
        )

    print(results['fci_E'])
    print(type(results['fci_E']))
    print(results['fci_O'])
    print(type(results['fci_O']))
    print(results['ref_E'])
    print(type(results['ref_E']))
