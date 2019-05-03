import hammer
import collections
import numpy as np

def reference_matrices():
    
    """ OrderedDict of known matrices """

    matrices = collections.OrderedDict()
    matrices['I'] = np.array([[1.0, 0.0], [0.0, 1.0]])
    matrices['X'] = np.array([[0.0, 1.0], [1.0, 0.0]])
    matrices['Y'] = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    matrices['Z'] = np.array([[1.0, 0.0], [0.0, -1.0]])
    matrices['S'] = np.array([[1.0, 0.0], [0.0, 1.0j]])
    matrices['T'] = np.array([[1.0, 0.0], [0.0, np.exp(+0.25j * np.pi)]])
    matrices['H'] = 1.0 / np.sqrt(2.0) * np.array([[1.0, 1.0], [1.0, -1.0]])
    matrices['Rx2'] = 1.0 / np.sqrt(2.0) * np.array([[1.0, 1.0j], [1.0j, 1.0]])
    matrices['Rx2T'] = matrices['Rx2'].conj()
    
    for A in ['I', 'X', 'Y', 'Z']:
        for B in ['I', 'X', 'Y', 'Z']:
            matrices[A + B] = np.kron(matrices[A], matrices[B])

    matrices['CX'] = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        ])
    matrices['CY'] = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0j],
        [0.0, 0.0, 1.0j, 0.0],
        ])
    matrices['CZ'] = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        ])
    matrices['SWAP'] = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        ])
    return matrices
     
def test_matrix():

    """ Test explicit matrix definitions in Matrix """

    refs = reference_matrices()

    # Dict of test results
    checks = collections.OrderedDict()
    for key, val in refs.items():
        x = np.max(np.abs(val - hammer.Matrix.__dict__[key]))
        checks[key] = (x, x < 1.0E-16)

    # => Print and Return Results <= #

    # Print out detailed test results
    print('Test Matrix:')
    for key, value in checks.items():
        print(('%-20s: %11.3E %s' % (
            key,
            value[0],
            'OK' if value[1] else 'BAD',
            )))

    # Return True if all tests passed else False
    return all(x[1] for x in list(checks.values()))
    
def test_explicit_gates():

    """ Test explicit gates """

    refs = reference_matrices()

    gates = collections.OrderedDict()
    for key in ['I', 'X', 'Y', 'Z', 'S', 'T', 'H', 'Rx2', 'Rx2T']: 
        gates[key] = (1, refs[key], key, [key])
    gates['CX'] = (2, refs['CX'], 'CNOT', ['@', 'X'])
    gates['CNOT'] = (2, refs['CX'], 'CNOT', ['@', 'X'])
    gates['CY'] = (2, refs['CY'], 'CY', ['@', 'Y'])
    gates['CZ'] = (2, refs['CZ'], 'CZ', ['@', 'Z'])
    gates['SWAP'] = (2, refs['SWAP'], 'SWAP', ['X', 'X'])

    checks = collections.OrderedDict()
    for key, val in gates.items():
        gate = hammer.Gate.__dict__[key]
        checks[key] = (
            gate.N == val[0],
            np.max(np.abs(gate.U - val[1])) < 1.0E-16,
            gate.name == val[2],
            gate.ascii_symbols == val[3],
            len(gate.params) == 0,
            )
    
    # Print out detailed test results
    print('Test Explicit Gates: (N, U, name, ascii_symbols, params)')
    for key, value in checks.items():
        print(('%-20s: %r %s' % (
            key, value, 'OK' if all(value) else 'BAD',
            )))

    return all(all(x) for x in list(checks.values()))


def test_gates():

    I = hammer.Gate.I
    print(I)
    print(I.U)

def test_ghz_5():

    circuit = hammer.Circuit(N=5)
    circuit.add_gate(T=0, key=0, gate=hammer.Gate.H)   
    circuit.add_gate(T=1, key=(0,1), gate=hammer.Gate.CNOT)
    circuit.add_gate(T=2, key=(1,2), gate=hammer.Gate.CNOT)
    circuit.add_gate(T=3, key=(2,3), gate=hammer.Gate.CNOT)
    circuit.add_gate(T=4, key=3, gate=hammer.Gate.H)   
    circuit.add_gate(T=4, key=4, gate=hammer.Gate.H)   
    circuit.add_gate(T=5, key=(4,3), gate=hammer.Gate.CNOT)
    circuit.add_gate(T=6, key=3, gate=hammer.Gate.H)   
    circuit.add_gate(T=6, key=4, gate=hammer.Gate.H)   

    print(circuit)

    print(circuit.compressed())
    
    print(circuit.Ts)

def test_linear_4():

    circuit = hammer.Circuit(N=4)
    circuit.add_gate(T=1, key=(1,2), gate=hammer.Gate.SO4(A=0.0, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0))
    circuit.add_gate(T=1, key=(3,0), gate=hammer.Gate.SO4(A=0.0, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0))
    circuit.add_gate(T=0, key=(0,1), gate=hammer.Gate.SO4(A=0.0, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0))
    circuit.add_gate(T=0, key=(2,3), gate=hammer.Gate.SO4(A=0.0, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0))
    print(circuit) 
    print(circuit.param_str)
        

if __name__ == '__main__':
    
    test_matrix()
    test_explicit_gates()

