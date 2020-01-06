import quasar
import vulcan
import time
import numpy as np

# => Circuit Library <= #
    
def randomized_so4(
    nqubit,
    depth=3,
    ):

    if nqubit == 1:
        return quasar.Circuit().Ry(0, theta=2.0 * np.pi * (np.random.rand() - 0.5))

    circuit = quasar.Circuit()
    for index in range(depth):
        qubits = list(range(nqubit))
        np.random.shuffle(qubits)
        for k in range(0, nqubit-1, 2):
            qubit0 = qubits[k + 0] 
            qubit1 = qubits[k + 1] 
            # Not Haar random, but random enough for testing
            thetas = 2.0 * np.pi * (np.random.rand(6) - 0.5)
            circuit.Ry(qubit0, theta=thetas[0])
            circuit.Ry(qubit1, theta=thetas[1])
            circuit.CX(qubit0, qubit1)
            circuit.Ry(qubit0, theta=thetas[2])
            circuit.Ry(qubit1, theta=thetas[3])
            circuit.CX(qubit0, qubit1)
            circuit.Ry(qubit0, theta=thetas[4])
            circuit.Ry(qubit1, theta=thetas[5])
    return circuit

def randomized_su4(
    nqubit,
    depth=3,
    ):

    if nqubit == 1:
        thetas = 2.0 * np.pi * (np.random.rand(3) - 0.5)
        return quasar.Circuit().u3(0, theta=thetas[ 0], phi=thetas[ 1], lam=thetas[ 2])

    circuit = quasar.Circuit()
    for index in range(depth):
        qubits = list(range(nqubit))
        np.random.shuffle(qubits)
        for k in range(0, nqubit-1, 2):
            qubit0 = qubits[k + 0] 
            qubit1 = qubits[k + 1] 
            # Not Haar random, but random enough for testing
            thetas = 2.0 * np.pi * (np.random.rand(24) - 0.5)
            circuit.u3(qubit0, theta=thetas[ 0], phi=thetas[ 1], lam=thetas[ 2])
            circuit.u3(qubit1, theta=thetas[ 3], phi=thetas[ 4], lam=thetas[ 5])
            circuit.CX(qubit0, qubit1)
            circuit.u3(qubit0, theta=thetas[ 6], phi=thetas[ 7], lam=thetas[ 8])
            circuit.u3(qubit1, theta=thetas[ 9], phi=thetas[10], lam=thetas[11])
            circuit.CX(qubit0, qubit1)
            circuit.u3(qubit0, theta=thetas[12], phi=thetas[13], lam=thetas[14])
            circuit.u3(qubit1, theta=thetas[15], phi=thetas[16], lam=thetas[17])
            circuit.CX(qubit0, qubit1)
            circuit.u3(qubit0, theta=thetas[18], phi=thetas[19], lam=thetas[20])
            circuit.u3(qubit1, theta=thetas[21], phi=thetas[22], lam=thetas[23])
    return circuit

def randomized_su4_grad(
    nqubit,
    depth=3,
    ):

    circuit = randomized_su4(
        nqubit=nqubit,
        depth=depth,
        )

    circuit2 = quasar.Circuit()
    for key, gate in circuit.gates.items():
        times, qubits = key
        if gate.name == 'u3':
            gate = quasar.Gate.U1(gate.operator)
        circuit2.add_gate(gate, times=times, qubits=qubits)

    circuitA = quasar.Circuit()
    for qubit in range(nqubit):
        if qubit % 3 == 0:
            circuitA.Rx(qubit, theta=2.0*np.pi*(np.random.rand() - 0.5))
        elif qubit % 3 == 1:
            circuitA.Ry(qubit, theta=2.0*np.pi*(np.random.rand() - 0.5))
        elif qubit % 3 == 2:
            circuitA.Rz(qubit, theta=2.0*np.pi*(np.random.rand() - 0.5))
        else:
            raise RuntimeError('Impossible')

    circuitB = quasar.Circuit()
    for qubit in range(nqubit):
        if qubit % 3 == 0:
            circuitB.Rx(qubit, theta=2.0*np.pi*(np.random.rand() - 0.5))
        elif qubit % 3 == 1:
            circuitB.Ry(qubit, theta=2.0*np.pi*(np.random.rand() - 0.5))
        elif qubit % 3 == 2:
            circuitB.Rz(qubit, theta=2.0*np.pi*(np.random.rand() - 0.5))
        else:
            raise RuntimeError('Impossible')

    return quasar.Circuit.join_in_time([circuitA, circuit2, circuitB])

# => Pauli Library <= #

def randomized_pauli(
    nqubit,
    max_nbody=4,
    symmetric=False,
    ):

    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = quasar.Pauli.zero()

    pauli += I[-1]

    paulis = [X, Y, Z]
    for nbody in range(1, max_nbody+1):
        qubits = list(range(nqubit))
        np.random.shuffle(qubits)
        for k in range(0, nqubit - (nbody - 1), nbody):
            types = []
            for l in range(nbody):
                dec = np.random.rand()
                if dec < (1.0 / 3.0):
                    types.append(0)
                elif dec < (2.0 / 3.0):
                    types.append(1)
                else:
                    types.append(2) 
            if symmetric:
                if types.count(1) % 2 != 0:
                    types[types.index(1)] = 0
            term = I[-1]
            for l in range(nbody):
                term *= paulis[types[l]][qubits[k + l]]
            pauli += np.random.randn() * term
    return pauli 

def randomized_hermitian_pauli(
    nqubit,
    max_nbody=4,
    ):

    return randomized_pauli(nqubit=nqubit, max_nbody=max_nbody, symmetric=False)

def randomized_symmetric_pauli(
    nqubit,
    max_nbody=4,
    ):

    return randomized_pauli(nqubit=nqubit, max_nbody=max_nbody, symmetric=True)

# => Initial Statevector Library <= #

def none_statevector(
    nqubit,
    dtype=np.complex128,
    ):

    return None

def randomized_statevector(
    nqubit,
    dtype=np.complex128,
    ):

    # Not Haar random, but random enough for testing
    statevector = np.random.randn(2**nqubit) + 1.j * np.random.randn(2**nqubit)
    statevector = np.array(statevector, dtype=dtype)
    statevector /= np.sqrt(np.sum(statevector.conj() * statevector)) 
    return statevector

# => Test Run Statevector <= #

def test_run_statevector(
    backend1,
    backend2,
    circuit,
    statevector=None,
    min_qubit=None,
    nqubit=None,
    dtype=np.complex128,
    kwargs1={},
    kwargs2={},
    ):

    start = time.time()
    statevector1 = backend1.run_statevector(
        circuit=circuit,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1)
    t1 = time.time() - start

    start = time.time()
    statevector2 = backend2.run_statevector(
        circuit=circuit,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1)
    t2 = time.time() - start

    delta = np.max(np.abs(statevector1 - statevector2))
        
    return delta, t1, t2

def benchmark_run_statevector(
    nqubits,
    backend1,
    backend2,
    circuit_function,
    statevector_function,
    dtype,
    tolerance,
    kwargs1={},
    kwargs2={}, 
    ):

    allOK = True
    print('%2s: %11s %11s %11s %s' % (
        'N',
        'Delta',
        'T1',
        'T2',
        'Status',
        ))
    for nqubit in nqubits:
        circuit = circuit_function(nqubit)
        statevector = statevector_function(nqubit, dtype=dtype)
        delta, t1, t2 = test_run_statevector(
            backend1=backend1,
            backend2=backend2,
            circuit=circuit,
            statevector=statevector,
            min_qubit=0,
            nqubit=nqubit,
            dtype=dtype,
            kwargs1=kwargs1,
            kwargs2=kwargs2,
            )
        OK = delta < tolerance
        allOK &= OK
        print('%2d: %11.3E %11.3E %11.3E %s' % (
            nqubit,
            delta,
            t1, 
            t2,
            'OK' if OK else 'BAD'
            ))
    print('All OK' if allOK else 'Some BAD')
    print('')

    return allOK

def traverse_run_statevector():

    allOK = True
    
    backend1 = quasar.QuasarSimulatorBackend()
    backend2 = vulcan.VulcanSimulatorBackend()

    print('=> Traverse Run Statevector <=\n')

    print('Quasar-Vulcan: Randomized SU(4) Circuit, None Statevector, complex128\n')
    allOK &= benchmark_run_statevector(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        statevector_function=none_statevector,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, None Statevector, complex64\n')
    allOK &= benchmark_run_statevector(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        statevector_function=none_statevector,
        dtype=np.complex64,
        tolerance=1.0E-6,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, None Statevector, float64\n')
    allOK &= benchmark_run_statevector(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        statevector_function=none_statevector,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, None Statevector, float32\n')
    allOK &= benchmark_run_statevector(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        statevector_function=none_statevector,
        dtype=np.float32,
        tolerance=1.0E-6,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Statevector, complex128\n')
    allOK &= benchmark_run_statevector(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        statevector_function=randomized_statevector,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Statevector, complex64\n')
    allOK &= benchmark_run_statevector(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        statevector_function=randomized_statevector,
        dtype=np.complex64,
        tolerance=1.0E-6,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Statevector, float64\n')
    allOK &= benchmark_run_statevector(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        statevector_function=randomized_statevector,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Statevector, float32\n')
    allOK &= benchmark_run_statevector(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        statevector_function=randomized_statevector,
        dtype=np.float32,
        tolerance=1.0E-6,  
        )

    print('Traverse Status: %s' % ('All OK' if allOK else 'Some BAD'))
    print('')

    print('=> End Traverse Run Statevector <=\n')

    return allOK
    
# => Test Run Pauli Sigma <= #

def test_run_pauli_sigma(
    backend1,
    backend2,
    pauli,
    statevector=None,
    min_qubit=None,
    nqubit=None,
    dtype=np.complex128,
    kwargs1={},
    kwargs2={},
    ):

    start = time.time()
    statevector1 = backend1.run_pauli_sigma(
        pauli=pauli,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1)
    t1 = time.time() - start

    start = time.time()
    statevector2 = backend2.run_pauli_sigma(
        pauli=pauli,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1)
    t2 = time.time() - start

    delta = np.max(np.abs(statevector1 - statevector2))
        
    return delta, t1, t2

def benchmark_run_pauli_sigma(
    nqubits,
    backend1,
    backend2,
    pauli_function,
    statevector_function,
    dtype,
    tolerance,
    kwargs1={},
    kwargs2={}, 
    ):

    allOK = True
    print('%2s: %11s %11s %11s %s' % (
        'N',
        'Delta',
        'T1',
        'T2',
        'Status',
        ))
    for nqubit in nqubits:
        pauli = pauli_function(nqubit)
        statevector = statevector_function(nqubit, dtype=dtype)
        delta, t1, t2 = test_run_pauli_sigma(
            backend1=backend1,
            backend2=backend2,
            pauli=pauli,
            statevector=statevector,
            min_qubit=0,
            nqubit=nqubit,
            dtype=dtype,
            kwargs1=kwargs1,
            kwargs2=kwargs2,
            )
        OK = delta < tolerance
        allOK &= OK
        print('%2d: %11.3E %11.3E %11.3E %s' % (
            nqubit,
            delta,
            t1, 
            t2,
            'OK' if OK else 'BAD'
            ))
    print('All OK' if allOK else 'Some BAD')
    print('')

    return allOK

def traverse_run_pauli_sigma():

    allOK = True
    
    backend1 = quasar.QuasarSimulatorBackend()
    backend2 = vulcan.VulcanSimulatorBackend()

    print('=> Traverse Run Pauli Sigma <=\n')

    print('Quasar-Vulcan: Randomized Hermitian Pauli, Randomized Statevector, complex128\n')
    allOK &= benchmark_run_pauli_sigma(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=randomized_statevector,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized Hermitian Pauli, Randomized Statevector, complex64\n')
    allOK &= benchmark_run_pauli_sigma(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=randomized_statevector,
        dtype=np.complex64,
        tolerance=1.0E-6,  
        )

    print('Quasar-Vulcan: Randomized Symmetric Pauli, Randomized Statevector, float64\n')
    allOK &= benchmark_run_pauli_sigma(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=randomized_statevector,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized Symmetric Pauli, Randomized Statevector, float32\n')
    allOK &= benchmark_run_pauli_sigma(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=randomized_statevector,
        dtype=np.float32,
        tolerance=1.0E-6,  
        )

    print('Traverse Status: %s' % ('All OK' if allOK else 'Some BAD'))
    print('')

    print('=> End Traverse Run Pauli Sigma <=\n')

    return allOK

# => Test Run Pauli Expectation <= #

def test_run_pauli_expectation(
    backend1,
    backend2,
    circuit,
    pauli,
    statevector=None,
    min_qubit=None,
    nqubit=None,
    dtype=np.complex128,
    kwargs1={},
    kwargs2={},
    ):

    start = time.time()
    O1 = np.array(backend1.run_pauli_expectation(
        circuit=circuit,
        pauli=pauli,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1).values())
    t1 = time.time() - start

    start = time.time()
    O2 = np.array(backend2.run_pauli_expectation(
        circuit=circuit,
        pauli=pauli,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1).values())
    t2 = time.time() - start

    delta = np.max(np.abs(O1 - O2))
        
    return delta, t1, t2

def benchmark_run_pauli_expectation(
    nqubits,
    backend1,
    backend2,
    circuit_function,
    pauli_function,
    statevector_function,
    dtype,
    tolerance,
    kwargs1={},
    kwargs2={}, 
    ):

    allOK = True
    print('%2s: %11s %11s %11s %s' % (
        'N',
        'Delta',
        'T1',
        'T2',
        'Status',
        ))
    for nqubit in nqubits:
        circuit = circuit_function(nqubit)
        pauli = pauli_function(nqubit)
        statevector = statevector_function(nqubit, dtype=dtype)
        delta, t1, t2 = test_run_pauli_expectation(
            backend1=backend1,
            backend2=backend2,
            circuit=circuit,
            pauli=pauli,
            statevector=statevector,
            min_qubit=0,
            nqubit=nqubit,
            dtype=dtype,
            kwargs1=kwargs1,
            kwargs2=kwargs2,
            )
        OK = delta < tolerance
        allOK &= OK
        print('%2d: %11.3E %11.3E %11.3E %s' % (
            nqubit,
            delta,
            t1, 
            t2,
            'OK' if OK else 'BAD'
            ))
    print('All OK' if allOK else 'Some BAD')
    print('')

    return allOK

def traverse_run_pauli_expectation():

    allOK = True
    
    backend1 = quasar.QuasarSimulatorBackend()
    backend2 = vulcan.VulcanSimulatorBackend()

    print('=> Traverse Run Pauli Expectation <=\n')

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Hermitian Pauli, None Statevector, complex128\n')
    allOK &= benchmark_run_pauli_expectation(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=none_statevector,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Hermitian Pauli, None Statevector, complex64\n')
    allOK &= benchmark_run_pauli_expectation(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=none_statevector,
        dtype=np.complex64,
        tolerance=1.0E-5,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, None Statevector, float64\n')
    allOK &= benchmark_run_pauli_expectation(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=none_statevector,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, None Statevector, float32\n')
    allOK &= benchmark_run_pauli_expectation(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=none_statevector,
        dtype=np.float32,
        tolerance=1.0E-5,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Hermitian Pauli, Randomized Statevector, complex128\n')
    allOK &= benchmark_run_pauli_expectation(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=randomized_statevector,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Hermitian Pauli, Randomized Statevector, complex64\n')
    allOK &= benchmark_run_pauli_expectation(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=randomized_statevector,
        dtype=np.complex64,
        tolerance=1.0E-5,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, Randomized Statevector, float64\n')
    allOK &= benchmark_run_pauli_expectation(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=randomized_statevector,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, Randomized Statevector, float32\n')
    allOK &= benchmark_run_pauli_expectation(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=randomized_statevector,
        dtype=np.float32,
        tolerance=1.0E-5,  
        )

    print('Traverse Status: %s' % ('All OK' if allOK else 'Some BAD'))
    print('')

    print('=> End Traverse Run Pauli Expectation <=\n')

    return allOK

# => Test Run Pauli Expectation Value <= #

def test_run_pauli_expectation_value(
    backend1,
    backend2,
    circuit,
    pauli,
    statevector=None,
    min_qubit=None,
    nqubit=None,
    dtype=np.complex128,
    kwargs1={},
    kwargs2={},
    ):

    start = time.time()
    O1 = backend1.run_pauli_expectation_value(
        circuit=circuit,
        pauli=pauli,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1)
    t1 = time.time() - start

    start = time.time()
    O2 = backend2.run_pauli_expectation_value(
        circuit=circuit,
        pauli=pauli,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1)
    t2 = time.time() - start

    delta = np.max(np.abs(O1 - O2))
        
    return delta, t1, t2

def benchmark_run_pauli_expectation_value(
    nqubits,
    backend1,
    backend2,
    circuit_function,
    pauli_function,
    statevector_function,
    dtype,
    tolerance,
    kwargs1={},
    kwargs2={}, 
    ):

    allOK = True
    print('%2s: %11s %11s %11s %s' % (
        'N',
        'Delta',
        'T1',
        'T2',
        'Status',
        ))
    for nqubit in nqubits:
        circuit = circuit_function(nqubit)
        pauli = pauli_function(nqubit)
        statevector = statevector_function(nqubit, dtype=dtype)
        delta, t1, t2 = test_run_pauli_expectation_value(
            backend1=backend1,
            backend2=backend2,
            circuit=circuit,
            pauli=pauli,
            statevector=statevector,
            min_qubit=0,
            nqubit=nqubit,
            dtype=dtype,
            kwargs1=kwargs1,
            kwargs2=kwargs2,
            )
        OK = delta < tolerance
        allOK &= OK
        print('%2d: %11.3E %11.3E %11.3E %s' % (
            nqubit,
            delta,
            t1, 
            t2,
            'OK' if OK else 'BAD'
            ))
    print('All OK' if allOK else 'Some BAD')
    print('')

    return allOK

def traverse_run_pauli_expectation_value():

    allOK = True
    
    backend1 = quasar.QuasarSimulatorBackend()
    backend2 = vulcan.VulcanSimulatorBackend()

    print('=> Traverse Run Pauli Expectation Value <=\n')

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Hermitian Pauli, None Statevector, complex128\n')
    allOK &= benchmark_run_pauli_expectation_value(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=none_statevector,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Hermitian Pauli, None Statevector, complex64\n')
    allOK &= benchmark_run_pauli_expectation_value(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=none_statevector,
        dtype=np.complex64,
        tolerance=1.0E-5,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, None Statevector, float64\n')
    allOK &= benchmark_run_pauli_expectation_value(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=none_statevector,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, None Statevector, float32\n')
    allOK &= benchmark_run_pauli_expectation_value(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=none_statevector,
        dtype=np.float32,
        tolerance=1.0E-5,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Hermitian Pauli, Randomized Statevector, complex128\n')
    allOK &= benchmark_run_pauli_expectation_value(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=randomized_statevector,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Hermitian Pauli, Randomized Statevector, complex64\n')
    allOK &= benchmark_run_pauli_expectation_value(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=randomized_statevector,
        dtype=np.complex64,
        tolerance=1.0E-5,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, Randomized Statevector, float64\n')
    allOK &= benchmark_run_pauli_expectation_value(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=randomized_statevector,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, Randomized Statevector, float32\n')
    allOK &= benchmark_run_pauli_expectation_value(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=randomized_statevector,
        dtype=np.float32,
        tolerance=1.0E-5,  
        )

    print('Traverse Status: %s' % ('All OK' if allOK else 'Some BAD'))
    print('')

    print('=> End Traverse Run Pauli Expectation Value <=\n')

    return allOK

# => Test Run Pauli Expectation Value Gradient <= #

def test_run_pauli_expectation_value_gradient(
    backend1,
    backend2,
    circuit,
    pauli,
    statevector=None,
    min_qubit=None,
    nqubit=None,
    dtype=np.complex128,
    kwargs1={},
    kwargs2={},
    ):

    start = time.time()
    G1 = backend1.run_pauli_expectation_value_gradient(
        circuit=circuit,
        pauli=pauli,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1)
    t1 = time.time() - start

    start = time.time()
    G2 = backend2.run_pauli_expectation_value_gradient(
        circuit=circuit,
        pauli=pauli,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1)
    t2 = time.time() - start

    delta = np.max(np.abs(G1 - G2))
        
    return delta, t1, t2

def benchmark_run_pauli_expectation_value_gradient(
    nqubits,
    backend1,
    backend2,
    circuit_function,
    pauli_function,
    statevector_function,
    dtype,
    tolerance,
    kwargs1={},
    kwargs2={}, 
    ):

    allOK = True
    print('%2s: %11s %11s %11s %s' % (
        'N',
        'Delta',
        'T1',
        'T2',
        'Status',
        ))
    for nqubit in nqubits:
        circuit = circuit_function(nqubit)
        pauli = pauli_function(nqubit)
        statevector = statevector_function(nqubit, dtype=dtype)
        delta, t1, t2 = test_run_pauli_expectation_value_gradient(
            backend1=backend1,
            backend2=backend2,
            circuit=circuit,
            pauli=pauli,
            statevector=statevector,
            min_qubit=0,
            nqubit=nqubit,
            dtype=dtype,
            kwargs1=kwargs1,
            kwargs2=kwargs2,
            )
        OK = delta < tolerance
        allOK &= OK
        print('%2d: %11.3E %11.3E %11.3E %s' % (
            nqubit,
            delta,
            t1, 
            t2,
            'OK' if OK else 'BAD'
            ))
    print('All OK' if allOK else 'Some BAD')
    print('')

    return allOK

def traverse_run_pauli_expectation_value_gradient():

    allOK = True
    
    backend1 = quasar.QuasarUltrafastBackend()
    backend2 = vulcan.VulcanSimulatorBackend()

    print('=> Traverse Run Pauli Expectation Value Gradient <=\n')

    print('Quasar-Vulcan: Randomized SU(4) Grad Circuit, Randomized Hermitian Pauli, None Statevector, complex128\n')
    allOK &= benchmark_run_pauli_expectation_value_gradient(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4_grad,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=none_statevector,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Grad Circuit, Randomized Hermitian Pauli, None Statevector, complex64\n')
    allOK &= benchmark_run_pauli_expectation_value_gradient(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4_grad,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=none_statevector,
        dtype=np.complex64,
        tolerance=1.0E-5,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, None Statevector, float64\n')
    allOK &= benchmark_run_pauli_expectation_value_gradient(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=none_statevector,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, None Statevector, float32\n')
    allOK &= benchmark_run_pauli_expectation_value_gradient(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=none_statevector,
        dtype=np.float32,
        tolerance=1.0E-5,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Grad Circuit, Randomized Hermitian Pauli, Randomized Statevector, complex128\n')
    allOK &= benchmark_run_pauli_expectation_value_gradient(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4_grad,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=randomized_statevector,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Grad Circuit, Randomized Hermitian Pauli, Randomized Statevector, complex64\n')
    allOK &= benchmark_run_pauli_expectation_value_gradient(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4_grad,
        pauli_function=randomized_hermitian_pauli,
        statevector_function=randomized_statevector,
        dtype=np.complex64,
        tolerance=1.0E-5,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, Randomized Statevector, float64\n')
    allOK &= benchmark_run_pauli_expectation_value_gradient(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=randomized_statevector,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Symmetric Pauli, Randomized Statevector, float32\n')
    allOK &= benchmark_run_pauli_expectation_value_gradient(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        pauli_function=randomized_symmetric_pauli,
        statevector_function=randomized_statevector,
        dtype=np.float32,
        tolerance=1.0E-5,  
        )

    print('Traverse Status: %s' % ('All OK' if allOK else 'Some BAD'))
    print('')

    print('=> End Traverse Run Pauli Expectation Value Gradient <=\n')

    return allOK

# => Test Run Measurement <= #

def test_run_measurement(
    backend1,
    backend2,
    circuit,
    nmeasurement,
    statevector=None,
    min_qubit=None,
    nqubit=None,
    dtype=np.complex128,
    kwargs1={},
    kwargs2={},
    ):

    start = time.time()
    P1 = backend1.run_measurement(
        circuit=circuit,
        nmeasurement=nmeasurement,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1)
    t1 = time.time() - start

    start = time.time()
    P2 = backend2.run_measurement(
        circuit=circuit,
        nmeasurement=nmeasurement,
        statevector=statevector,
        min_qubit=min_qubit,
        nqubit=nqubit,
        dtype=dtype,
        **kwargs1)
    t2 = time.time() - start

    delta = 0.0 # This just checks if the characteristics are good
    delta += np.max(np.abs(P1.nmeasurement - P2.nmeasurement))
    delta += np.max(np.abs(P1.nqubit - P2.nqubit))
        
    return delta, t1, t2

def benchmark_run_measurement(
    nqubits,
    backend1,
    backend2,
    circuit_function,
    statevector_function,
    nmeasurement,
    dtype,
    tolerance,
    kwargs1={},
    kwargs2={}, 
    ):

    allOK = True
    print('%2s: %11s %11s %11s %s' % (
        'N',
        'Delta',
        'T1',
        'T2',
        'Status',
        ))
    for nqubit in nqubits:
        circuit = circuit_function(nqubit)
        statevector = statevector_function(nqubit, dtype=dtype)
        delta, t1, t2 = test_run_measurement(
            backend1=backend1,
            backend2=backend2,
            circuit=circuit,
            statevector=statevector,
            nmeasurement=nmeasurement,
            min_qubit=0,
            nqubit=nqubit,
            dtype=dtype,
            kwargs1=kwargs1,
            kwargs2=kwargs2,
            )
        OK = delta < tolerance
        allOK &= OK
        print('%2d: %11.3E %11.3E %11.3E %s' % (
            nqubit,
            delta,
            t1, 
            t2,
            'OK' if OK else 'BAD'
            ))
    print('All OK' if allOK else 'Some BAD')
    print('')

    return allOK

def traverse_run_measurement():

    allOK = True
    
    backend1 = quasar.QuasarSimulatorBackend()
    backend2 = vulcan.VulcanSimulatorBackend()

    nmeasurement = 1000

    print('=> Traverse Run Measurement <=\n')

    print('Quasar-Vulcan: Randomized SU(4) Circuit, None Statevector, complex128\n')
    allOK &= benchmark_run_measurement(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        statevector_function=none_statevector,
        nmeasurement=nmeasurement,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, None Statevector, complex64\n')
    allOK &= benchmark_run_measurement(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        statevector_function=none_statevector,
        nmeasurement=nmeasurement,
        dtype=np.complex64,
        tolerance=1.0E-6,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, None Statevector, float64\n')
    allOK &= benchmark_run_measurement(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        statevector_function=none_statevector,
        nmeasurement=nmeasurement,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, None Statevector, float32\n')
    allOK &= benchmark_run_measurement(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        statevector_function=none_statevector,
        nmeasurement=nmeasurement,
        dtype=np.float32,
        tolerance=1.0E-6,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Statevector, complex128\n')
    allOK &= benchmark_run_measurement(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        statevector_function=randomized_statevector,
        nmeasurement=nmeasurement,
        dtype=np.complex128,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SU(4) Circuit, Randomized Statevector, complex64\n')
    allOK &= benchmark_run_measurement(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_su4,
        statevector_function=randomized_statevector,
        nmeasurement=nmeasurement,
        dtype=np.complex64,
        tolerance=1.0E-6,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Statevector, float64\n')
    allOK &= benchmark_run_measurement(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        statevector_function=randomized_statevector,
        nmeasurement=nmeasurement,
        dtype=np.float64,
        tolerance=1.0E-12,  
        )

    print('Quasar-Vulcan: Randomized SO(4) Circuit, Randomized Statevector, float32\n')
    allOK &= benchmark_run_measurement(
        nqubits=list(range(1,16)),
        backend1=backend1,
        backend2=backend2,
        circuit_function=randomized_so4,
        statevector_function=randomized_statevector,
        nmeasurement=nmeasurement,
        dtype=np.float32,
        tolerance=1.0E-6,  
        )

    print('Traverse Status: %s' % ('All OK' if allOK else 'Some BAD'))
    print('')

    print('=> End Traverse Run Measurement <=\n')

    return allOK
    
def run_all():

    allOK = True

    allOK &= traverse_run_statevector()
    allOK &= traverse_run_pauli_sigma()
    allOK &= traverse_run_pauli_expectation()
    allOK &= traverse_run_pauli_expectation_value()
    allOK &= traverse_run_pauli_expectation_value_gradient()
    allOK &= traverse_run_measurement()

    print('Final Status: %s' % ('All OK' if allOK else 'Some BAD'))
    print('')

    return allOK

if __name__ == '__main__':

    run_all()    

            
            
            
    

    
        
