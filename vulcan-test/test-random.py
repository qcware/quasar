import numpy as np
import quasar
import vulcan
import time
from qiskit.quantum_info.synthesis import two_qubit_cnot_decompose

def random_su4():
    X = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    Q, R = np.linalg.qr(X)           
    Q /= pow(np.linalg.det(Q), 0.25)  
    return Q

def random_su4_gate():
    Q = random_su4()
    return quasar.Gate.U2(Q)

def random_circuit(
    nqubit,
    depth=1,
    ):

    circuit = quasar.Circuit()
    for layer in range(depth):
        P = list(range(nqubit))
        np.random.shuffle(P)
        for index in range(0, nqubit-1, 2):
            circuit.add_gate(
                gate=random_su4_gate(),
                qubits=(P[index], P[index+1]),
                times=layer)
    return circuit

def random_circuit2(
    nqubit,
    depth=1,
    ):

    circuits = []
    for layer in range(depth):
        circuit = quasar.Circuit()
        P = list(range(nqubit))
        np.random.shuffle(P)
        for index in range(0, nqubit-1, 2):
            decomposed_SU = two_qubit_cnot_decompose(random_su4())
            for shit in decomposed_SU:
                gate = shit[0]
                qubits = [P[_.index + index] for _ in shit[1]]
                if gate.name == 'cx':
                    circuit.CX(qubits[0], qubits[1])
                elif gate.name == 'u3':
                    circuit.u3(qubits[0], theta=float(gate.params[0]), phi=float(gate.params[1]), lam=float(gate.params[2]))
                else:
                    raise RuntimeError('WTF')
        circuits.append(circuit)
    return quasar.Circuit.join_in_time(circuits)

def test_timing(
    backend,
    circuit,
    nmeasurement,
    dtype, 
    **kwargs):
    
    start = time.time()
    measurement = backend.run_measurement(
        circuit, 
        nmeasurement=nmeasurement,
        dtype=dtype,
        **kwargs)
    stop = time.time()
    return stop - start

def test_error(
    backend,
    circuit,
    dtype, 
    **kwargs):

    statevector = backend.run_statevector(
        circuit,
        dtype=dtype,
        **kwargs)
    
    backend2 = quasar.QuasarSimulatorBackend()
    statevector2 = backend2.run_statevector(
        circuit,
        dtype=np.complex128,
        )

    return np.max(np.abs(statevector - statevector2))

def test1():

    import sys
    nqubit = int(sys.argv[1])
    depth = int(sys.argv[2])
    nmeasurement = int(sys.argv[3])
    dtype = {
        'float32' : np.float32,
        'float64' : np.float64,
        'complex64' : np.complex64,
        'complex128' : np.complex128,
    }[sys.argv[4]]
    
    circuit = random_circuit2(nqubit, depth)
    print(circuit)
    print(circuit.ngate)
    
    backend = vulcan.VulcanSimulatorBackend()
    
    d = test_error(
        backend,
        circuit=circuit,
        dtype=dtype,
        )
    print(d)
    
    t = test_timing(
        backend,
        circuit=circuit,
        nmeasurement=nmeasurement,
        dtype=dtype,
        )
    
    t = test_timing(
        backend,
        circuit=circuit,
        nmeasurement=nmeasurement,
        dtype=dtype,
        )
    print(t)
    
    t = test_timing(
        backend,
        circuit=circuit,
        nmeasurement=nmeasurement,
        dtype=dtype,
        compressed=False)
    print(t)

def test2():

    backend = vulcan.VulcanSimulatorBackend()
    
    min_nqubit = 20
    max_nqubit = 30
    depth = 10
    nmeasurement = 10000

    data = {}

    circuits = { nqubit : random_circuit2(nqubit, depth) for nqubit in range(min_nqubit, max_nqubit+1) }

    # Burner
    t = test_timing(
        backend=backend,
        circuit=circuits[min_nqubit],
        nmeasurement=nmeasurement,
        dtype=np.complex64,
        compressed=True)    
    
    nqubits = []
    ts = []
    for nqubit in range(min_nqubit, max_nqubit+1):
        t = test_timing(
            backend=backend,
            circuit=circuits[nqubit],
            nmeasurement=nmeasurement,
            dtype=np.complex64,
            compressed=True)    
        nqubits.append(nqubit)
        ts.append(t)
        print(nqubit, t)
    nqubits = np.array(nqubits)
    ts = np.array(ts)
    data['nqubits_complex64_compressed'] = nqubits
    data['ts_complex64_compressed'] = ts
    
    # nqubits = []
    # ts = []
    # for nqubit in range(min_nqubit, max_nqubit+1):
    #     t = test_timing(
    #         backend=backend,
    #         circuit=circuits[nqubit],
    #         nmeasurement=nmeasurement,
    #         dtype=np.complex64,
    #         compressed=False)    
    #     nqubits.append(nqubit)
    #     ts.append(t)
    #     print(nqubit, t)
    # nqubits = np.array(nqubits)
    # ts = np.array(ts)
    # data['nqubits_complex64_uncompressed'] = nqubits
    # data['ts_complex64_uncompressed'] = ts
    
    nqubits = []
    ts = []
    for nqubit in range(min_nqubit, max_nqubit):
        t = test_timing(
            backend=backend,
            circuit=circuits[nqubit],
            nmeasurement=nmeasurement,
            dtype=np.complex128,
            compressed=True)    
        nqubits.append(nqubit)
        ts.append(t)
        print(nqubit, t)
    nqubits = np.array(nqubits)
    ts = np.array(ts)
    data['nqubits_complex128_compressed'] = nqubits
    data['ts_complex128_compressed'] = ts
    
    # nqubits = []
    # ts = []
    # for nqubit in range(min_nqubit, max_nqubit):
    #     t = test_timing(
    #         backend=backend,
    #         circuit=circuits[nqubit],
    #         nmeasurement=nmeasurement,
    #         dtype=np.complex128,
    #         compressed=False)    
    #     nqubits.append(nqubit)
    #     ts.append(t)
    #     print(nqubit, t)
    # nqubits = np.array(nqubits)
    # ts = np.array(ts)
    # data['nqubits_complex128_uncompressed'] = nqubits
    # data['ts_complex128_uncompressed'] = ts

    np.savez('timings.npz', **data)

if __name__ == '__main__':

    test2()
