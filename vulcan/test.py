import vulcan
import time

def test2(n=28):

    start = time.time()
    
    X = vulcan.Gate_complex128(
        1,
        "X",
        [
            vulcan.complex128(0.0, 0.0),
            vulcan.complex128(1.0, 0.0),
            vulcan.complex128(1.0, 0.0),
            vulcan.complex128(0.0, 0.0),
        ])

    circuit = vulcan.Circuit_complex128(
        n,
        [X]*n,
        [[_] for _ in range(n)],
        )

    pauli = vulcan.Pauli_complex128(
        n,
        [[2]]*n,
        [[_] for _ in range(n)],
        [vulcan.complex128(1.0, 0.0) for _ in range(n)],
        )

    statevector = vulcan.run_statevector_complex128(circuit)
    print(statevector)

    # for val in expectation.values():
    #     print(val.real())

    print('%11.3E\n' % (time.time() - start))

test2()
test2()
test2()

    
