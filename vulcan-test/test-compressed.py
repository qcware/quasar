import numpy as np
import quasar
import vulcan

circuit = quasar.Circuit()
circuit.S(0)
circuit.S(1).T(1)
circuit.H(2)
circuit.S(3).S(4).S(5)
circuit.S(4).S(5)
circuit.CX(4,5)
circuit.CX(3,4)
circuit.S(3)
circuit.S(4).S(4)
circuit.S(5)
circuit.CX(4,5)
circuit.CX(5,4)
circuit.CX(4,5)
print(circuit)

statevector = np.ones((2**circuit.nqubit,)) / np.sqrt(2**circuit.nqubit)

backend = vulcan.VulcanSimulatorBackend()
statevector1 = backend.run_statevector(circuit, statevector=statevector, compressed=False)
statevector2 = backend.run_statevector(circuit, statevector=statevector, compressed=True)
print(np.sum(statevector1.conj() * statevector2))
