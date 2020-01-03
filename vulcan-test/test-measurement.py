import quasar
import vulcan
import numpy as np
import time

# circuit = quasar.Circuit().H(0).CX(0,1)
circuit = quasar.Circuit()
# circuit.Ry(0, theta=0.2).Ry(1, theta=-0.3)
circuit.H(0)
for k in range(25):
    circuit.CX(k, k+1)
# for k in range(30):
#     circuit.H(k)

# circuit.X(1).X(0)



backend = vulcan.VulcanSimulatorBackend()
# backend = quasar.CirqSimulatorBackend()
# backend = quasar.QuasarSimulatorBackend()

print(circuit)
print(circuit.nqubit)
start = time.time()
# print(backend.run_measurement(circuit, nmeasurement=10, dtype=np.float64))
print(backend.run_measurement(circuit, nmeasurement=1000, dtype=np.complex128))
print('%11.3E' % (time.time() - start))
