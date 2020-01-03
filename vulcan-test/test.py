import quasar
import vulcan
import time 
import sys
import numpy as np

backend1 = vulcan.VulcanSimulatorBackend()
# backend1 = quasar.QuasarSimulatorBackend()
# backend2 = quasar.PyquilSimulatorBackend()
# backend2 = quasar.QiskitSimulatorBackend()
backend2 = quasar.CirqSimulatorBackend()

N = int(sys.argv[1]) 

gadget = quasar.Circuit().Ry(1).CZ(0,1).Ry(1).CX(1,0)

circuit = quasar.Circuit().X(0)
circuit.T(0)
for I in range(N):
    circuit.add_gates(circuit=gadget, qubits=(I, I+1))
for I in range(N+1):
    circuit.T(I)
circuit.T(N)
circuit = circuit.slice(qubits=list(reversed(range(N+1))))
circuit.Ry(0, theta=0.3, time_placement='next')
print(circuit)

parameter_values = []
for I in range(N):
    value = (1.0 - I / 17.0)
    parameter_values.append(+value)
    parameter_values.append(-value)
circuit.set_parameter_values(parameter_values)
print(circuit.parameter_str)

statevector1 = backend1.run_statevector(circuit, dtype=np.complex64)
start = time.time()
statevector1 = backend1.run_statevector(circuit, dtype=np.complex64)
print('%11.3E\n' % (time.time() - start))

statevector2 = backend2.run_statevector(circuit)
start = time.time()
statevector2 = backend2.run_statevector(circuit)
print('%11.3E\n' % (time.time() - start))

print(np.sum(statevector1.conj() * statevector2))
