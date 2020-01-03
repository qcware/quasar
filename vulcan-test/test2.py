import quasar
import vulcan
import numpy as np
import time
import sys

N = int(sys.argv[1]) # Increase this to get some more dramatic timing examples below

gadget = quasar.Circuit().Ry(1).CZ(0,1).Ry(1).CX(1,0)
print(gadget)    

circuit = quasar.Circuit().X(0)
# circuit.T(0)
for I in range(N):
    circuit.add_gates(circuit=gadget, qubits=(I, I+1))
# for I in range(N+1):
#     circuit.T(I)
# circuit.T(N)
# circuit.TT(N)
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

I, X, Y, Z = quasar.Pauli.IXYZ()
pauli = quasar.Pauli.zero()
for k in range(N+1):
    pauli += (k + 1) / 10.0 * Z[k]
print(pauli)

backend1 = quasar.QuasarSimulatorBackend()
backend2 = vulcan.VulcanSimulatorBackend()

# dtype = np.complex64
dtype = np.float64

start = time.time()
statevector1 = backend1.run_statevector(circuit, dtype=dtype)
print('%11.3E' % (time.time() - start))

start = time.time()
statevector2 = backend2.run_statevector(circuit, dtype=dtype)
print('%11.3E' % (time.time() - start))

print(np.sum(statevector1.conj() * statevector2))

start = time.time()
sigma1 = backend1.run_pauli_sigma(pauli, statevector1, dtype=dtype)
print('%11.3E' % (time.time() - start))
print(np.sum(sigma1.conj() * statevector1))

start = time.time()
sigma2 = backend2.run_pauli_sigma(pauli, statevector2, dtype=dtype)
print('%11.3E' % (time.time() - start))
print(np.sum(sigma2.conj() * statevector2))

start = time.time()
energy = backend1.run_pauli_expectation_value(circuit, pauli, dtype=dtype)
print('%11.3E' % (time.time() - start))
print(energy)

start = time.time()
energy = backend2.run_pauli_expectation_value(circuit, pauli, dtype=dtype)
print('%11.3E' % (time.time() - start))
print(energy)

start = time.time()
gradient1 = backend1.run_pauli_expectation_value_gradient(circuit, pauli, dtype=dtype)
print('%11.3E' % (time.time() - start))

start = time.time()
gradient2 = backend2.run_pauli_expectation_value_gradient(circuit, pauli, dtype=dtype)
print('%11.3E' % (time.time() - start))

print(np.max(np.abs(gradient1 - gradient2)))
