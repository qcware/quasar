import quasar
import time

start = time.time()

circuit = quasar.Circuit().Ry(-1, 0.1).Ry(0, -0.2).CX(-1, 0)
print(circuit)

backend = quasar.QuasarSimulatorBackend()
print(backend)
print(backend.has_run_statevector)
print(backend.has_statevector_input)

print(backend.run_statevector(circuit))
print(backend.run_measurement(circuit, 1000))

print(backend.run_unitary(circuit))
print(backend.run_density_matrix(circuit))

print('%11.3E' % (time.time() - start))
