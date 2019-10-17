import quasar
import time

start = time.time()

# circuit1 = quasar.Circuit().CX(0,1)
circuit1 = quasar.Circuit().CX(-1, 0)
circuit2 = quasar.Circuit().add_gate(circuit1, (0,1))
print(circuit1)
print(circuit2)

backend = quasar.QuasarSimulatorBackend()
print(backend.run_unitary(circuit1))
print(backend.run_unitary(circuit2))
print(circuit2.gates[((0,), (0,1))].operator)
print('%11.3E' % (time.time() - start))
