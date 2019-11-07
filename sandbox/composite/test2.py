import quasar

# circuit1 = quasar.Circuit().CX(0,1)
circuit1 = quasar.Circuit().CX(-1, 0)
# circuit2 = quasar.Circuit().add_controlled_gate(circuit1, (0, 1, 2))
circuit2 = quasar.Circuit().add_controlled_gate(circuit1, (2, 1, 0))
print(circuit2)

backend = quasar.QuasarSimulatorBackend()
print(backend.run_unitary(circuit2))
print(backend.run_unitary(circuit2.explode()))

