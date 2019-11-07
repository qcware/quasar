import quasar

backend = quasar.QuasarSimulatorBackend()

circuit = quasar.Circuit().X(0).CX(0,1).CX(1,2)
# circuit = quasar.Circuit().X(-1).CX(-1,0).CX(0,1)

# gadget = quasar.Circuit().CX(-1,0)
gadget = quasar.Circuit().CX(0, 1)
# circuit = quasar.Circuit().X(0).CX(0,1).add_gate(gadget, (1,2))
circuit = quasar.Circuit().X(-1).CX(-1,0).add_gate(gadget, (0,1))

measurement = backend.run_measurement(
    circuit,
#     min_qubit=-1,
#     nqubit=4,
    )

print(circuit)

print(measurement)
