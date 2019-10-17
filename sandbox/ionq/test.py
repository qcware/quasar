import quasar

parrish_api_key = '7MduR9Pj1ogAdlFFis05S2:5CU6u38eyIj9jZae5IyLHT'

# backend = quasar.IonQBackend(
#     api_key=parrish_api_key,
#     )
# print(backend)
# print(backend.summary_str)

# backend = quasar.QuasarSimulatorBackend()
backend = quasar.IonQBackend(api_key=parrish_api_key)

circuit = quasar.Circuit().H(0)

circuit = quasar.Circuit().Ry(0, theta=0.1).Ry(1, theta=0.2).CX(0, 1)

circuit = quasar.Circuit().Ry(0).X(1)
circuit = quasar.Circuit().X(0).Ry(1)

circuit = quasar.Circuit().H(0)
for I in range(6):
    circuit.CX(I, I+1)

# circuit = quasar.Circuit().Ry(0, theta=0.4)

print(circuit)
histogram1 = backend.run_measurement(circuit, nmeasurement=1000)
# histogram1 = backend.run_measurement(circuit, nmeasurement=None)
print(histogram1)
histogram2 = histogram1.to_count_histogram()
print(histogram2)
histogram3 = histogram2.to_probability_histogram()
print(histogram3)


# print(backend.run_measurement(circuit, nmeasurement=None))
