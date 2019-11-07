import quasar    

# backend = quasar.QuasarSimulatorBackend()
backend = quasar.QuasarUltrafastBackend()

circuit = quasar.Circuit().Ry(0,theta=0.1).Ry(1, theta=-0.2).CZ(0,1).Ry(1, theta=+0.2).CX(1,0)
print(circuit)

I, X, Y, Z = quasar.Pauli.IXYZ()
pauli = I[-1] + Z[0] + 2.0 * Z[1] + 3.0 * Z[0] * Z[1]
print(pauli)

print(circuit.parameter_indices)
energy = backend.run_pauli_expectation_value(
        circuit, 
        pauli, 
        nmeasurement=None,
        )
gradient = backend.run_pauli_expectation_value_gradient(
        circuit, 
        pauli, 
        nmeasurement=None,
#         parameter_indices=[1],
        )

print(energy)
print(gradient)

backend = quasar.QuasarSimulatorBackend()
gradient = backend.run_pauli_expectation_value_gradient(
        circuit, 
        pauli, 
        nmeasurement=None,
        )
print(gradient)
