import numpy as np
import quasar
import vulcan
import time

def cis_circuit(
    nqubit=4,
    ):

    gadget = quasar.Circuit().Ry(1).CZ(0,1).Ry(1).CX(1,0)
    
    circuit = quasar.Circuit().X(0)
    for I in range(nqubit-1):
        circuit.add_gates(circuit=gadget, qubits=(I, I+1))
    
    parameter_values = []
    for I in range(nqubit - 1):
        value = (1.0 - I / 17.5)
        parameter_values.append(+value)
        parameter_values.append(-value)
    circuit.set_parameter_values(parameter_values)
    
    return circuit

import sys
nqubit = int(sys.argv[1])
circuit = cis_circuit(nqubit)

circuit = quasar.Circuit.join_in_time([circuit]*5)

backend = vulcan.VulcanSimulatorBackend()

print(circuit.nqubit)

nmeasurement = 10000
dtype=np.complex64

backend.run_measurement(circuit, nmeasurement=nmeasurement, dtype=dtype)

start = time.time()
measurement1 = backend.run_measurement(circuit, nmeasurement=nmeasurement, dtype=dtype)
print('%11.3E' % (time.time() - start))

start = time.time()
measurement2 = backend.run_measurement(circuit, nmeasurement=nmeasurement, dtype=dtype, compressed=False)
print('%11.3E' % (time.time() - start))

# print(measurement1)
# print(measurement2)
