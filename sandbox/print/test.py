import quasar
import time

start = time.time()

circuit = quasar.Circuit()
# circuit.add_gate(quasar.Gate.H, 0, time_start=-1)
circuit.add_gate(quasar.Gate.H, -1, time_start=-1)
circuit.add_gate(quasar.Gate.CX, (0, 1), time_start=1)
circuit.add_gate(quasar.Gate.CX, (1, 2))
circuit.add_gate(quasar.ControlledGate(quasar.Gate.X, [True]), (0, 1))
circuit.add_gate(quasar.ControlledGate(quasar.Gate.X, [False, True]), (0, 1, 2))
print(circuit.min_time)
print(circuit.max_time)
print(circuit.min_qubit)
print(circuit.max_qubit)
print(circuit.is_composite)
print(circuit.gates)
print(circuit.times)
print(circuit.qubits)
print(circuit.times_and_qubits)
print(circuit)
print(circuit.recentered())



print(quasar.Circuit.test_equivalence(circuit, circuit.recentered()))
print(quasar.Circuit.test_equivalence(circuit, circuit))

print(circuit.reversed())
print(circuit.dagger())

gadget = quasar.Circuit()
gadget.add_gate(quasar.Gate.H, 0)
gadget.add_gate(quasar.Gate.CX, (0, 1))
gadget.add_gate(quasar.Gate.CX, (1, 0))
print(gadget)

circuit.add_gate(gadget, (0, 2), ascii_symbols=['A']*gadget.nqubit, )
circuit.add_gate(quasar.CompositeGate(gadget, ascii_symbols=['B']*gadget.nqubit), (1, 3))
print(circuit)
print(circuit.is_composite)

print(quasar.Circuit.join_in_time([circuit]*3))
print(quasar.Circuit.join_in_qubits([circuit]*3))

circuit.Ry(4)
print(circuit)

circuit.add_controlled_gate(gadget,  qubits=(0, 2, 3))

print(circuit)
print(circuit.recentered())
print(circuit.sparse())
print(circuit.exploded())
print(circuit.exploded().is_composite)
print('%11.3E' % (time.time() - start))
