import quasar

circuit1 = quasar.Circuit().Ry(0).Rz(1).Rx(0).Z(1)
print(circuit1)

circuit2 = quasar.Circuit().X(2).add_controlled_gate(circuit1, (0, 1, 2))
print(circuit2)

print(circuit2.explode())

print(circuit2.serialize_in_time())
