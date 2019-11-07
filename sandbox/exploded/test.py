import quasar

gadget1 = quasar.Circuit().SO4(1,2)
print(gadget1.exploded())

gadget2 = quasar.Circuit().add_gate(gadget1, qubits=(3,1))
print(gadget2.gates)
print(gadget2.exploded())
