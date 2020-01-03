import quasar

gadget = quasar.Circuit().SO4(1,2)
# gadget = quasar.Circuit().SO4(0,1)
print(gadget)

cgadget = quasar.Circuit().Ry(1).add_controlled_gate(gadget, (0,1,2))
print(cgadget)

print(cgadget.parameters)
print(cgadget.parameter_str)
print(cgadget.exploded().parameters)
print(cgadget.exploded().parameter_str)

cgadget.set_parameter_values([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
print(cgadget.parameter_str)
print(cgadget.exploded().parameter_str)
