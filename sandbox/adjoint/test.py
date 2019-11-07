import quasar

print(quasar.Gate.I.adjoint())
print(quasar.Gate.I.adjoint().operator)

print(quasar.Gate.S.adjoint().adjoint())

print(quasar.Gate.Ry(theta=0.1).operator)
print(quasar.Gate.Ry(theta=0.1).adjoint().adjoint().operator)
