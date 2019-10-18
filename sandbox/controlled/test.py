import quasar

X = quasar.Gate.X

cX = quasar.ControlledGate(X, controls=[False, False,])
print(cX)
print(cX.operator)

cX = quasar.ControlledGate(X, controls=[False, True,])
print(cX)
print(cX.operator)

cX = quasar.ControlledGate(X, controls=[True, False,])
print(cX)
print(cX.operator)

cX = quasar.ControlledGate(X, controls=[True, True,])
print(cX)
print(cX.operator)
