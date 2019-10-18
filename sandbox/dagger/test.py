import quasar

print(quasar.Gate.I.dagger())
print(quasar.Gate.I.dagger().operator)

print(quasar.Gate.S.dagger().dagger())

print(quasar.Gate.Ry(theta=0.1).operator)
print(quasar.Gate.Ry(theta=0.1).dagger().dagger().operator)
