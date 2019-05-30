import quasar

X = quasar.Pauli.X
Y = quasar.Pauli.Y
Z = quasar.Pauli.Z

print(X[0])

what = 1.0 * X[1]
what = X[1] * 1.0
what = X[1] / 2.0
what = X[0] * Z[1]
what = 1.0 * X[0] * Z[1]
what = X[0] * Z[1] * 1.0
what = - X[0] * Y[1] * Z[2]
what = + X[0] * Y[1] * Z[2]
what = + X[0] * Y[1] * Z[2] * X[3] / 2.0
what = + X[0] * Y[1] * Z[2] * X[3] / 2.0 + X[0] * Y[1] * Z[2] * X[3] / 2.0
what = + X[0] * Y[1] * Z[2] * X[3] / 2.0 - X[0] * Y[1] * Z[2] * X[3] / 2.0
what = + X[0] * Y[1] * Z[2] * X[3] / 2.0 - Z[0] * Y[1] * Z[2] * X[3] / 2.0
print(what)
print(what.content_str)
print(what.dot(what))
