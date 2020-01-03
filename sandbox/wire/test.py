import quasar
import pickle
import dill

# circuit = quasar.Circuit().Ry(-1, 0.1).Ry(0, -0.2).CX(-1, 0)
circuit = quasar.Circuit().X(-1).X(0).CX(-1, 0)
print(circuit)
print(pickle.dumps(circuit))
