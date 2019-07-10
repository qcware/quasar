"""
Testing functions in Circuit.py from the top down
"""
import quasar
import collections

def init_circuit():
    circuit = quasar.Circuit(2)
    if circuit.N != 2:
        return False
    if circuit.Ts:
        return False
    if not isinstance(circuit.TAs, set):
        return False
    if not isinstance(circuit.gates, dict): 
        return False

    return True

if __name__ == "__main__":
    a = init_circuit()


