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

def ntime():
    circuit = quasar.Circuit(3)
    if circuit.ntime != 0:
        return False

    circuit.H(0).CX(0,1).CX(1,2)
    if circuit.ntime != 3:
        return False
    
    return True
    
def ngate():
    circuit = quasar.Circuit(2)
    if circuit.ngate != 0:
        return False

    circuit.H(0).CX(0,1)
    if circuit.ngate != 2:
        return False

    return True

def ngate1():
    circuit = quasar.Circuit(2)
    if circuit.ngate1 != 0:
        return False

    circuit.H(0).CX(0,1)
    if circuit.ngate1 != 1:
        return False

    return True

def ngate2():
    circuit = quasar.Circuit(2)
    if circuit.ngate2 != 0:
        return False

    circuit.H(0).CX(0,1)
    if circuit.ngate2 != 1:
        return False

    return True

def add_gate():
    circuit = quasar.Circuit(2)
    circuit.add_gate(quasar.Gate.X, 1)
    circuit.add_gate(quasar.Gate.Y, 0, time=3)
    circuit.add_gate(quasar.Gate.H, 0, time=5)
    circuit.add_gate(quasar.Gate.Z, 1, time_placement="early")
    circuit.add_gate(quasar.Gate.X, 1, time_placement="late")
    circuit.add_gate(quasar.Gate.CX, (0,1), time_placement="next")

    #print(circuit)
    # checking gates added at the right times and on the right qubits
    time_qubit_dict = {(0, 1), (3, 0), (5, 0), (1,1), (5,1), (6,1), (6,0)}
    if circuit.TAs != time_qubit_dict:
        return False

    # make sure the correct gates have been added
    gates = ["X", "Y", "H", "Z", "CX"]
    for gate in list(circuit.gates.values()):
        if gate.name not in gates:
            return False

    return True

def gate():
    circuit = quasar.Circuit(2).X(0).CX(0,1).H(1)
    if circuit.gate(0, 0).name != "X":
        return False
    if circuit.gate((0,1), 1).name != "CX":
        return False
    if circuit.gate((1,), 2).name != "H":
        return False
    
    return True

def copy():
    circuit1 = quasar.Circuit(1).X(0)
    gates1 = circuit1.gates
    circuit2 = circuit1.copy()
    gates2 = circuit2.gates
    
    # check that gates are different objects
    if gates1 == gates2:
        return False
    
    # check that changing circuit2 does not affect circuit1
    circuit2.H(0)
    
    if gates1 != circuit1.gates:
        return False

    return True 

def subset():
    # Creating circuit with 5 time slices
    circuit = quasar.Circuit(1).X(0)
    circuit.add_gate(quasar.Gate.X, 0, time=5)
    circuit.add_gate(quasar.Gate.X, 0, time=3)
    #print(circuit)

    # Cuting circuit to slices 0+3-5 (i.e. removing slices 1 and 2) 
    subset_moments = ([0,3,4,5])
    subcirc = circuit.subset(subset_moments)
    #print(subcirc.TAs) 

    # subcirc should be the length of the subset_moments list
    if subcirc.ntime != len(subset_moments):
        return False

    test_circ = quasar.Circuit(1).X(0).X(0)
    test_circ.add_gate(quasar.Gate.X, 0, time=3)
    # And the circ should have gates in the right locations
    if test_circ.TAs != subcirc.TAs:
        return False

    return True

def concatenate():
    circ1 = quasar.Circuit(2).H(0).H(1)
    circ2 = quasar.Circuit(2).X(0).X(1)
    circ3 = quasar.Circuit(2).H(0).H(1).X(0).X(1)

    # Testing that concatenating circ1 and circ2 produces circ3
    cat_circ = circ1.concatenate([circ1, circ2])
    for gate1, gate2 in zip(list(cat_circ.gates.values()), list(circ3.gates.values())):
        if gate1.name != gate2.name:
            return False

    return True

def deadjoin():
    circ1 = quasar.Circuit(3).H(0).H(1).H(2)
    circ2 = quasar.Circuit(2).H(0).H(1)

    # remove first (oneth) qubit
    dj_circ = circ1.deadjoin([0,2])
    # Testing that circuit created with deadjoin from circ1 and circ2 are the same
    for gate1, gate2 in zip(list(dj_circ.gates.values()), list(circ2.gates.values())):
        if gate1.name != gate2.name:
            return False

    return True

def adjoin():
    circ1 = quasar.Circuit(2).H(0).CX(0, 1)
    circ2 = quasar.Circuit(2).X(0).X(1)
    circ3 = quasar.Circuit(4).H(0).CX(0,1).X(2).X(3)

    ad_circ = circ1.adjoin([circ1, circ2])
    # Testing that circuit created with adjoin from circ1 and circ2 are the same as circ3
    for gate1, gate2 in zip(list(ad_circ.gates.values()), list(circ3.gates.values())):
        if gate1.name != gate2.name:
            return False

    return True

def test_reversed():
    circ1 = quasar.Circuit(2).H(0).H(1).CX(0,1).X(0).X(1)
    circ2 = quasar.Circuit(2).X(1).X(0).CX(0,1).H(1).H(0)
    reversed_circ = circ1.reversed()
    for t in range(circ1.ntime):
        if t == 1:
            # Catching the two qubit gate
            if circ2.gate((0,1),t).name != reversed_circ.gate((0,1),t).name:
                return False
        else:
            if circ2.gate(0,t).name != reversed_circ.gate(0,t).name:
                return False
            if circ2.gate(1,t).name != reversed_circ.gate(1,t).name:
                return False

    return True

if __name__ == "__main__":
    init_circuit()
    ntime()
    ngate()
    ngate1()
    ngate2()
    add_gate()
    gate()
    copy()
    subset()
    concatenate()
    deadjoin()
    adjoin()
    test_reversed()
    print(test_reversed())

