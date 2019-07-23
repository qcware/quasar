"""
Testing functions of the Circuit class from circuit.py
"""

import numpy as np
import quasar
import collections
from util.circuit_generator import random_circuit, simple_circuit
from util.error import L1_error

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

def add_gate_qubit_validation():
    # Assert statements have been added to the function add_gate to validate qubit arguments
    # Uncommenting the following lines activate the assert statements
    circ2 = quasar.Circuit(2)
    #circ2.add_gate(quasar.Gate.X, -1, time=0)
    #circ2.add_gate(quasar.Gate.CX, (-1,0), time=1)
    #circ2.add_gate(quasar.Gate.CX, (1,1), time=2)

    #print(circ2)
    #print(circ2.simulate())
 

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

def is_equivalent():
    circ1 = quasar.Circuit(1) 
    circ2 = quasar.Circuit(1) 
    # Check that empty circuits of 1 qubit are equivalent
    if not circ1.is_equivalent(circ2):
        return False

    # Check that circuits of different number of qubit registers are unequal 
    circ3 = quasar.Circuit(2) 
    if circ1.is_equivalent(circ3):
        return False

    # Circuits comprised of the same gates are equal
    circ4 = quasar.Circuit(2).H(0).CX(0,1).X(1)
    circ5 = quasar.Circuit(2).H(0).CX(0,1).X(1)
    if not circ4.is_equivalent(circ5):
        return False

    # Check that circuits with same gates at different times are not equal
    circ6 = quasar.Circuit(2).H(0).CX(0,1)
    circ6.add_gate(quasar.Gate.X, 1, time=10)
    if circ6.is_equivalent(circ5):
        return False

    return True

def nonredundant():
    circ1 = quasar.Circuit(2).H(0).CX(0,1)
    # Putting a lot of empty time in circ1
    circ1.add_gate(quasar.Gate.X, 1, time=10)
    circ2 = quasar.Circuit(2).H(0).CX(0,1).X(1)
    
    # Removing empty time so that it should be equivalent to circ2
    rn_circ = circ1.nonredundant()
    if not rn_circ.is_equivalent(circ2):
        return False
    
    return True

def compressed():
    circ1 = quasar.Circuit(3).H(0).CX(0,1).X(1).CZ(1,2).CZ(1,2).Rx(2, np.pi).u1(0, -np.pi)
    c_circ = circ1.compressed()
    
    circ1_vec = circ1.simulate()
    c_circ_vec = c_circ.simulate()

    # Statevector for original circ and compressed circ should be equalivalent
    if not np.array_equal(circ1_vec,  c_circ_vec):
        return False

    return True

def subcircuit():
    circ = quasar.Circuit(3).H(0)
    circ.add_gate(quasar.Gate.Y, 0, time=3)
    circ.add_gate(quasar.Gate.Z, 1, time=5)
    circ.add_gate(quasar.Gate.CZ, (0,1), time=7)
    circ.add_gate(quasar.Gate.Ry(np.pi), 0, time=10)
    circ.add_gate(quasar.Gate.Z, 2, time=12)
    
    qubits = [0,1]
    times = [0,2,3,5,7]
    sub_circ = circ.subcircuit(qubits, times)

    circ2 = quasar.Circuit(2).H(0)
    circ2.add_gate(quasar.Gate.Y, 0, time=2)
    circ2.add_gate(quasar.Gate.Z, 1, time=3)
    circ2.CZ(0,1)

    if not sub_circ.is_equivalent(circ2):
        return False
    
    return True

def add_circuit():
    circ1 = quasar.Circuit(2).H(0)
    circ1.add_gate(quasar.Gate.CX, (0,1), time=5)
    #print(circ1)

    circ2 = quasar.Circuit(3).H(0).H(1).H(2)
    #print(circ2)

    circ6 = circ2.add_circuit(circ1, (0,1), time=2)
    circ7 = quasar.Circuit(1).X(0)
    circ7 = circ7.add_gate(quasar.Gate.Y, 0, time=3)
    #print("circ7", circ7)
    #print(circ4)
    circ5 = quasar.Circuit(1).X(0)
    circ6 = circ6.add_circuit(circ5, qubits=2, time_placement= "early")
    circ6 = circ6.add_circuit(circ5, qubits=2, time_placement="late")
    circ6 = circ6.add_circuit(circ5, qubits=2, time_placement= "next")
    circ6 = circ6.add_circuit(circ7, qubits=2, times=[10,11,12,13])
    #print("circ6")
    
    test_circ = quasar.Circuit(3).H(0).H(1).H(2)
    test_circ = test_circ.add_gate(quasar.Gate.H, 0, 2)
    test_circ = test_circ.add_gate(quasar.Gate.CX, (0,1), 7)
    test_circ = test_circ.add_gate(quasar.Gate.X, 2, 1)
    test_circ = test_circ.add_gate(quasar.Gate.X, 2, 7).X(2)
    test_circ = test_circ.add_gate(quasar.Gate.X, 2, 10).add_gate(quasar.Gate.Y, 2, 13)
    

    if not circ6.is_equivalent(test_circ):
        print("it's here")
        return False

    # Adding a single qubit register to a new register in self circuit
    circ8 = quasar.Circuit(2).H(0)
    circ9 = quasar.Circuit(1).X(0)
    circ10 = circ8.add_circuit(circ9, qubits=1)
    if not circ10.is_equivalent(quasar.Circuit(2).H(0).X(1)):
        return Fasle
    #print(circ10)

    return True

def sort_gates():
    # Needs to be sorted
    circ1 = quasar.Circuit(3).Y(1)
    circ1.add_gate(quasar.Gate.X, 0, time=0)
    circ1.add_gate(quasar.Gate.CX, (0,2), time=2)
    circ1.add_gate(quasar.Gate.Z, 1, time=1)
    circ1.add_gate(quasar.Gate.X, 1, time=2)

    # Inserted in time and qubit order
    circ2 = quasar.Circuit(3).X(0).Y(1)
    circ2.add_gate(quasar.Gate.Z, 1, time=1)
    circ2.add_gate(quasar.Gate.CX, (0,2), time=2)
    circ2.add_gate(quasar.Gate.X, 1, time=2)
    #print(circ2)
    
    # Sort circ1
    circ1.sort_gates()
    if circ1.is_equivalent(circ2):
        return True
    else:
        return False

def is_equivalent_order():
    # Inserted out of order, but same as circ2
    circ1 = quasar.Circuit(3).Y(1)
    circ1.add_gate(quasar.Gate.X, 0, time=0)
    circ1.add_gate(quasar.Gate.CX, (0,2), time=2)
    circ1.add_gate(quasar.Gate.Z, 1, time=1)
    circ1.add_gate(quasar.Gate.X, 1, time=2)

    # Inserted in time and qubit order
    circ2 = quasar.Circuit(3).X(0).Y(1)
    circ2.add_gate(quasar.Gate.Z, 1, time=1)
    circ2.add_gate(quasar.Gate.CX, (0,2), time=2)
    circ2.add_gate(quasar.Gate.X, 1, time=2)
    #print(circ2)
    
    # Not explicitly sorting circ1
    if circ1.is_equivalent(circ2):
        return True
    else:
        return False

def simulate():
    """
    Validate "Circuit.simulate()"
    """
    result = []
    # test case 0: run simulate() with simple_circuit(0)
    circuit, ans = simple_circuit(0)
    wfn = circuit.simulate()
    result.append((wfn, ans))
    # test case 1: run simulate() with simple_circuit(1)
    circuit, ans = simple_circuit(1)
    wfn = circuit.simulate()
    result.append((wfn, ans))
    return L1_error(result)

    
def simulate_steps():
    """
    Validate "Circuit.simulate_steps()"
    """
    circuit, _ = simple_circuit(0)
    result = []
    wfn0, wfn1 = circuit.simulate_steps()
    ans0, ans1 = [np.sqrt(1/2),0,np.sqrt(1/2),0], [np.sqrt(1/2),0,0,np.sqrt(1/2)]
    result.append((wfn0[1], ans0))
    result.append((wfn1[1], ans1))
    return L1_error(result)

    
def apply_gate_1():
    """
    Validate "Circuit.apply_gate_1()"
    """
    result = []
    # test case 1
    U = quasar.Matrix.X
    circuit = quasar.Circuit(N=1)
    initial_wfn = np.array([1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_1(initial_wfn, np.zeros_like(initial_wfn), U, 0)
    ans = [0, 1]
    result.append((wfn, ans))
    # test case 2
    U = quasar.Matrix.H
    circuit = quasar.Circuit(N=1)
    initial_wfn = np.array([1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_1(initial_wfn, np.zeros_like(initial_wfn), U, 0)
    ans = [np.sqrt(1/2), np.sqrt(1/2)]
    result.append((wfn, ans))
    # test case 3
    U = quasar.Matrix.Rx(theta=np.pi/6)
    circuit = quasar.Circuit(N=1)
    initial_wfn = np.array([1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_1(initial_wfn, np.zeros_like(initial_wfn), U, 0)
    ans = [np.sqrt(3)/2, -1j/2]
    result.append((wfn, ans))
    
    return L1_error(result)
    
    
def apply_gate_1():
    """
    Validate "Circuit.apply_gate_1()"
    """
    result = []
    # test case 1 (X)
    U = quasar.Matrix.X
    circuit = quasar.Circuit(N=1)
    initial_wfn = np.array([1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_1(initial_wfn, np.zeros_like(initial_wfn), U, 0)
    ans = [0, 1]
    result.append((wfn, ans))
    # test case 2 (H)
    U = quasar.Matrix.H
    circuit = quasar.Circuit(N=1)
    initial_wfn = np.array([1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_1(initial_wfn, np.zeros_like(initial_wfn), U, 0)
    ans = [np.sqrt(1/2), np.sqrt(1/2)]
    result.append((wfn, ans))
    # test case 3 (Rx)
    U = quasar.Matrix.Rx(theta=np.pi/6)
    circuit = quasar.Circuit(N=1)
    initial_wfn = np.array([1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_1(initial_wfn, np.zeros_like(initial_wfn), U, 0)
    ans = [np.sqrt(3)/2, -1j/2]
    result.append((wfn, ans))
    
    return L1_error(result)
    

def apply_gate_1_format():
    """
    Validate the shape and typy of "Circuit.apply_gate_1()"
    """
    # test case (X)
    U = quasar.Matrix.X
    circuit = quasar.Circuit(N=1)
    initial_wfn = np.array([1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_1(initial_wfn, np.zeros_like(initial_wfn), U, 0)
    
    if not wfn.shape==(2,):
        return False
    if not wfn.dtype == np.complex128:
        return False
    
    return True
    

def apply_gate_2():
    """
    Validate "Circuit.apply_gate_2()"
    """
    result = []
    # test case 1 (CX)
    U = quasar.Matrix.CX
    circuit = quasar.Circuit(N=2)
    initial_wfn = np.array([0,0,1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_2(initial_wfn, np.zeros_like(initial_wfn), U, 0, 1)
    ans = [0,0,0,1]
    result.append((wfn, ans))
    # test case 2 (CS)
    U = quasar.Matrix.CS
    circuit = quasar.Circuit(N=2)
    initial_wfn = np.array([0,0,0,1], dtype=np.complex128)
    wfn = circuit.apply_gate_2(initial_wfn, np.zeros_like(initial_wfn), U, 0, 1)
    ans = [0,0,0,1j]
    result.append((wfn, ans))
    
    return L1_error(result) 

    
def apply_gate_2_format():
    """
    Validate the shape and typy of "Circuit.apply_gate_2()"
    """
    # test case (CX)
    U = quasar.Matrix.CX
    circuit = quasar.Circuit(N=2)
    initial_wfn = np.array([0,0,0,1], dtype=np.complex128)
    wfn = circuit.apply_gate_2(initial_wfn, np.zeros_like(initial_wfn), U, 0, 1)
    
    if not wfn.shape==(4,):
        return False
    if not wfn.dtype == np.complex128:
        return False
    
    return True
    

def apply_gate_3():
    """
    Validate "Circuit.apply_gate_3()"
    """
    result = []
    # test case 1 (CCX)
    U = quasar.Matrix.CCX
    circuit = quasar.Circuit(N=3)
    initial_wfn = np.array([0,0,0,0,0,0,1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_3(initial_wfn, np.zeros_like(initial_wfn), U, 0, 1, 2)
    ans = [0,0,0,0,0,0,0,1]
    result.append((wfn, ans))
    # test case 2 (CSWAP)
    U = quasar.Matrix.CSWAP
    circuit = quasar.Circuit(N=3)
    initial_wfn = np.array([0,0,0,0,0,1,0,0], dtype=np.complex128)
    wfn = circuit.apply_gate_3(initial_wfn, np.zeros_like(initial_wfn), U, 0, 1, 2)
    ans = [0,0,0,0,0,0,1,0]
    result.append((wfn, ans))
    
    return L1_error(result) 

    
def apply_gate_3_format():
    """
    Validate the shape and typy of "Circuit.apply_gate_3()"
    """
    # test case (CCX)
    U = quasar.Matrix.CCX
    circuit = quasar.Circuit(N=3)
    initial_wfn = np.array([0,0,0,0,0,0,1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_3(initial_wfn, np.zeros_like(initial_wfn), U, 0, 1, 2)
    
    if not wfn.shape==(8,):
        return False
    if not wfn.dtype == np.complex128:
        return False
    
    return True
    
    
def compute_1pdm():
    """
    Validate "Circuit.compute_1pdm()"
    """
    circuit = quasar.Circuit(N=1)
    wfn = np.array([0,1], dtype=np.complex128)
    dm = circuit.compute_1pdm(wfn,wfn,0)
    ans = [[0,0],[0,1]]

    return L1_error([(dm, ans)])

    
def compute_1pdm_format():
    """
    Validate the shape and dtype "Circuit.compute_1pdm()"
    """
    circuit = quasar.Circuit(N=1)
    wfn = np.array([0,1], dtype=np.complex128)
    dm = circuit.compute_1pdm(wfn,wfn,0)

    if not dm.shape==(2,2):
        return False
    if not dm.dtype == np.complex128:
        return False
    
    return True
    

def compute_2pdm():
    """
    Validate "Circuit.compute_2pdm()"
    """
    circuit = quasar.Circuit(N=2)
    wfn = np.array([0,1,0,0], dtype=np.complex128)
    dm = circuit.compute_2pdm(wfn,wfn,0,1)
    ans = [[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]
    
    return L1_error([(dm, ans)])

    
def compute_2pdm_format():
    """
    Validate the shape and dtype "Circuit.compute_2pdm()"
    """
    circuit = quasar.Circuit(N=2)
    wfn = np.array([0,1,0,0], dtype=np.complex128)
    dm = circuit.compute_2pdm(wfn,wfn,0,1)

    if not dm.shape==(4,4):
        return False
    if not dm.dtype == np.complex128:
        return False
    
    return True
    

def compute_3pdm():
    """
    Validate "Circuit.compute_3pdm()"
    """
    circuit = quasar.Circuit(N=3)
    wfn = np.zeros((8,), dtype=np.complex128)
    wfn[6] = 1.0
    dm = circuit.compute_3pdm(wfn,wfn,0,1,2)
    ans = np.zeros((8,8), dtype=np.complex128)
    ans[6,6] = 1.0
    
    return L1_error([(dm, ans)])

    
def compute_3pdm_format():
    """
    Validate the shape and dtype "Circuit.compute_3pdm()"
    """
    circuit = quasar.Circuit(N=3)
    wfn = np.zeros((8,), dtype=np.complex128)
    wfn[6] = 1.0
    dm = circuit.compute_3pdm(wfn,wfn,0,1,2)

    if not dm.shape==(8,8):
        return False
    if not dm.dtype == np.complex128:
        return False

    return True
    

def compute_4pdm():
    """
    Validate "Circuit.compute_4pdm()"
    """
    circuit = quasar.Circuit(N=4)
    wfn = np.zeros((16,), dtype=np.complex128)
    wfn[13] = 1.0
    dm = circuit.compute_4pdm(wfn,wfn,0,1,2,3)
    ans = np.zeros((16,16), dtype=np.complex128)
    ans[13,13] = 1.0
    
    return L1_error([(dm, ans)])

    
def compute_4pdm_format():
    """
    Validate the shape and dtype "Circuit.compute_4pdm()"
    """
    circuit = quasar.Circuit(N=4)
    wfn = np.zeros((16,), dtype=np.complex128)
    wfn[13] = 1.0
    dm = circuit.compute_4pdm(wfn,wfn,0,1,2,3)

    if not dm.shape==(16,16):
        return False
    if not dm.dtype == np.complex128:
        return False

    return True   
    
    
def compute_npdm():
    """
    Validate "Circuit.compute_npdm()"
    """
    circuit = quasar.Circuit(N=4)
    wfn = np.zeros((2**7,), dtype=np.complex128)
    wfn[101] = 1.0
    dm = circuit.compute_npdm(wfn,wfn,[i for i in range(7)])
    ans = np.zeros((2**7,2**7), dtype=np.complex128)
    ans[101,101] = 1.0
    
    return L1_error([(dm, ans)])

    
def compute_npdm_format():
    """
    Validate the shape and dtype "Circuit.compute_npdm()"
    """
    circuit = quasar.Circuit(N=7)
    wfn = np.zeros((2**7,), dtype=np.complex128)
    wfn[13] = 1.0
    dm = circuit.compute_npdm(wfn,wfn,[i for i in range(7)])

    if not dm.shape==(2**7,2**7):
        return False
    if not dm.dtype == np.complex128:
        return False

    return True  
    
    
def compute_pauli_1():
    """
    Validate "Circuit.compute_pauli_1()"
    """
    circuit = quasar.Circuit(N=1)
    wfn = np.array([0,1], dtype=np.complex128)
    expectation = circuit.compute_pauli_1(wfn,0)
    ans = [1,0,0,-1]
    
    return L1_error([(expectation, ans)])
    
    
def compute_pauli_2():
    """
    Validate "Circuit.compute_pauli_2()"
    """
    circuit = quasar.Circuit(N=2)
    wfn = np.array([0,0,1,0], dtype=np.complex128)
    expectation = circuit.compute_pauli_2(wfn,0,1)
    ans = [[1,0,0,1],[0,0,0,0],[0,0,0,0],[-1,0,0,-1]]

    return L1_error([(expectation, ans)])
    
    
def compute_pauli_3():
    """
    Validate "Circuit.compute_pauli_3()"
    """
    circuit = quasar.Circuit(N=3)
    wfn = np.zeros((8,), dtype=np.complex128)
    wfn[6] = 1
    expectation = circuit.compute_pauli_3(wfn,0,1,2)
    ans = np.zeros((4,4,4), dtype=np.complex128)
    ans[0,0,:] = [1,0,0,1]
    ans[0,3,:] = [-1,0,0,-1]
    ans[3,0,:] = [-1,0,0,-1]
    ans[3,3,:] = [1,0,0,1]

    return L1_error([(expectation, ans)])
    
    
def compute_pauli_4():
    """
    Validate "Circuit.compute_pauli_4()"
    """
    circuit = quasar.Circuit(N=4)
    wfn = np.zeros((16,), dtype=np.complex128)
    wfn[0] = 1
    expectation = circuit.compute_pauli_4(wfn,0,1,2,3)
    ans = np.zeros((4,4,4,4), dtype=np.complex128)
    ans[0,0,0,:] = [1,0,0,1]
    ans[0,0,3,:] = [1,0,0,1]
    ans[0,3,0,:] = [1,0,0,1]
    ans[0,3,3,:] = [1,0,0,1]
    ans[3,0,0,:] = [1,0,0,1]
    ans[3,0,3,:] = [1,0,0,1]
    ans[3,3,0,:] = [1,0,0,1]
    ans[3,3,3,:] = [1,0,0,1]

    return L1_error([(expectation, ans)])
    
    
def compute_pauli_n():
    """
    Validate "Circuit.compute_pauli_n()"
    Only verify the shape and dtype of output.
    """
    circuit = quasar.Circuit(N=5)
    wfn = np.zeros((2**5,), dtype=np.complex128)
    wfn[27] = 1
    expectation = circuit.compute_pauli_n(wfn,[i for i in range(5)])
    
    if not expectation.shape==(4,4,4,4,4):
        return False
    if not expectation.dtype == np.float64:
        return False

    return True  
        

def measure():
    """
    Validate "Circuit.measure()"
    """
    circuit, ans = simple_circuit(0)
    measurement = circuit.measure(nmeasurement=10000)
    measurement = [measurement.get('{0:02b}'.format(k)) for k in range(4)]
    measurement = np.where(measurement, measurement, np.zeros_like(measurement)) 
    measurement = measurement / np.linalg.norm(measurement)
    ans    = ans / np.linalg.norm(ans)

    return L1_error([(measurement, ans)], margin=1e-1)

    
def compute_measurements_from_statevector():
    """
    Validate "Circuit.compute_measurements_from_statevector()"
    """
    circuit = quasar.Circuit(N=2)
    statevector = [1,0,0,1]
    statevector = statevector / np.linalg.norm(statevector)
    measurement = circuit.compute_measurements_from_statevector(statevector=statevector, nmeasurement=10000)
    measurement = [measurement.get('{0:02b}'.format(k)) for k in range(4)]
    measurement = np.where(measurement, measurement, np.zeros_like(measurement)) 
    measurement = measurement / np.linalg.norm(measurement)
    
    return L1_error([(measurement, statevector)], margin=1e-1)
    
    
def nparam():
    """
    Validate "Circuit.nparam()"
    """
    circuit = quasar.Circuit(N=2).H(0).Rx(0,1.5).CX(0,1).Rz(1,0.7)  
    return circuit.nparam == 2

    
def param_keys():
    """
    Validate "Circuit.param_keys()"
    """
    circuit = quasar.Circuit(N=2).H(0).Rx(1,1.5).CX(0,1).Rz(0,0.5)  
    param_keys = circuit.param_keys
    # The following 3 checks are: (1)length (2)timestamp (3)qubit indices
    if not len(param_keys)==2: return False
    if not (param_keys[0][0]==0 and param_keys[1][0]==2): return False
    if not (param_keys[0][1]==(1,) and param_keys[1][1]==(0,)): return False
    return True
    

def param_values():
    """
    Validate "Circuit.param_values()"
    """
    circuit = quasar.Circuit(N=2).H(0).Rx(1,1.5).CX(0,1).Rz(0,0.5)  
    param_values = circuit.param_values
    # The following 2 checks are: (1)length (2)values    
    if not len(param_values)==2: return False
    if not (param_values[0]==1.5 and param_values[1]==0.5): return False
    return True
    

def set_param_values():
    """
    Validate "Circuit.set_param_values()"
    """
    circuit = quasar.Circuit(N=2).H(0).Rx(1,1.5).CX(0,1).Rz(0,0.5).Ry(1,2.5)
    circuit = circuit.set_param_values([0.7,1.7],[0,2])
    return circuit.param_values == [0.7, 0.5, 1.7]
       
    
def params():
    """
    Validate "Circuit.params()"
    """
    circuit = quasar.Circuit(N=2).H(0).Rx(1,1.5).CX(0,1).Rz(0,0.5)
    params = circuit.params.items()
    if not len(params)==2: return False

    # The following 2 checks are: (1)timestamp (2)qubit indices
    param_keys = [x[0] for x in params]
    if not (param_keys[0][0]==0 and param_keys[1][0]==2): return False
    if not (param_keys[0][1]==(1,) and param_keys[1][1]==(0,)): return False
    # The following 1 check is: (1)values
    param_values = [x[1] for x in params]
    if not (param_values[0]==1.5 and param_values[1]==0.5): return False

    return True
    

def set_param():
    """
    Validate "Circuit.set_param_values()"
    """
    from collections import OrderedDict
    circuit = quasar.Circuit(N=2).H(0).Rx(1,1.5).CX(0,1).Rz(0,0.5).Ry(1,2.5)
    param = OrderedDict({ (2, (0,), 'theta'): 0.8 })
    circuit = circuit.set_params(param)
    return circuit.param_values == [1.5, 0.8, 2.5]    
    
    
def param_str():
    """
    Validate "Circuit.param_str()"
    We decide not to test simple printing function. 
    """
    return True
    
    
def I():
    """
    Validate "Circuit.I()"
    """
    circuit = quasar.Circuit(N=1).I(0)
    wfn = circuit.simulate()
    ans = [1,0]
    
    return L1_error([(wfn, ans)])
    
    
def X():
    """
    Validate "Circuit.X()"
    """
    circuit = quasar.Circuit(N=1).X(0)
    wfn = circuit.simulate()
    ans = [0,1]
    return L1_error([(wfn, ans)])


def Y():
    """
    Validate "Circuit.Y()"
    """
    circuit = quasar.Circuit(N=1).Y(0)
    wfn = circuit.simulate()
    ans = [0,1j]
    return L1_error([(wfn, ans)])


def Z():
    """
    Validate "Circuit.Z()"
    """
    circuit = quasar.Circuit(N=1).Z(0)
    wfn = circuit.simulate([0,1])
    ans = [0,-1]
    return L1_error([(wfn, ans)])
    

def H():
    """
    Validate "Circuit.H()"
    """
    circuit = quasar.Circuit(N=1).H(0)
    wfn = circuit.simulate()
    ans = [np.sqrt(1/2),np.sqrt(1/2)]
    return L1_error([(wfn, ans)])
    

def S():
    """
    Validate "Circuit.S()"
    """
    circuit = quasar.Circuit(N=1).S(0)
    wfn = circuit.simulate([0,1])
    ans = [0,1j]
    return L1_error([(wfn, ans)])

    
def T():
    """
    Validate "Circuit.T()"
    """
    circuit = quasar.Circuit(N=1).T(0)
    wfn = circuit.simulate([0,1])
    ans = [0,(1.0+1.0j)*np.sqrt(1/2)]
    return L1_error([(wfn, ans)])
    
    
def Rx2():
    """
    Validate "Circuit.Rx2()"
    """
    circuit = quasar.Circuit(N=1).Rx2(0)
    wfn = circuit.simulate([0,1])
    ans = [np.sqrt(1/2)*1j, np.sqrt(1/2)]
    return L1_error([(wfn, ans)])
    
    
def Rx2T():
    """
    Validate "Circuit.Rx2T()"
    """
    circuit = quasar.Circuit(N=1).Rx2T(0)
    wfn = circuit.simulate([0,1])
    ans = [-np.sqrt(1/2)*1j, np.sqrt(1/2)]
    return L1_error([(wfn, ans)])
    
    
def CX():
    """
    Validate "Circuit.CX()"
    """
    circuit = quasar.Circuit(N=2).CX(0,1)
    wfn = circuit.simulate([0,0,1,0])
    ans = [0,0,0,1]
    return L1_error([(wfn, ans)])    
    

def CY():
    """
    Validate "Circuit.CY()"
    """
    circuit = quasar.Circuit(N=2).CY(0,1)
    wfn = circuit.simulate([0,0,1,0])
    ans = [0,0,0,1j]
    return L1_error([(wfn, ans)])    
    
    
def CZ():
    """
    Validate "Circuit.CZ()"
    """
    circuit = quasar.Circuit(N=2).CZ(0,1)
    wfn = circuit.simulate([0,0,0,1])
    ans = [0,0,0,-1]
    return L1_error([(wfn, ans)])      
    

def CS():
    """
    Validate "Circuit.CS()"
    """
    circuit = quasar.Circuit(N=2).CS(0,1)
    wfn = circuit.simulate([0,0,0,1])
    ans = [0,0,0,1j]
    return L1_error([(wfn, ans)])    
    
    
def SWAP():
    """
    Validate "Circuit.SWAP()"
    """
    circuit = quasar.Circuit(N=2).SWAP(0,1)
    wfn = circuit.simulate([0,0,1,0])
    ans = [0,1,0,0]
    return L1_error([(wfn, ans)])       
    

def CCX():
    """
    Validate "Circuit.CCX()"
    """
    circuit = quasar.Circuit(N=3).CCX(0,1,2)
    wfn = np.zeros((8,))
    wfn[6] = 1
    wfn = circuit.simulate(wfn)
    ans = np.zeros((8,))
    ans[7] = 1
    return L1_error([(wfn, ans)])    
    
    
def CSWAP():
    """
    Validate "Circuit.CSWAP()"
    """
    circuit = quasar.Circuit(N=3).CSWAP(0,1,2)
    wfn = np.zeros((8,))
    wfn[6] = 1
    wfn = circuit.simulate(wfn)
    ans = np.zeros((8,))
    ans[5] = 1
    return L1_error([(wfn, ans)])    
    
    
def Rx():
    """
    Validate "Circuit.Rx()"
    """
    circuit = quasar.Circuit(N=1).Rx(0, np.pi/6)
    wfn = circuit.simulate([0,1])
    ans = [-1/2*1j, np.sqrt(3)/2]
    return L1_error([(wfn, ans)])
    
    
def Ry():
    """
    Validate "Circuit.Ry()"
    """
    circuit = quasar.Circuit(N=1).Ry(0, np.pi/6)
    wfn = circuit.simulate([0,1])
    ans = [-1/2, np.sqrt(3)/2]
    return L1_error([(wfn, ans)])
    
    
def Rz():
    """
    Validate "Circuit.Rz()"
    """
    circuit = quasar.Circuit(N=1).Rz(0, np.pi/6)
    wfn = circuit.simulate([0,1])
    ans = [0, np.sqrt(3)/2+1/2*1j]
    return L1_error([(wfn, ans)])
    
    
def u1():
    """
    Validate "Circuit.u1()"
    """
    circuit = quasar.Circuit(N=1).u1(0, np.pi/6)
    wfn = circuit.simulate([0,1])
    ans = [0, np.sqrt(3)/2+1/2*1j]
    return L1_error([(wfn, ans)])
    
    
def u2():
    """
    Validate "Circuit.u2()"
    """
    circuit = quasar.Circuit(N=1).u2(0, np.pi/2, np.pi/2)
    wfn = circuit.simulate([0,1])
    ans = [-np.sqrt(1/2)*1j, -np.sqrt(1/2)]
    return L1_error([(wfn, ans)])
    
    
def u3():
    """
    Validate "Circuit.u3()"
    """
    circuit = quasar.Circuit(N=1).u3(0, np.pi/2 ,np.pi/4, np.pi/4)
    wfn = circuit.simulate([0,1])
    ans = [-1/2-1/2*1j, np.sqrt(1/2)*1j]
    return L1_error([(wfn, ans)])
    

def SO4():
    """
    Validate "Circuit.SO4()"
    """
    circuit = quasar.Circuit(N=2).SO4(0, 1, A=np.pi/2, F=np.pi/2)
    wfn = circuit.simulate([0.0,np.sqrt(1/2),0.0,np.sqrt(1/2)])
    ans = [np.sqrt(1/2),0,np.sqrt(1/2),0]
    return L1_error([(wfn, ans)])
    
    
def SO42():
    """
    Validate "Circuit.SO42()"
    """
    circuit = quasar.Circuit(N=2).SO42(0, 1, thetaIY=np.pi/4, thetaZY=-np.pi/4)
    wfn = circuit.simulate([0.0,np.sqrt(1/2),0.0,np.sqrt(1/2)])
    ans = [0,np.sqrt(1/2),-np.sqrt(1/2),0] 
    return L1_error([(wfn, ans)])
    
    
def CF():
    """
    Validate "Circuit.CF()"
    """
    circuit = quasar.Circuit(N=2).CF(0, 1, theta=np.pi/4)
    wfn = circuit.simulate([0,0,0,1])
    ans = [0,0,np.sqrt(1/2),-np.sqrt(1/2)]
    return L1_error([(wfn, ans)])

    
def R_ion():
    """
    Validate "Circuit.R_ion()"
    """
    circuit = quasar.Circuit(N=1).R_ion(0, np.pi/3, np.pi/2)
    wfn = circuit.simulate([0,1])
    ans = [-0.5, np.sqrt(3)/2]
    return L1_error([(wfn, ans)])

    
def Rx_ion():
    """
    Validate "Circuit.Rx_ion()"
    """
    circuit = quasar.Circuit(N=1).Rx_ion(0, np.pi/3)
    wfn = circuit.simulate([0,1])
    ans = [-0.5*1j, np.sqrt(3)/2]
    return L1_error([(wfn, ans)])    
    
    
def Ry_ion():
    """
    Validate "Circuit.Ry_ion()"
    """
    circuit = quasar.Circuit(N=1).Ry_ion(0, np.pi/3)
    wfn = circuit.simulate([0,1])
    ans = [-0.5, np.sqrt(3)/2]
    return L1_error([(wfn, ans)])       
    
    
def Rz_ion():
    """
    Validate "Circuit.Rz_ion()"
    """
    circuit = quasar.Circuit(N=1).Rz_ion(0, np.pi/3)
    wfn = circuit.simulate([0,1])
    ans = [0, np.sqrt(3)/2+0.5*1j]
    return L1_error([(wfn, ans)])   
    

def XX_ion():
    """
    Validate "Circuit.XX_ion()"
    """
    result = []
    # test case 1: wfn = [0,0,0,1]
    circuit = quasar.Circuit(N=2).XX_ion(0,1, chi = np.pi/3)
    wfn = circuit.simulate([0,0,0,1])
    ans = [np.sqrt(3)/2*1j, 0, 0, 0.5]
    result.append((wfn,ans))
    # test case 2: wfn = [0,1,0,0]
    circuit = quasar.Circuit(N=2).XX_ion(0,1, chi = np.pi/3)
    wfn = circuit.simulate([0,1,0,0])
    ans = [0, 0.5, -np.sqrt(3)/2*1j, 0]
    result.append((wfn,ans))
    print(L1_error([(wfn, ans)])  )
    return L1_error([(wfn, ans)])  


def U1():
    """
    Validate "Circuit.U1()"
    """
    U = quasar.Matrix.H
    circuit = quasar.Circuit(N=1).U1(0, U)
    wfn = circuit.simulate([0,1])
    ans = [np.sqrt(1/2), -np.sqrt(1/2)]
    return L1_error([(wfn, ans)])   


def U2():
    """
    Validate "Circuit.U2()"
    """
    U = quasar.Matrix.XX_ion(chi=np.pi/4)
    circuit = quasar.Circuit(N=2).U2(0, 1, U)
    wfn = circuit.simulate([0,1,0,0])
    ans = [0, np.sqrt(1/2), -np.sqrt(1/2)*1j, 0]
    return L1_error([(wfn, ans)])   

if __name__ == "__main__":
    init_circuit()
    ntime()
    ngate()
    ngate1()
    ngate2()
    add_gate()
    add_gate_qubit_validation()
    gate()
    copy()
    subset()
    concatenate()
    deadjoin()
    adjoin()
    test_reversed()
    test_reversed()
    is_equivalent()
    nonredundant()
    compressed()
    subcircuit()
    add_circuit()
    sort_gates()
    is_equivalent_order()
    # Not all fucntions above called here 
