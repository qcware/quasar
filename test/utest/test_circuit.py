import quasar
import numpy as np
from util.circuit_generator import random_circuit, simple_circuit
from util.error import L1_error

"""
Test "Circuit.simulate()"
"""

def simulate():
    """
    Test "Circuit.simulate()"
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
    Test "Circuit.simulate_steps()"
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
    Test "Circuit.apply_gate_1()"
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
    Test "Circuit.apply_gate_1()"
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
    Valid "Circuit.apply_gate_2()"
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
    Valid "Circuit.apply_gate_3()"
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
    
    
    
    
    
    
    
    
    
if __name__ == "__main__()":
    simulate()
    simulate_steps()
    apply_gate_1()
    apply_gate_1_format()
    apply_gate_2()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    