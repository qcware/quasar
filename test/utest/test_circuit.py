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
    
    
def compute_1pdm():
    """
    Valid "Circuit.compute_1pdm()"
    """
    circuit = quasar.Circuit(N=1)
    wfn = np.array([0,1], dtype=np.complex128)
    dm = circuit.compute_1pdm(wfn,wfn,0)
    ans = [[0,0],[0,1]]

    return L1_error([(dm, ans)])

    
def compute_1pdm_format():
    """
    Valid the shape and dtype "Circuit.compute_1pdm()"
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
    Valid "Circuit.compute_2pdm()"
    """
    circuit = quasar.Circuit(N=2)
    wfn = np.array([0,1,0,0], dtype=np.complex128)
    dm = circuit.compute_2pdm(wfn,wfn,0,1)
    ans = [[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]
    
    return L1_error([(dm, ans)])

    
def compute_2pdm_format():
    """
    Valid the shape and dtype "Circuit.compute_2pdm()"
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
    Valid "Circuit.compute_3pdm()"
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
    Valid the shape and dtype "Circuit.compute_3pdm()"
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
    Valid "Circuit.compute_4pdm()"
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
    Valid the shape and dtype "Circuit.compute_4pdm()"
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
    Valid "Circuit.compute_npdm()"
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
    Valid the shape and dtype "Circuit.compute_npdm()"
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
    Valid "Circuit.compute_pauli_1()"
    """
    circuit = quasar.Circuit(N=1)
    wfn = np.array([0,1], dtype=np.complex128)
    expectation = circuit.compute_pauli_1(wfn,0)
    ans = [1,0,0,-1]
    
    return L1_error([(expectation, ans)])
    
    
def compute_pauli_2():
    """
    Valid "Circuit.compute_pauli_2()"
    """
    circuit = quasar.Circuit(N=2)
    wfn = np.array([0,0,1,0], dtype=np.complex128)
    expectation = circuit.compute_pauli_2(wfn,0,1)
    ans = [[1,0,0,1],[0,0,0,0],[0,0,0,0],[-1,0,0,-1]]

    return L1_error([(expectation, ans)])
    
    
def compute_pauli_3():
    """
    Valid "Circuit.compute_pauli_3()"
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
    Valid "Circuit.compute_pauli_4()"
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
    Valid "Circuit.compute_pauli_n()"
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
        

# def measure():
    # """
    # Valid "Circuit.measure()"
    # """
    # circuit, ans = simple_circuit(0)
    # result = circuit.measure()
    # result = [result.get(k) for ]
    # print(result)
    # print(ans)
    
    
    
    
    
    
    
    
# measure()
    
    
if __name__ == "__main__()":
    simulate()
    simulate_steps()
    apply_gate_1()
    apply_gate_1_format()
    apply_gate_2()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    