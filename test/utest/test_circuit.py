import quasar
import numpy as np
from util.circuit_generator import random_circuit, simple_circuit
from util.error import L1_error

"""
Test "Circuit.simulate()"
"""

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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    