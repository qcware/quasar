import quasar
import numpy as np
from util.circuit_generator import random_circuit, simple_circuit
from util.error import L1_error

"""
Test "Circuit.simulate()"
"""

def test_simulate():
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

    
def test_simulate_steps():
    """
    Test "Circuit.simulate_steps()"
    """
    circuit, _ = simple_circuit(0)
    result = []
    wfn0, wfn1 = circuit.simulate_steps()
    result.append((wfn0[1], [np.sqrt(1/2),0,np.sqrt(1/2),0] ))
    result.append((wfn1[1], [np.sqrt(1/2),0,0,np.sqrt(1/2)] ))
    return L1_error(result)

    
def test_apply_gate_1():
    """
    Test "Circuit.apply_gate_1()"
    """
    result = []
    # test case 1
    U = quasar.Matrix.X
    circuit = quasar.Circuit(N=1)
    initial_wfn = np.array([1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_1(initial_wfn, np.zeros_like(initial_wfn), U, 0)
    result.append((wfn, [0, 1]))
    # test case 2
    U = quasar.Matrix.H
    circuit = quasar.Circuit(N=1)
    initial_wfn = np.array([1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_1(initial_wfn, np.zeros_like(initial_wfn), U, 0)
    result.append((wfn, [np.sqrt(1/2), np.sqrt(1/2)]))
    # test case 2
    U = quasar.Matrix.Rx(theta=np.pi/6)
    circuit = quasar.Circuit(N=1)
    initial_wfn = np.array([1,0], dtype=np.complex128)
    wfn = circuit.apply_gate_1(initial_wfn, np.zeros_like(initial_wfn), U, 0)
    result.append((wfn, [np.sqrt(3)/2, -1j/2]))
    
    return L1_error(result)
    
    
    
    
    
    
if __name__ == "__main__()":
    test_simulate()
    test_simulate_steps()

test_apply_gate_1()