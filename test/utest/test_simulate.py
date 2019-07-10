import quasar
import numpy as np
from util.circuit_generator import random_circuit, simple_circuit

"""
Test "Circuit.simulate()"
"""

def test_simulate():
    """
    Test "Circuit.simulate()"
    """
    # run simulate() with simple_circuit(0)
    circuit, ans = simple_circuit(0)
    wfn = circuit.simulate()
    error1 = np.subtract(wfn, ans)
    error1 = np.sum(np.abs(error1))
    # run simulate() with simple_circuit(1)
    circuit, ans = simple_circuit(1)
    wfn = circuit.simulate()
    error2 = np.subtract(wfn, ans)
    error2 = np.sum(np.abs(error2))
    error = error1+error2
    return error<10-4

    
def test_simulate_steps():
    """
    Test "Circuit.simulate_steps()"
    """
    error = 0
    circuit, _ = simple_circuit(0)
    
    wfn0, wfn1 = circuit.simulate_steps()
    error += np.sum(np.abs(np.subtract(wfn0[1], [np.sqrt(1/2),0,np.sqrt(1/2),0] )))
    error += np.sum(np.abs(np.subtract(wfn1[1], [np.sqrt(1/2),0,0,np.sqrt(1/2)] )))
    return error<10-4

    
if __name__ == "__main__()":
    test_simulate()
    test_simulate_steps()

# test_simulate_steps()