import quasar
import numpy as np
from util.error import L1_error
from util.circuit_generator import random_circuit

"""
Test "ForestBackend" Class
"""

def angle():
    """
    Validate quasar_to_forest_angle() and forest_to_quasar_angle() of the "ForestBackend" class. 
    """
    backend = quasar.ForestBackend()
    angle1 = backend.quasar_to_forest_angle(10)
    
    backend = quasar.ForestBackend()
    angle2 = backend.forest_to_quasar_angle(10)

    return angle1==20 and angle2==5


def build_native_circuit():
    """
    Validate build_native_circuit() of the "ForestBackend" class. 
    """
    # translate circuit from quasar circuit
    quasar_circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    forest_backend = quasar.ForestBackend()
    forest_circuit = forest_backend.build_native_circuit(quasar_circuit)
    # directly build a forest circuit
    import pyquil
    bell = pyquil.Program()
    bell += pyquil.gates.H(0)
    bell += pyquil.gates.CNOT(0,1)
    
    return forest_circuit==bell


def build_quasar_circuit():
    """
    Validate build_quasar_circuit() of the "ForestBackend" class. 
    """
    # translate circuit from forest circuit
    import pyquil
    forest_circuit = pyquil.Program()
    forest_circuit += pyquil.gates.H(0)
    forest_circuit += pyquil.gates.CNOT(0,1)
    forest_backend = quasar.ForestBackend()
    quasar_circuit = forest_backend.build_quasar_circuit(forest_circuit)
    # directly build a quasar circuit
    bell = quasar.Circuit(N=2).H(0).CX(0,1)

    return quasar_circuit==bell

    
def build_native_circuit_in_basis():
    """
    Validate build_native_circuit_in_basis() of the "ForestBackend" class. 
    """
    # translate circuit from quasar circuit
    quasar_circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    forest_backend = quasar.ForestBackend()
    forest_circuit = forest_backend.build_native_circuit_in_basis(quasar_circuit,['X','Z'])
    # directly build a forest circuit
    import pyquil
    bell = pyquil.Program()
    bell += pyquil.gates.H(0)
    bell += pyquil.gates.CNOT(0,1)
    bell += pyquil.gates.H(0)
    
    return forest_circuit==bell


def build_native_circuit_measurement():
    """
    Validate build_native_circuit_measurement() of the "ForestBackend" class. 
    """
    # translate circuit from quasar circuit
    quasar_circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    forest_backend = quasar.ForestBackend()
    forest_circuit = forest_backend.build_native_circuit_measurement(quasar_circuit)
    # directly build a forest circuit
    import pyquil
    bell = pyquil.Program()
    bell += pyquil.gates.H(0)
    bell += pyquil.gates.CNOT(0,1)
    bell.measure_all()
    
    return forest_circuit==bell


"""
Test "ForestSimulatorBackend" Class
"""
def util_build_circuit():
    """
    A utility function for the testing functions below.
    It creates a quasar Circuit object.
    """
    circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    
    return circuit


def forest_simulator_backend():
    """
    Validate initialization of the "ForestSimulatorBackend" class.
    """
    backend = quasar.ForestSimulatorBackend('2q-qvm')
    
    if not backend.has_statevector: return False
    if not backend.has_measurement: return False
    
    return True

    
def run_statevector():
    """
    Validate run_statevector() in the "ForestSimulatorBackend" class.
    """
    backend = quasar.ForestSimulatorBackend('2q-qvm')
    # test case 1: Bell's state
    circuit = util_build_circuit()
    result = backend.run_statevector(circuit)
    ans = [np.sqrt(1/2),0,0,np.sqrt(1/2)]
    # test case 2: Random circuit
    circuit = random_circuit(seed=5566, depth=10)
    result = backend.run_statevector(circuit)
    ans = circuit.simulate()
    
    return L1_error((result, ans))   
   
   
def run_measurement():
    """
    Validate run_measurement() in the "ForestSimulatorBackend" class.
    """
    backend = quasar.ForestSimulatorBackend('2q-qvm')
    circuit = util_build_circuit()
    result = backend.run_measurement(circuit)
    ans = [np.sqrt(1/2),0,0,np.sqrt(1/2)]
    
    # these two lines normalize the result
    result = [result.get('{0:02b}'.format(k), 0) for k in range(4)]
    result = np.array(result)/np.linalg.norm(result)
    
    return L1_error((result, ans), margin=0.1)


"""
Test "ForestHardwareBackend" Class
Hardware test not implemented yet
"""

# def forest_hardware_backend():
    # """
    # Validate initialization of the "ForestHardwareBackend" class.
    # """
    # backend = quasar.ForestHardwareBackend()
    
    # if backend.has_statevector: return False
    # if not backend.has_measurement: return False
    
    # return True
    
    








