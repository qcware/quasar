import quasar
import numpy as np
import collections

def init_gate():
    # Initialize an X gate
    npx = np.array([[0,1], [1,0]])
    xfun = lambda dummy: npx
    custom_x = quasar.Gate(N=1, Ufun=xfun, params=collections.OrderedDict(),
                            name="X", ascii_symbols=['X'])

    circ = quasar.Circuit(1)
    circ = circ.add_gate(custom_x, 0, 0) 

    if custom_x.name != "X":
        return False
    if not np.array_equal(custom_x.U, npx):
        return False
    if not np.array_equal(circ.simulate().real, np.array([0., 1.])):
        return False

    return True

def init_param_gate():
    # Initialize a parameterized gate
    params=collections.OrderedDict()
    theta = np.pi
    params["theta"] = theta 
    ryfun = lambda p: np.array([[np.cos(p["theta"]), -np.sin(p["theta"])],
                                    [np.sin(p["theta"]), np.cos(p["theta"])]])
    custom_ry = quasar.Gate(N=1, Ufun=ryfun, params=params,
                            name='Ry', ascii_symbols=['Ry'])

    if custom_ry.name != "Ry":
        return False
    circ = quasar.Circuit(1)
    circ = circ.add_gate(custom_ry, 0, 0) 
    
    if not custom_ry.same_unitary(quasar.Gate.Ry(theta)):
        return False
    if not np.allclose(circ.simulate().real, np.array([-1., 0.])):
        return False

    return True

def same_unitary1():
    # Check a predefined gate
    if not quasar.Gate.X.same_unitary(quasar.Gate.X):
        return False
    if quasar.Gate.Y.same_unitary(quasar.Gate.X):
        return False

    return True

def same_unitary2():
    # Check a custom gate
    npx = np.array([[0,1], [1,0]])
    xfun = lambda dummy: npx
    custom_x = quasar.Gate(N=1, Ufun=xfun, params=collections.OrderedDict(),
                            name="X", ascii_symbols=['X'])
    
    if not custom_x.same_unitary(quasar.Gate.X):
        return False

    return True

def same_unitary3():
    # Check a parameterized gate
    theta = np.pi/4
    if not quasar.Gate.Rx(theta).same_unitary(quasar.Gate.Rx(theta)):
        return False
    if quasar.Gate.Ry(theta).same_unitary(quasar.Gate.Rx(theta)):
        return False
    
    return True

def U():
    # Ensure it's creating the proper unitaries for non-parametric gates
    y_gate = np.array([[0, -1j], [1j, 0]])
   
    if not np.allclose(y_gate, quasar.Gate.Y.U):
        return False
    
    # 2 qubit gate
    cx_gate = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [0,0,1,0]])
    
    if not np.allclose(cx_gate, quasar.Gate.CX.U):
        return False

    return True

def U1():
    # for parametric
    theta = np.pi/4
    rx_pi4 = np.array([[np.cos(theta), -1j*np.sin(theta)],
                       [-1j*np.sin(theta), np.cos(theta)]])
    
    if np.allclose(rx_pi4, quasar.Gate.Rx(theta).U):
        return True
    else:
        return False

def copy():
    theta = np.pi/4
    rx1 = quasar.Gate.Rx(theta)
    rx2 = rx1.copy()
    if not rx1.same_unitary(rx2):
        return False
    # Changin params should change rx2 and not rx1
    rx2.set_params({"theta": np.pi})
    if rx1.same_unitary(rx2):
        return False
    
    return True

def set_params(): 
    # Changing parameters of an intialized parametric gate
    theta = np.pi/4
    rx = quasar.Gate.Rx(theta)
    # Set theta = pi
    rx.set_params({"theta": np.pi})

    rx_pi_unitary = np.array([[-1, 0],
                              [0, -1]])

    if np.allclose(rx.U, rx_pi_unitary):
        return True
    else:
        return False
   
def set_params1():
    # should have no affect on non-parametric gate
    theta = np.pi
    x = quasar.Gate.X
    # artifically adding parameters
    x.params = {"a": 0}
    x.set_params({"a": theta})
    
    x_gate = np.array([[0, 1], 
                       [1, 0]])

    if np.allclose(x.U, x_gate):
        return True
    else:
        return False

if __name__ == "__main__":
    init_gate()
    init_param_gate()
    same_unitary1()
    same_unitary2()
    same_unitary3()
    U()
    U1()
    copy()
    set_params()
    print(set_params1())
