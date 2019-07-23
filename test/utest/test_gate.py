"""
Tests for the Gate and ControlledGate classes in circuit.py
"""

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
    x.params = collections.OrderedDict([("a", 0)])
    x.set_params({"a": theta})
    
    x_gate = np.array([[0, 1], 
                       [1, 0]])

    if np.allclose(x.U, x_gate):
        return True
    else:
        return False

# next is set_param (singular)
def set_param():
    # Changing parameters of an intialized parametric gate
    theta = np.pi/4
    rx = quasar.Gate.Rx(theta)
    # Set theta = pi
    rx.set_param("theta", np.pi)

    rx_pi_unitary = np.array([[-1, 0],
                              [0, -1]])

    if np.allclose(rx.U, rx_pi_unitary):
        return True
    else:
        return False
   
def set_param1():
    # should have no affect on non-parametric gate
    theta = np.pi
    x = quasar.Gate.X
    # artifically adding parameters
    x.params = {"a": 0}
    x.set_param("a", theta)
    
    x_gate = np.array([[0, 1], 
                       [1, 0]])

    if np.allclose(x.U, x_gate):
        return True
    else:
        return False

"""
Choosing not to test the predefined gate member and methods whose matrix representation rely on
already tested methods in the Matrix class.
"""

def gateRx():
    # Gate.Rx is the method in _GateRx in gate class
    thetas = [0, np.pi/4, np.pi/2, np.pi, -np.pi, 2*np.pi, 3*np.pi, 8*np.pi/7]
   
    for theta in thetas:
        rx = np.array([[np.cos(theta), -1j*np.sin(theta)], 
                   [-1j*np.sin(theta), np.cos(theta)]])

        if not np.allclose(quasar.Gate.Rx(theta).U, rx):
            return False
    
    return True

def gateRy():
    # Gate.Rx is the method in _GateRx in gate class
    thetas = [0, np.pi/4, np.pi/2, np.pi, -np.pi, 2*np.pi, 3*np.pi, 8*np.pi/7]
   
    for theta in thetas:
        ry = np.array([[np.cos(theta), -1*np.sin(theta)], 
                   [1*np.sin(theta), np.cos(theta)]])
        
        if not np.allclose(quasar.Gate.Ry(theta).U, ry):
            return False
    
    return True

def gateRz():
    # Gate.Rx is the method in _GateRx in gate class
    thetas = [0, np.pi/4, np.pi/2, np.pi, -np.pi, 2*np.pi, 3*np.pi, 8*np.pi/7]
   
    for theta in thetas:
        rz = np.array([[np.exp(-1j*theta), 0], 
                        [0, np.exp(1j*theta)]])
        
        if not np.allclose(quasar.Gate.Rz(theta).U, rz):
            return False
    
    return True

def gateSO4():
    # Testing basics
    A = 1
    B = 1
    C = 1
    D = 1
    E = 1
    F = 1
    so4 = quasar.Gate.SO4(A, B, C, D, E, F)

    if so4.N != 2:
        return False
    if so4.U.shape != (4,4):
        return False
    if so4.name != "SO4":
        return False

    return True

def gateSO42():
    # Testing basics
    thetaIY = 1
    thetaYI = 1
    thetaXY = 1
    thetaYX = 1
    thetaZY = 1
    thetaYZ = 1
    so42 = quasar.Gate.SO42(thetaIY, thetaYI, thetaXY, thetaYX, thetaZY, thetaYZ)

    if so42.N != 2:
        return False
    if so42.U.shape != (4,4):
        return False
    if so42.name != "SO42":
        return False

    return True

def gateCF():
    thetas = [0, np.pi/4, np.pi/2, np.pi, -np.pi, 2*np.pi, 3*np.pi, 8*np.pi/7]

    for theta in thetas:
        cf = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0, np.cos(theta), np.sin(theta)],
            [0,0, np.sin(theta), -np.cos(theta) ]])

        if not np.allclose(quasar.Gate.CF(theta).U, cf):
            return False
   
    if quasar.Gate.CF(theta).N !=2:
        return False
    if quasar.Gate.CF(theta).name != "CF":
        return False

    return True

# testing explict gates
def gateU1():
    eye = np.eye(2)
    custom_u = eye 
    u1 = quasar.Gate.U1(custom_u)

    if not np.allclose(u1.U, eye):
        return False
    if u1.name != "U1":
        return False
    
    return True

def gateU2():
    eye4 = np.eye(4)
    u2 = quasar.Gate.U2(eye4)

    if not np.allclose(u2.U, eye4):
        return False
    if u2.name != "U2":
        return False
    
    return True

def init_controlled_gate():
    x = quasar.Gate.X
    cx = quasar.ControlledGate(x, ncontrol=1)

    cx_matrix = np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [0,0,1,0]])

    if cx.N != 2:
        return False
    if cx.ncontrol != 1:
        return False
    if not np.allclose(cx_matrix, cx.U):
        return False
    
    crx = quasar.ControlledGate(quasar.Gate.Rx(np.pi), ncontrol=2)

    if not "theta" in crx.params.keys():
        return False

    # testing copy and set param
    crx2 = crx.copy()
    crx2.set_param("theta", 0)
    if np.allclose(crx2.U, crx.U):
        return False
    
    return True

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
    set_params1()
    set_param()
    set_param1()
    gateRx()
    gateRy()
    gateRz()
    gateSO4()
    gateSO42()
    gateCF()
    gateU1()
    gateU2()
    init_controlled_gate()
