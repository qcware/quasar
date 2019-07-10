import quasar
import numpy as np
import random
import math


def random_circuit(seed=2019, N=3, depth=6, type='all'):
    gate_list = get_gete_list(type)
    
    circuit = quasar.Circuit(N=N)
    for d in range(depth):
        gate = np.random.choice(gate_list)
        add_gate(gate, circuit)
    
    return circuit

    
def simple_circuit(index):
    if index==0:
        # Bell State
        circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    if index==1:
        # GHZ State
        circuit = quasar.Circuit(N=3).H(0).CX(0,1).CX(1,2)
    
    
    return circuit
    
    
    
# => Static Function <= 
def get_gete_list(type):
    if type == 'all':
        gate_list = ['I','X','Y','Z','S','T','H','RX','RY','RZ','CX','CZ','SWAP']
    elif type == '1qubit':
        gate_list = ['I','X','Y','Z','S','T','H','RX','RY','RZ']
    elif type == '1qubit-with-param':
        gate_list = ['RX','RY','RZ']
    elif type == '1qubit-without-param':
        gate_list = ['I','X','Y','Z','S','T','H']
    elif type == '2qubit':
        gate_list = ['CX','CZ','SWAP']
    return gate_list
    

def add_gate(gate, circuit):
    N = circuit.N
    
    if gate in ['I','X','Y','Z','S','T','H']:
        qubit = np.random.randint(0,N)
        if gate == 'I':
            circuit.I(qubit)
        elif gate == 'X':
            circuit.X(qubit)
        elif gate == 'Y':
            circuit.Y(qubit)
        elif gate == 'Z':
            circuit.Z(qubit)
        elif gate == 'H':
            circuit.H(qubit)
        elif gate == 'S':
            circuit.S(qubit)
        elif gate == 'T':
            circuit.T(qubit)
     
    elif gate in ['RX','RY','RZ']:
        qubit = np.random.randint(0,N)
        param = np.random.uniform(0,2*math.pi)
        
        if gate == 'RX':
            circuit.Rx(qubit, theta=param)
        elif gate == 'RY':
            circuit.Ry(qubit, theta=param)
        elif gate == 'RZ':
            circuit.Rz(qubit, theta=param)
            
    elif gate in ['CX','CY','CZ','SWAP']:
        qubitA, qubitB = random.sample(range(N),2)
        
        if gate == 'CX':
            circuit.CX(qubitA, qubitB)
        elif gate == 'CY':
            circuit.CY(qubitA, qubitB)
        elif gate == 'CZ':
            circuit.CZ(qubitA, qubitB)
        elif gate == 'SWAP':
            circuit.SWAP(qubitA, qubitB)
            
            
            
            
            
# circuit = random_circuit_generator(N=3, depth=6)
# print(circuit)
            
            
            
            