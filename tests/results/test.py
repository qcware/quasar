import tomcat

if __name__ == '__main__':

    # GHZ circuit
    circuit = tomcat.Circuit(N=3)
    circuit.add_gate(T=0, key=0, gate=tomcat.Gate.H)
    circuit.add_gate(T=1, key=(0,1), gate=tomcat.Gate.CX)
    circuit.add_gate(T=2, key=(1,2), gate=tomcat.Gate.CX)
    print(circuit)
    
    # X1
    # circuit = tomcat.Circuit(N=3)
    # circuit.add_gate(T=0, key=2, gate=tomcat.Gate.X)
    # print(circuit)

    backend = tomcat.QiskitAerBackend() 
    print(backend.emit_circuit(circuit))
    counts = backend.run_shots(circuit, shots=8192)
    print(counts)

    ket = list(counts.keys())[0]
    print(ket[0])
    print(ket[1])
    print(ket[2])
    
    print(counts.string(N=circuit.N))
