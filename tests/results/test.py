import quasar

if __name__ == '__main__':

    # GHZ circuit
    circuit = quasar.Circuit(N=3)
    circuit.add_gate(T=0, key=0, gate=quasar.Gate.H)
    circuit.add_gate(T=1, key=(0,1), gate=quasar.Gate.CX)
    circuit.add_gate(T=2, key=(1,2), gate=quasar.Gate.CX)
    print(circuit)
    
    # X1
    # circuit = quasar.Circuit(N=3)
    # circuit.add_gate(T=0, key=2, gate=quasar.Gate.X)
    # print(circuit)

    backend = quasar.QiskitSimulatorBackend() 
    print(backend.native_circuit(circuit))
    counts = backend.run_measurement(circuit, nmeasurement=8192)
    print(counts)

    ket = list(counts.keys())[0]
    print(ket[0])
    print(ket[1])
    print(ket[2])
    
    print(counts.string(N=circuit.N))
