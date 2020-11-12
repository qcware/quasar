import qiskit
import numpy


def trim(a: numpy.array) -> numpy.array:
    eps = 0.000001
    a.real[abs(a.real) < eps] = 0
    a.imag[abs(a.imag) < eps] = 0
    return a*a


def show_before_after(circuit):
    sv_sim = qiskit.BasicAer.get_backend('statevector_simulator')
    c1 = circuit.copy()
    sv0 = trim(qiskit.execute(c1, sv_sim).result().get_statevector())
    crxx = circuit.copy()
    crxx.rxx(0, 0, 1)
    svrxx = trim(qiskit.execute(crxx, sv_sim).result().get_statevector())

    cryy = circuit.copy()
    cryy.ryy(0, 0, 1)
    svryy = trim(qiskit.execute(cryy, sv_sim).result().get_statevector())
    print("Rxx(theta=0)")
    print(f"{sv0} -> {svrxx}")
    print("Ryy(theta=0)")
    print(f"{sv0} -> {svryy}")


if __name__ == "__main__":
    blank_circuit = qiskit.QuantumCircuit(qiskit.QuantumRegister(2))
    show_before_after(blank_circuit.copy())
    cx0 = blank_circuit.copy()
    cx0.x(0)
    show_before_after(cx0)

    cx1 = blank_circuit.copy()
    cx1.x(1)
    show_before_after(cx1)

    cx01 = blank_circuit.copy()
    cx01.x(0)
    cx01.x(1)
    show_before_after(cx01)
