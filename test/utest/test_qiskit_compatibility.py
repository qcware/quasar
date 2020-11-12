from quasar.testing_strategies.circuits import circuits, allowed_gates
from quasar import QiskitSimulatorBackend, Gate
from hypothesis import given, settings, note

disallowed_gates = {
    Gate.CS, Gate.ST, Gate.TT, Gate.Rx2, Gate.CST, Gate.Rx2T, Gate.CCX,
    Gate.CSWAP, Gate.XX_ion, Gate.Rx_ion, Gate.Ry_ion, Gate.R_ion, Gate.CF,
    Gate.Rz_ion, Gate.u1, Gate.u2, Gate.u3, Gate.iRBS, Gate.SO42, Gate.SO4
}

gate_list = list(set(allowed_gates) - disallowed_gates)


@given(
    circuits(min_qubits=4,
             max_qubits=4,
             min_length=5,
             max_length=10,
             allowed_gates=gate_list,
             use_all_allowed_gates=False))
@settings(deadline=None, max_examples=100)
def test_qiskit_compatibility(circuit):
    backend = QiskitSimulatorBackend()
    note(f"Circuit: {str(circuit)}")
    qc = backend.build_native_circuit(circuit)
    assert True
