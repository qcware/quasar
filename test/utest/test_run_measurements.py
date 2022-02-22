from quasar import Circuit, QuasarSimulatorBackend
import pytest

def test_run_measurements():
    # basic brute-force test here to check validity
    # of nmeasurement duplication
    circuit = Circuit().H(0).CX(0,1).X(2)
    num_circuits = 3
    circuits = [circuit] * num_circuits

    # should get three ProbabilityHistograms back
    # with nmeasurement as int, None, or list equal to number
    # of circuits
    backend = QuasarSimulatorBackend()
    result_1 = backend.run_measurements(circuits, nmeasurement=1)
    assert len(result_1) == num_circuits
    result_2 = backend.run_measurements(circuits, nmeasurement=None)
    assert len(result_2) == num_circuits
    result_3 = backend.run_measurements(circuits, nmeasurement=[1]*num_circuits)
    assert len(result_3) == num_circuits
    
    # should have default behaviour of zip and return length the shortest of the two sequences
    result_4 = backend.run_measurements(circuits, nmeasurement=[1]*(num_circuits-1))
    assert len(result_4) == num_circuits -1
    result_5 = backend.run_measurements(circuits, nmeasurement=[1]*(num_circuits+1))
    assert len(result_5) == num_circuits
