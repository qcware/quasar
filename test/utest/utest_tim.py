import unittest
import simple_test
import test_circuit


class Test(unittest.TestCase):
    def test_test(self):
        # should always be true
        self.assertTrue(simple_test.test())
    def test_test1(self, one=1):
        # First statement should return True. Second should return False.
        self.assertTrue(simple_test.test1(1))
        # self.assertTrue(simple_test.test1(2))
    
    # ==> Test Circuit class <==
    def test_circuit_simulate(self):
        self.assertTrue(test_circuit.simulate())
    def test_circuit_simulate_steps(self):
        self.assertTrue(test_circuit.simulate_steps())
    def test_circuit_apply_gate_1(self):     
        self.assertTrue(test_circuit.apply_gate_1())
        self.assertTrue(test_circuit.apply_gate_1_format())
    def test_circuit_apply_gate_2(self):
        self.assertTrue(test_circuit.apply_gate_2())
        self.assertTrue(test_circuit.apply_gate_2_format())
    def test_circuit_apply_gate_3(self):
        self.assertTrue(test_circuit.apply_gate_3())
        self.assertTrue(test_circuit.apply_gate_3_format())
    def test_circuit_compute_1pdm(self):
        self.assertTrue(test_circuit.compute_1pdm())
        self.assertTrue(test_circuit.compute_1pdm_format())
    def test_circuit_compute_2pdm(self):
        self.assertTrue(test_circuit.compute_2pdm())
        self.assertTrue(test_circuit.compute_2pdm_format())
    def test_circuit_compute_3pdm(self):
        self.assertTrue(test_circuit.compute_3pdm())
        self.assertTrue(test_circuit.compute_3pdm_format())
    def test_circuit_compute_4pdm(self):
        self.assertTrue(test_circuit.compute_4pdm())
        self.assertTrue(test_circuit.compute_4pdm_format())
    def test_circuit_compute_npdm(self):
        self.assertTrue(test_circuit.compute_npdm())
        self.assertTrue(test_circuit.compute_npdm_format())
    def test_circuit_compute_pauli_1(self):
        self.assertTrue(test_circuit.compute_pauli_1())
    def test_circuit_compute_pauli_2(self):
        self.assertTrue(test_circuit.compute_pauli_2())
    def test_circuit_compute_pauli_3(self):
        self.assertTrue(test_circuit.compute_pauli_3())
    def test_circuit_compute_pauli_4(self):
        self.assertTrue(test_circuit.compute_pauli_4())
    def test_circuit_compute_pauli_n(self):
        self.assertTrue(test_circuit.compute_pauli_n())
    def test_circuit_measure(self):
        self.assertTrue(test_circuit.measure())
    def test_circuit_compute_measurements_from_statevector(self):
        self.assertTrue(test_circuit.compute_measurements_from_statevector())
    def test_circuit_nparam(self):
        self.assertTrue(test_circuit.nparam())
    def test_circuit_param_keys(self):
        self.assertTrue(test_circuit.param_keys())
    def test_circuit_param_values(self):
        self.assertTrue(test_circuit.param_values())
    def test_circuit_set_param_values(self):
        self.assertTrue(test_circuit.set_param_values())
    def test_circuit_params(self):
        self.assertTrue(test_circuit.params())    
    def test_circuit_set_param(self):
        self.assertTrue(test_circuit.set_param())
    def test_circuit_param_str(self):
        self.assertTrue(test_circuit.param_str())    
        
    
if __name__ == '__main__':
    unittest.main()
